# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""GRPO training loop for multi-turn tool-calling rollouts.

Each step:

  1. **Rollout.** The hybrid engine switches to inference mode and the policy
     solves each rank's prompts, calling the code interpreter between turns
     (:class:`~deepspeed.runtime.rollout.HybridEngineRollout`).
  2. **Reward.** Every trajectory gets the strict-box answer score, and its
     advantage is the score normalized within its prompt's group
     (:func:`~deepspeed.runtime.rollout.group_advantages`).
  3. **Update.** One optimizer step over all trajectories, accumulated over
     micro-batches: PPO-clip with clip-higher on the generated tokens
     (:func:`~deepspeed.runtime.rollout.grpo_policy_loss`), averaged over every
     generated token across ranks, or per trajectory first with
     ``algorithm.loss_aggregation = "sample_mean"``. Tool output is masked out
     (:class:`~deepspeed.runtime.rollout.ResponseTokenBatch`).

With one update per rollout, the old policy is the current one, so by default
the old log-probs are the current ones detached; the importance ratio is 1 and
the loss reduces to the policy gradient. ``rollout.return_log_probs``
takes them from the rollout instead
(:func:`~deepspeed.runtime.rollout.old_log_probs_from_rollout`).
"""

import os
import re

import torch
from deepspeed import comm as dist
from deepspeed.accelerator import get_accelerator

from config import GRPOConfig
from reward import retool_reward
from deepspeed.runtime.rollout import (ResponseTokenBatch, RolloutBatch, RolloutEngine, RolloutRequest, SamplingConfig,
                                       group_advantages, grpo_policy_loss, old_log_probs_from_rollout)
from deepspeed.runtime.rollout.base import RolloutRecord


def _is_rank_zero() -> bool:
    return (not dist.is_initialized()) or dist.get_rank() == 0


def _mean_across_ranks(values: dict) -> dict:
    """Average each metric over ranks with one all-reduce."""
    names = sorted(values)
    tensor = torch.tensor([float(values[name]) for name in names], device=get_accelerator().current_device_name())
    dist.all_reduce(tensor)
    tensor /= dist.get_world_size()
    return dict(zip(names, tensor.tolist()))


class GRPOTrainer:

    def __init__(
        self,
        cfg: GRPOConfig,
        engine,
        tokenizer,
        rollout: RolloutEngine,
        dataloader,
        eval_rows=None,
    ):
        self.cfg = cfg
        self.engine = engine
        self.tokenizer = tokenizer
        self.rollout = rollout
        self.dataloader = dataloader
        self.eval_rows = eval_rows or []

        self.device = get_accelerator().current_device_name()
        self.pad_token_id = tokenizer.pad_token_id
        self.step = 0
        self.sampling = SamplingConfig(
            max_new_tokens=cfg.rollout.max_new_tokens,
            max_new_tokens_per_turn=cfg.rollout.max_new_tokens_per_turn,
            temperature=cfg.rollout.temperature,
            n_samples_per_prompt=cfg.rollout.n_samples_per_prompt,
            repetition_penalty=cfg.rollout.repetition_penalty,
        )
        self.eval_sampling = SamplingConfig(
            max_new_tokens=cfg.eval.max_new_tokens,
            max_new_tokens_per_turn=cfg.eval.max_new_tokens_per_turn,
            temperature=cfg.eval.temperature,
            n_samples_per_prompt=cfg.eval.n_samples_per_prompt,
            repetition_penalty=cfg.rollout.repetition_penalty,
        )

    # ------------------------------------------------------------------
    # Driver
    # ------------------------------------------------------------------

    def resume(self, checkpoint_dir: str) -> None:
        """Load the latest checkpoint in ``checkpoint_dir``: weights, optimizer state and the step it was saved at."""
        load_path, _ = self.engine.load_checkpoint(checkpoint_dir)
        if load_path is None:
            raise FileNotFoundError(f"No checkpoint to resume from in {checkpoint_dir}")
        # load_path is a file inside the checkpoint's tag directory, and tags are step_<N> (see _save).
        tag = re.search(r"(?:^|/)step_(\d+)(?:/|$)", load_path)
        if tag is None:
            raise ValueError(f"Cannot read the step from checkpoint path {load_path}")
        self.step = int(tag.group(1))
        if _is_rank_zero():
            print(f"[grpo] resumed from {load_path} at step {self.step}", flush=True)

    def train(self) -> None:
        # A resumed run picks up the data order where its checkpoint left off.
        epoch, skip = divmod(self.step, len(self.dataloader))
        while self.step < self.cfg.training.num_rollouts:
            # Reseed the DistributedSampler so each epoch reshuffles each rank's shard.
            self.dataloader.sampler.set_epoch(epoch)
            for rows in self.dataloader:
                if skip:
                    skip -= 1
                    continue
                if self.step >= self.cfg.training.num_rollouts:
                    break
                if self._eval_due():
                    self._log(self.evaluate(), prefix="eval")
                metrics = self._train_step(rows)
                self.step += 1
                if self.step % self.cfg.training.logging_steps == 0:
                    self._log(metrics, prefix="train")
                if self.step % self.cfg.training.save_steps == 0:
                    self._save()
            epoch += 1
        if self.cfg.eval.path and self.cfg.eval.at_end:
            self._log(self.evaluate(), prefix="eval")
        if self.step % self.cfg.training.save_steps:
            self._save()

    def _eval_due(self) -> bool:
        """Evaluate before training starts and every ``eval.interval`` steps."""
        return bool(self.cfg.eval.path) and self.cfg.eval.interval > 0 and self.step % self.cfg.eval.interval == 0

    # ------------------------------------------------------------------
    # Rollout and reward
    # ------------------------------------------------------------------

    def _rollout(self, rows, sampling: SamplingConfig) -> list:
        """Generate trajectories for ``rows`` in chunks, the hybrid engine in inference mode."""
        self.engine.eval()
        samples = []
        # A call decodes rollout.generate_batch_size trajectories, by default as many as a training step does
        # on this rank, so a step's rows roll out in one call. Evaluation's larger n gets fewer prompts per call.
        trajectories_per_call = self.cfg.rollout.generate_batch_size
        if trajectories_per_call is None:
            trajectories_per_call = self.cfg.training.global_batch_size // dist.get_world_size()
        prompts_per_chunk = max(1, trajectories_per_call // sampling.n_samples_per_prompt)
        for start in range(0, len(rows), prompts_per_chunk):
            request = RolloutRequest(samples=[
                RolloutRecord(group_index=row["index"], prompt=row["prompt"]) for row in rows[start:start + prompts_per_chunk]
            ])
            samples.extend(self.rollout.generate(request, sampling).samples)
        return samples

    def _score(self, samples, rows) -> list:
        """Each sample's reward, also stored on the sample."""
        labels = {row["index"]: row["label"] for row in rows}
        rewards = []
        for sample in samples:
            num_tool_calls = sample.response.count("<tool_call>")
            result = retool_reward(sample.response, labels[sample.group_index], num_tool_calls,
                                   self.cfg.algorithm.tool_call_bonus)
            sample.reward = result["score"]
            rewards.append(result["score"])
        return rewards

    @torch.no_grad()
    def evaluate(self) -> dict:
        """Accuracy over the eval set, averaged over every sample; its prompts are split across ranks."""
        rank, world_size = dist.get_rank(), dist.get_world_size()
        rows = self.eval_rows[rank::world_size]
        rewards = self._score(self._rollout(rows, self.eval_sampling), rows)
        # A correct answer scores +1; any wrong one, even with the tool bonus, stays below 0.
        num_correct = sum(reward > 0 for reward in rewards)
        totals = torch.tensor([num_correct, len(rewards)], dtype=torch.float, device=self.device)
        dist.all_reduce(totals)
        return {"accuracy": (totals[0] / totals[1].clamp(min=1)).item()}

    # ------------------------------------------------------------------
    # One step
    # ------------------------------------------------------------------

    def _train_step(self, rows) -> dict:
        samples = self._rollout(rows, self.sampling)
        rewards = self._score(samples, rows)
        advantages = group_advantages(samples, rewards).to(self.device)

        # Rollout and training take turns on the GPU: return each phase's cached memory before the other runs.
        get_accelerator().empty_cache()
        self.engine.train()
        loss = self._update(samples, advantages)
        get_accelerator().empty_cache()
        # Averaged over ranks, the per-rank losses are the step loss.
        return _mean_across_ranks({"loss": loss, "reward": sum(rewards) / len(rewards)})

    def _update(self, samples, advantages) -> float:
        """One optimizer step over ``samples``: PPO-clip, averaged over all ranks.

        Returns this rank's loss; averaged over ranks it is the step loss.
        """
        if self.cfg.algorithm.loss_aggregation == "sample_mean":
            # Each trajectory with a generated token weighs the same, however many it generated; one cut
            # before its first turn adds nothing, as grpo_policy_loss counts them.
            num_units = float(sum(1 for sample in samples if any(sample.loss_mask or [])))
        else:
            # Every generated token across ranks weighs the same.
            num_units = float(sum(sum(sample.loss_mask or []) for sample in samples))
        total_units = torch.tensor(num_units, device=self.device)
        dist.all_reduce(total_units)
        # grpo_policy_loss returns each micro-batch's unreduced loss_sum. DeepSpeed averages gradients over
        # ranks, so each rank's sum is scaled by world_size / total to give the step's mean.
        loss_scale = dist.get_world_size() / total_units.clamp(min=1).item()

        step_loss = 0.0
        # micro_batch_size counts trajectories across ranks.
        micro_batch_size = self.cfg.training.micro_batch_size // dist.get_world_size()
        for start in range(0, len(samples), micro_batch_size):
            rollout = RolloutBatch(samples=samples[start:start + micro_batch_size])
            batch = ResponseTokenBatch.from_rollout(rollout, pad_token_id=self.pad_token_id, device=self.device)
            # [B, T - 1] log-probs of batch.target_ids, aligned with batch.response_mask.
            log_probs = self.engine(input_ids=batch.input_ids,
                                    attention_mask=batch.attention_mask,
                                    temperature=self.cfg.rollout.temperature)
            if self.cfg.rollout.return_log_probs:
                old_log_probs = old_log_probs_from_rollout(rollout, device=self.device)
            else:
                old_log_probs = log_probs.detach()
            output = grpo_policy_loss(log_probs,
                                      old_log_probs,
                                      advantages[start:start + micro_batch_size],
                                      batch.response_mask,
                                      clip_low=self.cfg.algorithm.eps_clip,
                                      clip_high=self.cfg.algorithm.eps_clip_high,
                                      aggregation=self.cfg.algorithm.loss_aggregation)
            loss = output.loss_sum * loss_scale
            # The loss is already normalized over the whole step, so it must not
            # be divided again by the gradient accumulation steps.
            self.engine.backward(loss, scale_wrt_gas=False)
            self.engine.step()
            step_loss += loss.item()
        return step_loss

    # ------------------------------------------------------------------
    # Logging / checkpointing
    # ------------------------------------------------------------------

    def _log(self, metrics: dict, prefix: str) -> None:
        if not _is_rank_zero():
            return
        values = " ".join(f"{name}={value:.4g}" for name, value in sorted(metrics.items()))
        print(f"[grpo][{prefix}][step {self.step}] {values}", flush=True)

    def _save(self) -> None:
        """Save a resumable DeepSpeed checkpoint and a Hugging Face copy of the weights."""
        tag = f"step_{self.step}"
        save_dir = self.cfg.training.save_dir
        self.engine.save_checkpoint(save_dir, tag=tag)
        # ZeRO-2 keeps full weights on every rank, so rank 0 can export them alone.
        if _is_rank_zero():
            hf_dir = os.path.join(save_dir, f"hf_{tag}")
            self.engine.module.model.save_pretrained(hf_dir)
            self.tokenizer.save_pretrained(hf_dir)
            print(f"[grpo] saved checkpoint to {save_dir}/{tag} and {hf_dir}", flush=True)
        dist.barrier()
