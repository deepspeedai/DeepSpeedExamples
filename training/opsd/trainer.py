# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""On-policy distillation (OPSD) training loop.

Each step is three phases:

  0. **Rollout.** The student generates responses for the batch's prompts
     (via the configured :class:`~deepspeed.runtime.rollout.RolloutEngine`).
  1. **Teacher.** The frozen teacher runs a forward over prompt+response. The
     full logit tensor is parked on the host via
     :class:`~opsd.teacher.TeacherLogitCache` so teacher GPU buffers can be
     released before the student backward.
  2. **Student.** The student runs forward+backward on prompt+response. The
     loss is the per-token divergence to the teacher, streamed from the
     host-resident cache one sequence chunk at a time
     (:func:`~deepspeed.runtime.rlhf.losses.streamed_distillation_loss`), so
     the full ``[B, T, V]`` teacher tensor never co-resides with the student
     logits on the training device.

The trainer itself contains no DeepSpeed-specific control flow beyond the
``backward`` / ``step`` calls on the student engine; backend choice (ZeRO
stage, offload, hybrid engine) is owned entirely by the DeepSpeed JSON config.
"""

import os
import time
from abc import ABC, abstractmethod
from typing import Any

import torch
from deepspeed import comm as dist
from deepspeed.accelerator import get_accelerator

from config import OPSDConfig
from losses import streamed_distillation_loss
from deepspeed.runtime.rollout import ResponseTokenBatch, RolloutEngine, RolloutRequest, SamplingConfig
from deepspeed.runtime.rollout.base import RolloutRecord


def rollout_request(prompt_ids: torch.Tensor, prompt_attention_mask: torch.Tensor) -> RolloutRequest:
    """A rollout request with one record per row of a left-padded prompt batch, holding the row's real tokens."""
    samples = []
    for row, (ids, mask) in enumerate(zip(prompt_ids.tolist(), prompt_attention_mask.tolist())):
        tokens = [token for token, keep in zip(ids, mask) if keep]
        samples.append(RolloutRecord(group_index=row, tokens=tokens, attention_mask=[1] * len(tokens)))
    return RolloutRequest(samples=samples)


def _is_rank_zero() -> bool:
    return (not dist.is_initialized()) or dist.get_rank() == 0


class RLHFTrainer(ABC):
    """Base class for RLHF training loops."""

    @abstractmethod
    def train(self) -> None:
        ...

    @abstractmethod
    def _train_step(self, batch: Any) -> dict:
        ...


class OPSDTrainer(RLHFTrainer):

    def __init__(
        self,
        cfg: OPSDConfig,
        student_engine: Any,
        teacher: Any,
        tokenizer: Any,
        rollout: RolloutEngine,
        dataloader: Any,
    ):
        self.cfg = cfg
        self.student_engine = student_engine
        self.teacher = teacher
        self.tokenizer = tokenizer
        self.rollout = rollout
        self.dataloader = dataloader

        self.device = get_accelerator().current_device_name()
        self.step = 0

    # ------------------------------------------------------------------
    # Driver
    # ------------------------------------------------------------------

    def train(self) -> None:
        max_steps = self.cfg.training.max_steps
        for epoch in range(self.cfg.training.num_train_epochs):
            # Reseed the (Distributed)Sampler each epoch so each rank sees a
            # different, reshuffled shard. No-op when no sampler is attached.
            getattr(self.dataloader.sampler, "set_epoch", lambda _e: None)(epoch)
            for batch in self.dataloader:
                if max_steps > 0 and self.step >= max_steps:
                    return
                metrics = self._train_step(batch)
                self._maybe_log(metrics)
                self._maybe_save()
                self.step += 1
            if max_steps > 0 and self.step >= max_steps:
                return

    # ------------------------------------------------------------------
    # One step
    # ------------------------------------------------------------------

    def _train_step(self, batch) -> dict:
        t_start = time.time()

        # Sync student weights into the rollout backend.
        self.rollout.sync_weights(self.step)

        # --- Phase 0: rollout (student generates responses) ---------------
        # Switch hybrid engine to inference mode (gathers ZeRO-3 params).
        self.student_engine.eval()
        sampling = SamplingConfig(
            max_new_tokens=self.cfg.rollout.max_response_length,
            temperature=self.cfg.rollout.temperature,
            top_p=self.cfg.rollout.top_p,
            top_k=self.cfg.rollout.top_k,
            n_samples_per_prompt=self.cfg.rollout.n_samples_per_prompt,
        )
        roll = self.rollout.generate(rollout_request(batch["prompt_ids"], batch["prompt_attention_mask"]), sampling)
        # Prompt + response per sample, left-padded to one width, with the response targets shifted by one.
        tokens = ResponseTokenBatch.from_rollout(roll, pad_token_id=self._pad_token_id(), device=self.device)
        input_ids, attention_mask = tokens.input_ids, tokens.attention_mask
        t_rollout = time.time() - t_start

        # --- Phase 1: teacher forward → host-cached logits ----------------
        t1 = time.time()
        teacher_cache = self.teacher.forward_to_cache(input_ids, attention_mask)
        t_teacher = time.time() - t1

        # --- Phase 2: student forward + streamed KL + backward ------------
        t2 = time.time()
        self.student_engine.train()
        outputs = self.student_engine(input_ids=input_ids, attention_mask=attention_mask)
        student_logits = outputs.logits  # [B, T, V]

        # Shift for next-token prediction: logits at position t predict token
        # at t+1, so the loss aligns student_logits[:, :-1] with the position
        # t+1 entries of the response mask.
        student_logits_shifted = tokens.select_causal_logits(student_logits)
        mask_shifted = tokens.response_mask

        def _fetch(start: int, end: int) -> torch.Tensor:
            # The cache holds *unshifted* teacher logits; for the next-token
            # objective we ask the cache for positions [start, end) of the
            # shifted teacher, which is positions [start, end) of the original
            # since we already lopped off the final column in the student.
            return teacher_cache.chunk_to_device(start,
                                                 end,
                                                 device=student_logits_shifted.device,
                                                 dtype=student_logits_shifted.dtype)

        loss = streamed_distillation_loss(
            student_logits=student_logits_shifted,
            teacher_chunk_fetcher=_fetch,
            response_mask=mask_shifted,
            loss_type=self.cfg.distillation.loss_type,
            temperature=self.cfg.distillation.temperature,
            chunk_size=self.cfg.distillation.chunk_size,
        )

        self.student_engine.backward(loss)
        self.student_engine.step()

        teacher_cache.free()
        t_student = time.time() - t2

        # Reduce loss across ranks for a clean log line.
        loss_for_log = loss.detach().clone()
        if dist.is_initialized():
            dist.all_reduce(loss_for_log)
            loss_for_log /= dist.get_world_size()

        return {
            "loss": float(loss_for_log.item()),
            "rollout_s": t_rollout,
            "teacher_s": t_teacher,
            "student_s": t_student,
            "step_s": time.time() - t_start,
            "response_tokens": int(mask_shifted.sum().item()),
        }

    def _pad_token_id(self) -> int:
        pad = self.tokenizer.pad_token_id
        return pad if pad is not None else self.tokenizer.eos_token_id

    # ------------------------------------------------------------------
    # Logging / checkpointing
    # ------------------------------------------------------------------

    def _maybe_log(self, metrics: dict) -> None:
        if self.step % self.cfg.training.logging_steps != 0:
            return
        if not _is_rank_zero():
            return
        print(f"[opsd][step {self.step}] loss={metrics['loss']:.4f} "
              f"rollout={metrics['rollout_s']:.2f}s teacher={metrics['teacher_s']:.2f}s "
              f"student={metrics['student_s']:.2f}s step={metrics['step_s']:.2f}s "
              f"resp_tok={metrics['response_tokens']}")

    def _maybe_save(self) -> None:
        if self.step == 0:
            return
        if self.step % self.cfg.training.save_steps != 0:
            return
        tag = f"step_{self.step}"
        os.makedirs(self.cfg.training.save_dir, exist_ok=True)
        self.student_engine.save_checkpoint(self.cfg.training.save_dir, tag=tag)
        if _is_rank_zero():
            print(f"[opsd] saved checkpoint to {self.cfg.training.save_dir}/{tag}")
