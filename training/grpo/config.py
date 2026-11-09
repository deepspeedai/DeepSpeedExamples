# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""GRPO application configuration.

``GRPOConfig`` is loaded from JSON. The ``rollout`` sub-config is consumed
by DeepSpeed's rollout engine; optimizer, ZeRO and hybrid-engine settings
live in ``deepspeed_config``, either inline or as the path of a DeepSpeed JSON.
"""

import json
from dataclasses import asdict, dataclass, field
from typing import Optional, Union

from deepspeed.runtime.rollout import RolloutConfig as _BaseRolloutConfig


@dataclass
class ModelConfig:
    model_name_or_path: str
    dtype: str = "bfloat16"
    attn_implementation: str = "sdpa"
    gradient_checkpointing: bool = True
    trust_remote_code: bool = False


@dataclass
class RolloutConfig(_BaseRolloutConfig):
    """Extends DeepSpeed's RolloutConfig with the sampling knobs of training rollouts.

    The base config holds the engine settings: ``continuous_batch_size`` (decode rows for continuous
    batching, sized by ``max_context_len``; None keeps static batching) and ``return_log_probs`` (record
    each generated token's log-prob, used as the PPO old log-probs when set).
    """
    multi_turn: bool = True
    max_turns: Optional[int] = 16
    max_context_len: Optional[int] = 16384
    # Generated tokens per trajectory, summed over its turns.
    max_new_tokens: int = 8192
    # Generated tokens per turn, on top of max_new_tokens; None caps turns only by what is left of it.
    max_new_tokens_per_turn: Optional[int] = None
    temperature: float = 1.0
    # 1.0 disables the penalty; None takes the value in the model's generation_config.
    repetition_penalty: Optional[float] = 1.0
    n_samples_per_prompt: int = 8
    # Trajectories per generate call, in whole prompts and at least one; None rolls out a rank's share of the
    # step in one call. Bounds the KV cache and prefill memory of a call.
    generate_batch_size: Optional[int] = None


@dataclass
class ToolConfig:
    timeout: float = 120.0
    memory_limit_gb: int = 4
    max_concurrency: int = 32
    # Long tool outputs are cut so one print cannot overflow the context.
    max_output_chars: int = 4096
    # Also return the value of a bare expression on the code's last line, as a notebook shows it; off returns only
    # what the code prints.
    show_last_value: bool = False


@dataclass
class AlgorithmConfig:
    eps_clip: float = 0.2
    eps_clip_high: float = 0.28
    # Sequence positions per LM-head chunk when computing token log-probs.
    logprob_chunk_size: int = 1024
    # Reward bonus for tool calls on wrong answers.
    tool_call_bonus: bool = False
    # "token_mean": every generated token of the step weighs the same.
    # "sample_mean": each trajectory's loss is averaged over its own tokens, then over trajectories.
    loss_aggregation: str = "token_mean"


@dataclass
class TrainingConfig:
    # Trajectories per step across all ranks, in whole prompts per rank: rolled out together, then
    # one optimizer step over all of them.
    global_batch_size: int = 256
    # Trajectories per forward/backward pass across all ranks, split evenly between them; None means
    # one pass per step, global_batch_size.
    micro_batch_size: Optional[int] = None
    num_rollouts: int = 3000
    logging_steps: int = 1
    save_steps: int = 1000
    save_dir: str = "./grpo_ckpt"
    # Continue from the latest checkpoint in this directory (weights, optimizer state, step count and data
    # position); empty starts from the model. It may differ from save_dir.
    resume_from: str = ""
    seed: int = 42


@dataclass
class DataConfig:
    path: str = ""
    prompt_key: str = "prompt"
    label_key: str = "label"
    shuffle: bool = True


@dataclass
class EvalConfig:
    # Empty path disables evaluation.
    path: str = ""
    interval: int = 20
    # Evaluate once more after the last step.
    at_end: bool = True
    n_samples_per_prompt: int = 16
    max_new_tokens: int = 16384
    max_new_tokens_per_turn: Optional[int] = None
    temperature: float = 1.0


@dataclass
class GRPOConfig:
    model: ModelConfig
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    tool: ToolConfig = field(default_factory=ToolConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    # The DeepSpeed config: a dict, or the path of a JSON file.
    deepspeed_config: Union[str, dict] = ""

    @classmethod
    def from_json(cls, path: str) -> "GRPOConfig":
        with open(path, "r") as f:
            raw = json.load(f)
        return cls(
            model=ModelConfig(**raw["model"]),
            rollout=RolloutConfig(**raw.get("rollout", {})),
            tool=ToolConfig(**raw.get("tool", {})),
            algorithm=AlgorithmConfig(**raw.get("algorithm", {})),
            training=TrainingConfig(**raw.get("training", {})),
            data=DataConfig(**raw.get("data", {})),
            eval=EvalConfig(**raw.get("eval", {})),
            deepspeed_config=raw.get("deepspeed_config", ""),
        )

    def to_dict(self) -> dict:
        return asdict(self)

    def load_deepspeed_config(self) -> dict:
        """A copy of the DeepSpeed config, read from its file when it is a path."""
        if isinstance(self.deepspeed_config, dict):
            return json.loads(json.dumps(self.deepspeed_config))
        if not self.deepspeed_config:
            raise ValueError("deepspeed_config is not set")
        with open(self.deepspeed_config, "r") as f:
            return json.load(f)

    def validate(self, world_size: int) -> None:
        if self.rollout.engine != "hybrid_engine":
            raise ValueError(f"Unknown rollout engine {self.rollout.engine!r}; expected 'hybrid_engine'")
        if self.rollout.n_samples_per_prompt < 2:
            raise ValueError("GRPO needs at least 2 samples per prompt to compare them")
        if self.rollout.temperature <= 0:
            raise ValueError("GRPO samples with temperature > 0")
        if self.rollout.generate_batch_size is not None and self.rollout.generate_batch_size < 1:
            raise ValueError("rollout.generate_batch_size must be positive")
        if self.rollout.continuous_batch_size is not None and not self.rollout.max_context_len:
            raise ValueError("continuous batching sizes its rows by rollout.max_context_len; set it")
        if self.algorithm.loss_aggregation not in ("token_mean", "sample_mean"):
            raise ValueError(f"Unknown loss_aggregation {self.algorithm.loss_aggregation!r}; "
                             "expected 'token_mean' or 'sample_mean'")
        # Each rank rolls out whole prompts, every one with all of its samples.
        trajectories_per_prompt_round = world_size * self.rollout.n_samples_per_prompt
        if self.training.global_batch_size < 1 or self.training.global_batch_size % trajectories_per_prompt_round:
            raise ValueError(f"global_batch_size {self.training.global_batch_size} must be a multiple of world size "
                             f"x n_samples_per_prompt = {trajectories_per_prompt_round}")
        if self.training.micro_batch_size is None:
            self.training.micro_batch_size = self.training.global_batch_size
        micro_batch_size = self.training.micro_batch_size
        if micro_batch_size < 1 or micro_batch_size % world_size:
            raise ValueError(f"micro_batch_size {micro_batch_size} must split into whole trajectories across the "
                             f"world size {world_size}")
        if self.training.global_batch_size % micro_batch_size:
            raise ValueError(f"global_batch_size {self.training.global_batch_size} does not split into "
                             f"micro-batches of {micro_batch_size}")
