# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""OPSD application configuration.

``OPSDConfig`` is loaded from JSON and threads through the entire pipeline.
The ``rollout`` sub-config is consumed by DeepSpeed's rollout engine; the
rest is application-level (trainer, data, distillation).
"""

import json
import math
from dataclasses import dataclass, field
from typing import Optional

from deepspeed.runtime.rollout import RolloutConfig as _BaseRolloutConfig


@dataclass
class RolloutConfig(_BaseRolloutConfig):
    """Extends DeepSpeed's RolloutConfig with OPSD generation knobs."""
    use_shared_prefill: bool = False
    max_prompt_length: int = 1024
    max_response_length: int = 1024
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    n_samples_per_prompt: int = 1


@dataclass
class StudentConfig:
    model_name_or_path: str
    dtype: str = "bfloat16"
    trust_remote_code: bool = False


@dataclass
class TeacherConfig:
    model_name_or_path: str
    dtype: str = "bfloat16"
    trust_remote_code: bool = False
    offload_to_cpu: bool = True
    autotp_size: int = 1


@dataclass
class DistillationConfig:
    # "forward_kl" | "reverse_kl" | "jsd"
    loss_type: str = "reverse_kl"
    temperature: float = 1.0
    chunk_size: int = 512
    loss_backend: str = "dense"


@dataclass
class TrainingConfig:
    micro_batch_size_per_gpu: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-6
    weight_decay: float = 0.0
    num_train_epochs: int = 1
    max_steps: int = -1
    warmup_steps: int = 0
    save_steps: int = 500
    logging_steps: int = 10
    save_dir: str = "./opsd_ckpt"
    seed: int = 42


@dataclass
class DataConfig:
    path: str = ""
    prompt_field: str = "prompt"
    chat_template: Optional[str] = None
    shuffle: bool = True


@dataclass
class OPSDConfig:
    student: StudentConfig
    teacher: TeacherConfig
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    distillation: DistillationConfig = field(default_factory=DistillationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    deepspeed_config: str = ""

    @classmethod
    def from_json(cls, path: str) -> "OPSDConfig":
        with open(path, "r") as f:
            raw = json.load(f)
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict) -> "OPSDConfig":
        return cls(
            student=StudentConfig(**raw["student"]),
            teacher=TeacherConfig(**raw["teacher"]),
            rollout=RolloutConfig(**raw.get("rollout", {})),
            distillation=DistillationConfig(**raw.get("distillation", {})),
            training=TrainingConfig(**raw.get("training", {})),
            data=DataConfig(**raw.get("data", {})),
            deepspeed_config=raw.get("deepspeed_config", ""),
        )

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)

    def validate(self) -> None:
        if self.distillation.loss_type not in ("forward_kl", "reverse_kl", "jsd"):
            raise ValueError(f"Unknown loss_type {self.distillation.loss_type!r}")
        if self.rollout.engine != "hybrid_engine":
            raise ValueError(f"Unknown rollout engine {self.rollout.engine!r}; expected 'hybrid_engine'")
        if self.distillation.chunk_size <= 0:
            raise ValueError("distillation.chunk_size must be positive")
        if not math.isfinite(self.distillation.temperature) or self.distillation.temperature <= 0:
            raise ValueError("distillation.temperature must be finite and positive")
        if self.distillation.loss_backend not in ("dense", "vocab_parallel"):
            raise ValueError(f"Unknown loss_backend {self.distillation.loss_backend!r}")
        if self.distillation.loss_backend == "vocab_parallel":
            if self.distillation.loss_type == "jsd":
                raise ValueError("vocab_parallel supports forward/reverse KL; use dense for JSD")
            if self.rollout.temperature != 0 or self.rollout.n_samples_per_prompt != 1:
                raise ValueError("vocab_parallel currently supports single-sample greedy rollout")
            if self.rollout.use_graph_capture or self.rollout.use_shared_prefill:
                raise ValueError("vocab_parallel requires eager rollout without shared prefill")
