# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""Public PipelineConfig and its declarative per-stage overrides.

Only the compiler reads these; the runtime executes the lowered ExecutionPlan.
"""

from dataclasses import dataclass

from ray_deepspeed_pipeline.errors import ValidationError

# rdsp's own keys in a DeepSpeed config: the stage adapter takes them out
# before DeepSpeed reads it
MASTER_WEIGHTS_KEY = "rdsp_master_weights"
OPTIMIZER_GROUP_NUMEL_KEY = "rdsp_optimizer_group_numel"


@dataclass(frozen=True)
class UniformTransformerBlocks:
    """Split the model's transformer blocks as evenly as possible across stages."""


@dataclass(frozen=True)
class UniformSequential:
    """Split a plain sequential block list as evenly as possible across stages."""


@dataclass(frozen=True)
class MeasuredStages:
    """What profiled steps measured on each stage of a pipeline, for
    BalancedTransformerBlocks(measured=...). cuts: the cut they ran with.
    seconds: each stage's forward and backward time per step (its slowest
    rank). peak_gib: each stage's peak allocated memory (its fullest rank).
    bytes_per_param: each stage's weights, gradients and optimizer state per
    GPU per parameter of the stage (measurement.static_bytes_per_param).
    limit_gib: the most memory a stage may be cut to need."""

    cuts: tuple[int, ...]
    seconds: tuple[float, ...]
    peak_gib: tuple[float, ...]
    bytes_per_param: tuple[float, ...]
    limit_gib: float

    def __post_init__(self):
        for name in ("cuts", "seconds", "peak_gib", "bytes_per_param"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        n = len(self.seconds)
        if len(self.cuts) != n - 1 or len(self.peak_gib) != n or len(self.bytes_per_param) != n:
            raise ValidationError(
                "MeasuredStages: one cut fewer than stages, and one time, "
                "peak and bytes per parameter per stage"
            )
        if not all(seconds > 0 for seconds in self.seconds):
            raise ValidationError(
                f"MeasuredStages: every stage's time must be positive, got "
                f"{self.seconds}; measure again over more steps"
            )

    def block_ranges(self, n_blocks: int) -> list[tuple[int, int]]:
        """Each stage's [start, stop) blocks under the measured cuts."""
        bounds = [0, *self.cuts, n_blocks]
        return list(zip(bounds, bounds[1:]))


@dataclass(frozen=True)
class BalancedTransformerBlocks:
    """Cut so the most expensive stage is as cheap as possible. A block's cost
    is estimated by its parameter count (compute per token scales with it);
    embeddings are lookups and count as nothing. The first stage also carries
    what precedes the blocks (a vision encoder), the last what follows them
    (norm, output head). An estimate: measure, then tune with ExplicitCuts.

    vision_token_ratio: the vision encoder's cost per text token, which
    scales its parameter-count estimate. Measure it from a step's batch with
    partition.vision_token_ratio(): patches over the text tokens actually
    computed, weighted for the encoder's attention over each image.

    vision_stage: put the vision encoder alone on the first stage (as a
    first cut at 0 does) and balance the decoder blocks over the others.

    measured: stage times and memory from profiled steps (MeasuredStages).
    The blocks keep their parameter-count costs, scaled to seconds by the
    stages that hold blocks only (or the last stage with its head's
    estimate); each stage's remaining time (vision encoder, embeddings,
    head and loss, kernel launches) becomes a fixed cost of that stage. No
    stage is cut to need more than limit_gib, counting the weights,
    gradients and optimizer state of the blocks it gains."""

    vision_token_ratio: float = 1.0
    vision_stage: bool = False
    measured: MeasuredStages | None = None


@dataclass(frozen=True)
class ExplicitCuts:
    """Contiguous partition at explicit block boundaries."""

    cuts: tuple[int, ...]

    def __post_init__(self):
        object.__setattr__(self, "cuts", tuple(self.cuts))


PartitionConfig = (
    UniformTransformerBlocks | UniformSequential | BalancedTransformerBlocks | ExplicitCuts
)


@dataclass(frozen=True)
class StageOverride:
    """Per-stage GPU count, parallelism and memory settings.

    The stage's ranks form a grid num_gpus = dp x sp x tp, with dp derived.
    ep (DeepSpeed AutoEP) folds onto those ranks and must divide num_gpus;
    `fold` enables AutoEP Parallel Folding. recompute: keep only each
    block's input for backward and rerun the block there (activation
    checkpointing), trading about a third more compute for memory.
    offload_optimizer: keep the stage's optimizer state (fp32 master weights
    and Adam moments) in host memory and step it on the CPU (ZeRO-Offload;
    needs ZeRO stage 1 or 2). compile: torch.compile each of the stage's
    blocks (shapes dynamic), compiled by each rank on its first step;
    compile_vision: only the blocks of the vision encoder on this stage.
    recompute_blocks: recompute only the first N of the stage's blocks (and
    the blocks of the vision encoder the stage holds) instead of all of them;
    excludes recompute. Under a measured cut N moves with the blocks the stage
    gains or loses, so the blocks that keep their activations stay as many."""

    stage: int
    num_gpus: int = 1
    zero_stage: int | None = None  # None: inherit the DeepSpeed config's value
    tp: int = 1
    sp: int = 1
    ep: int = 1
    fold: bool = False
    recompute: bool = False
    offload_optimizer: bool = False
    compile: bool = False
    compile_vision: bool = False
    recompute_blocks: int | None = None


@dataclass(frozen=True)
class ColocatedVision:
    """Run the vision encoder on every GPU of the pipeline instead of on the
    first stage: each rank encodes a share of the step's images, the first
    stage receives their features (see vision.py). For vision encoders whose
    output enters the decoder only through the input embeddings (Qwen3.5;
    not Qwen3-VL's deepstack). The encoder trains with a plain torch
    optimizer (Adam/AdamW/SGD) built from the DeepSpeed config's optimizer
    and scheduler, with fp32 master weights and optimizer state split over
    all ranks. recompute: keep only each encoder block's input for backward.
    compile: torch.compile each encoder block (shapes dynamic).
    encode_per_microbatch: in training, encode each microbatch's images just ahead
    of the pipeline's need and backpropagate them as soon as their gradients
    exist, instead of all at the start and end of the step: a rank keeps
    graphs for about 2 x stages microbatches' images instead of the step's,
    but the encoder work lands inside the pipeline schedule (vision.py,
    vision_schedule), which slows the step when the encoder is heavy."""

    recompute: bool = False
    compile: bool = False
    encode_per_microbatch: bool = False


@dataclass(frozen=True)
class ConnectionOverride:
    """Declares the tensor conversion on the edge from stage `source` to
    stage `source + 1` (identity, replicate-to-shard, shard-to-replicate or
    shard-to-shard). The conversion is always derived from the two stages'
    layouts; a declared one must match it, or lower() raises
    ValidationError. Use it to make a layout's expectation explicit."""

    source: int
    dest: int
    # None: derived from the two stages' layouts; if given, must match it.
    conversion: str | None = None


@dataclass(frozen=True)
class PipelineConfig:
    """How one model is split into a pipeline.

    stages: pipeline depth. partition: where the cuts go
    (UniformTransformerBlocks, BalancedTransformerBlocks, ExplicitCuts).
    microbatches: microbatches per step; None takes the DeepSpeed config's
    gradient_accumulation_steps. stage_overrides: per-stage GPUs and
    parallelism (StageOverride); stages without one get 1 GPU.
    colocated_vision, prefetch: see their comments below. Invalid
    combinations raise ValidationError when the config is built or lowered,
    before any Ray actor exists."""

    stages: int
    partition: PartitionConfig
    schedule: str = "1f1b"
    microbatches: int | None = None
    stage_overrides: tuple[StageOverride, ...] = ()
    connection_overrides: tuple[ConnectionOverride, ...] = ()
    colocated_vision: ColocatedVision | None = None
    # read the next training step's microbatches from the data iterator while
    # this step runs, and ship them to the stages ahead of time. The caller
    # passes the same iterator to every train_batch() call (not available with
    # initialize(training_data=...)); load_checkpoint() drops the step read ahead.
    prefetch: bool = False

    def __post_init__(self):
        object.__setattr__(self, "stage_overrides", tuple(self.stage_overrides))
        object.__setattr__(self, "connection_overrides", tuple(self.connection_overrides))
        if self.stages < 1:
            raise ValidationError(f"stages must be >= 1, got {self.stages}")
        if self.schedule != "1f1b":
            raise ValidationError(f"unsupported schedule {self.schedule!r}; v1 supports '1f1b'")
        if self.microbatches is not None and self.microbatches < 1:
            raise ValidationError(f"microbatches must be positive, got {self.microbatches}")
        if not isinstance(
            self.partition,
            (UniformTransformerBlocks, UniformSequential, BalancedTransformerBlocks, ExplicitCuts),
        ):
            raise ValidationError(
                "partition must be a partition policy (UniformTransformerBlocks, "
                "UniformSequential, BalancedTransformerBlocks, or ExplicitCuts)"
            )
        for override in self.stage_overrides:
            if not isinstance(override, StageOverride):
                raise ValidationError(
                    f"stage_overrides entries must be StageOverride, got {type(override)}"
                )
            if not 0 <= override.stage < self.stages:
                raise ValidationError(f"StageOverride.stage {override.stage} out of range")
            if override.num_gpus < 1:
                raise ValidationError(
                    f"StageOverride.num_gpus must be >= 1, got {override.num_gpus}"
                )
            for knob in ("tp", "sp", "ep"):
                if getattr(override, knob) < 1:
                    raise ValidationError(
                        f"StageOverride.{knob} must be >= 1, got {getattr(override, knob)}"
                    )
        if self.colocated_vision is not None and not isinstance(
            self.colocated_vision, ColocatedVision
        ):
            raise ValidationError(
                f"colocated_vision must be a ColocatedVision, got {type(self.colocated_vision)}"
            )
        for connection in self.connection_overrides:
            if not isinstance(connection, ConnectionOverride):
                raise ValidationError(
                    "connection_overrides entries must be ConnectionOverride, "
                    f"got {type(connection)}"
                )
