# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""Supported configurations. Default-deny: a row is supported only once its
validation test has passed and the run is recorded here. When and where each
validation ran is in docs/BENCHMARK_RESULTS.md."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SupportRow:
    row_id: str
    dtype: str
    stages: int
    gpus_per_stage: int
    dp: int
    tp: int
    ep: int
    sp: int
    zero_stage: int
    schedule: str
    boundaries: str
    model_fixture: str
    microbatches: int
    supported: bool


BASELINE_ROW = SupportRow(
    row_id="two-stage-baseline",
    dtype="bf16",
    stages=2,
    gpus_per_stage=1,
    dp=1,
    tp=1,
    ep=1,
    sp=1,
    zero_stage=0,
    schedule="1f1b",
    boundaries="identity",
    model_fixture="Qwen/Qwen3-0.6B",
    microbatches=4,
    # Evidence (2x L4): eval and train loss parity with the unsplit model
    # (rel 2e-2, bf16), loss decreases, exact microbatch consumption, eval leaves
    # weights unchanged, engine surface.
    supported=True,
)


@dataclass(frozen=True)
class HeterogeneousRow:
    """A per-stage-heterogeneous configuration.

    `stages` describes each stage's grid as text; `requires` lists rows that must
    be supported first; `evidence` records the passing run of a supported row.
    """

    row_id: str
    scope: str
    stages: tuple[str, ...]
    requires: tuple[str, ...] = ()
    supported: bool = False
    evidence: str = ""
    note: str = ""  # why an unsupported row is not yet validated


_PARITY_EVIDENCE = (
    "8x L4, DeepSpeed 0.19.3+53a2ac44: test_row_parity_global_microbatch_checkpoint_failure"
    "[{row}] passed: SGD loss parity with a single-process fp32 baseline (rel 1e-4, learnable "
    "data), exact N-entry consumption, checkpoint round trip into a fresh differently "
    "initialized pipeline, non-zero-rank kill -> StepFailed -> whole-pipeline recovery"
)

HETEROGENEOUS_ROWS = (
    HeterogeneousRow(
        "static-boundary",
        "replicate<->shard StageConnection conversion",
        ("1 GPU", "dp=2", "1 GPU"),
        supported=True,
        evidence=_PARITY_EVIDENCE.format(row="static-boundary"),
    ),
    HeterogeneousRow(
        "resource-mesh",
        "differing per-stage GPU counts and meshes",
        ("dp=2", "dp=4", "dp=2"),
        supported=True,
        evidence=_PARITY_EVIDENCE.format(row="resource-mesh"),
    ),
    HeterogeneousRow(
        "dp-zero",
        "differing per-stage DP degrees and ZeRO stages",
        ("dp=4 ZeRO-2", "dp=2 ZeRO-1", "dp=2 ZeRO-0"),
        supported=True,
        evidence=_PARITY_EVIDENCE.format(row="dp-zero"),
    ),
    HeterogeneousRow(
        "autotp",
        "stage-local AutoTP",
        ("1 GPU", "tp=2", "tp=2 x dp=2"),
        supported=True,
        evidence=(
            _PARITY_EVIDENCE.format(row="autotp") + "; tiny random Qwen3 (fp32); per-TP-rank "
            "shards in the checkpoint; test_row_gradient_parity_per_parameter[autotp] passed: "
            "91 parameters, worst update error 2.4e-5 against matching reference slices. "
            "Needs rdsp's q_norm/k_norm TP gradient-sum hook (AutoTP leaves them partial)"
        ),
    ),
    HeterogeneousRow(
        "autoep-folding",
        "stage-local AutoEP plus Parallel Folding",
        ("1 GPU", "ep=4", "ep=2 folded with tp=2"),
        supported=True,
        evidence=(
            _PARITY_EVIDENCE.format(row="autoep-folding") + "; tiny random Qwen3-MoE (8 "
            "experts, top-2, fp32); test_row_gradient_parity_per_parameter passed: 76 "
            "parameters, worst 3.0e-4 (folded-stage input_layernorm), rest <= 1.8e-5. Needs "
            "rdsp's TP-average hook at folded MoE inputs"
        ),
    ),
    HeterogeneousRow(
        "sequence-parallel",
        "stage-local Ulysses sequence parallelism",
        ("sp=2", "sp=2 x dp=2", "1 GPU"),
        supported=True,
        evidence=(
            _PARITY_EVIDENCE.format(row="sequence-parallel") + "; tiny random Qwen3 (fp32); "
            "SP ranks >= 1 checkpoint through the model-rank-0 mpu wrapper; "
            "test_row_gradient_parity_per_parameter passed: 91 parameters, worst 2.4e-5. "
            "SP stages average gradients over dp only (DeepSpeed sums over SP)"
        ),
    ),
    HeterogeneousRow(
        "four-stage-mixed",
        "the four stage kinds above composed (32 GPUs, 4 nodes)",
        (
            "dp=4 x sp=2, ZeRO-2",
            "tp=4 x dp=2, ZeRO-1",
            "ep=8 AutoEP + Parallel Folding",
            "tp=2 x dp=4",
        ),
        requires=(
            "static-boundary",
            "resource-mesh",
            "dp-zero",
            "autotp",
            "autoep-folding",
            "sequence-parallel",
        ),
        note=(
            "all six prerequisites are supported, and the same four stage kinds composed "
            "on one 8-GPU node pass every check (test_*[four-stage-mixed-mini], 85 "
            "parameters, worst update error 2.4e-5). The 32-GPU, 4-node run has not been "
            "done, so cross-node rendezvous, placement and node-local checkpoints are "
            "unvalidated. Command: pytest tests/integration/test_heterogeneous_pipeline.py "
            "-k four-stage-mixed"
        ),
    ),
)

ROWS = (BASELINE_ROW, *HETEROGENEOUS_ROWS)
