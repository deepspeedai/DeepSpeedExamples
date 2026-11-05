# Tiled rematerialization and activation offload

Design notes, capacity estimates and a debugging probe for the long-sequence
activation-memory work (tiling, DeepSpeed CPU activation offload, and their combination)
on H200.

## Code lives in

- DeepSpeed: `sfc-gh-truwase/DeepSpeed`, branch `docs/cpu-checkpointing-perf-note`
  (packed-sequence tiled rematerialization with activation offload).
- ArcticTraining: `sfc-gh-truwase/ArcticTraining`, branch `tjruwase/activation-checkpointing`
  (trainer options, tiled remat for mixers, packing fix, Qwen3.5 GDN boundary fix, configs).
- Draft upstream PR for the GDN packed-boundary fix: snowflakedb/ArcticTraining#376.

## Contents

- `PLAN_packed_seq_tiled_remat.md`, `PLAN_sft_tiling_offload.md`: design plans.
- `WRITEUP_packed_seq_tiled_remat.md`: how packed-sequence tiled rematerialization works.
- `WRITEUP_tiled_remat_intuition.md` (and `.html`): the intuition behind tiling.
- `WRITEUP_qwen3_capacity_estimates.md`, `WRITEUP_qwen35_capacity_estimates.md`: memory
  capacity estimates for Qwen3 and Qwen3.5.
- `bench_qwen3_offload.py`: single-GPU (or torchrun) activation-memory microbenchmark for
  Qwen3 and Qwen3.5 full-parameter SFT. It compares gradient checkpointing, native DeepSpeed
  CPU activation offload, tiled MLP and packed-sequence tiled rematerialization, one child
  process per configuration. Packed tiled remat comes from
  `deepspeed.runtime.activation_checkpointing.hf_tiled_rematerialization` in the DeepSpeed
  branch above. Run `python bench_qwen3_offload.py --help` for the options.
- `at_skew_probe.py`: probe used to root-cause the packed-sequence collective hang. Set
  `AT_SKEW_PROBE_DIR` to choose where it writes; the default is `autorun/_skew3`.
