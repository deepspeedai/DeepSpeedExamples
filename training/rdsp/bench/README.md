# bench

The benchmark scripts: rdsp against Megatron-Bridge, MegatronMIMO and TorchTitan on Qwen3.5. How to run each one, the settings held equal and the numbers: [docs/GETTING_STARTED.md](../docs/GETTING_STARTED.md#reproducing-the-blogs-results). Every script writes into a `logs/` folder next to it, which stays out of git.

| Experiment | Script | What it writes |
|---|---|---|
| Set up a fresh GPU node | `setup_node.sh` | `~/venv`, the NeMo container |
| Qwen3.5-2B and 4B, one node, against MIMO and Bridge | `qwen35_small/bench.sh` | `~/bench/<model>/r<round>/<case>.log`, `summary.txt` |
| Qwen3.5-35B-A3B, two nodes: prepare both nodes | `qwen35_35b/prep.sh` (runs `node_prep.sh` on each) | `qwen35_35b/logs/prep/` |
| 35B: rdsp, vision encoder on its own stage | `qwen35_35b/rdsp_vision_stage.sh` | `qwen35_35b/logs/vision_stage/rdsp-k<K>*.log` |
| 35B: rdsp, TP1 × PP8 × DP2 | `qwen35_35b/rdsp_pp8dp2.sh` | `qwen35_35b/logs/pp8dp2/rdsp*.log` |
| 35B: Bridge, TP1 × PP8 × DP2 | `qwen35_35b/bridge_pp8dp2.sh` | `qwen35_35b/logs/pp8dp2/bridge*.log` |
| 35B: Bridge, TP1 × PP2 × EP8 (NVIDIA's layout) | `common/runs.sh custom-megatron` through `cluster/both.sh` (the command is in GETTING_STARTED) | `qwen35_35b/logs/pp2ep8/bridge-pp2ep8-nv.log` |
| 35B: TorchTitan, FSDP over 16 GPUs | `qwen35_35b/torchtitan.sh` | `qwen35_35b/logs/fsdp16/<fit\|run>-<config>-node<rank>.log` |
| 35B: rdsp, ZeRO-3 over 16 GPUs | `qwen35_35b/rdsp_zero3.sh` | `qwen35_35b/logs/fsdp16/rdsp-zero3*.log` |
| Qwen3.5-27B: prepare both nodes (35B prep, then the 27B) | `qwen35_27b/prep.sh` | `qwen35_27b/logs/prep/` |
| 27B, one node, TP1 × PP8: Bridge or rdsp | `qwen35_27b/pp8.sh` (rdsp side: `rdsp_pp8.sh`) | `qwen35_27b/logs/pp8/` |
| 27B: rdsp with recompute set per stage | `qwen35_27b/rdsp_tune.sh` | its output (GETTING_STARTED's commands save it in `qwen35_27b/logs/pp8/`) |
| 27B: one node's whole queue | `qwen35_27b/one_node.sh` | `qwen35_27b/logs/pp8/`, `torchtitan/logs/` |
| 9B against MIMO, one node | `mimo/mimo_9b.sh` | `mimo/logs/<layout>/`, `mimo/logs/sweep/` |
| 9B: Bridge against rdsp, TP4 × PP2, and the 9B prep | `mimo/bridge_9b.sh` (rdsp side: `rdsp_9b_tp4pp2.sh`) | `mimo/logs/9b_tp4pp2dp1/` |
| 27B against MIMO, two nodes: rdsp's side | `mimo/rdsp_27b.sh` | `mimo/logs/27b_v1_tp4pp2dp1/rdsp-9gpu.log` |
| 27B against MIMO: MIMO with Bridge's memory settings | `mimo/patch_mimo.py` (`patch.diff`: the result) | a patched copy of MIMO's launcher in `/tmp` |
| TorchTitan on Bridge's rows | `torchtitan/run.sh` (one node), `torchtitan/run_two_node.sh`, `torchtitan/setup_box.sh` | `torchtitan/logs/` |
| Vision encoder alone | `encoder/enc_bench.py`, `encoder/grids.py` | the JSON given with `--out` |
| Two-node helpers | `cluster/both.sh`, `ray_cluster.sh`, `efa_env.sh`, `nccl_check.sh` | `~/both-<time>.log` on node 2 |
| Shared by the scripts above | `common/runs.sh` (Bridge, MIMO and rdsp cases in the NeMo container), `two_node.sh`, `meg_extra_bf16.txt`, `export_bridge_batches.py`, `medians.py`, `summarize.py`, `layout_from_log.py` | `$LOGS/<case>.log` |
