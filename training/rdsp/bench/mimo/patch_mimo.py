# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""patch_mimo.py <copy of finetune_qwen35_vl.py>: MegatronMIMO's Qwen3.5 VL launcher with Megatron-
Bridge's memory settings, so the dense 27B fits on two nodes. Results from it are labeled "MIMO
with our memory settings"; MIMO's library itself is unchanged.

Changes, each asserted to match exactly once in the container's copy (NeMo 26.08):
  1. precision-aware optimizer: bf16 main gradients, fp32 master weights stored as bf16 + 2-byte
     remainders (Bridge's settings in our runs; the launcher fixes fp32 gradients and master)
  2. gradients reduced in bf16 (ddp and the mixed-precision preset both set fp32)
  3. MIMO_RECOMPUTE=1 in the environment: every language-model layer recomputed (full, uniform,
     1 layer); the launcher has no recompute option
The stock file in the container is untouched: runs.sh copies it to /tmp, patches the copy and runs
that, in a container removed after the run.
"""

import sys

path = sys.argv[1]
src = open(path).read()
EDITS = [
    (
        "    optimizer_cfg.use_precision_aware_optimizer = False\n"
        "    optimizer_cfg.main_grads_dtype = torch.float32\n"
        "    optimizer_cfg.main_params_dtype = torch.float32\n",
        "    optimizer_cfg.use_precision_aware_optimizer = True\n"
        "    optimizer_cfg.main_grads_dtype = torch.bfloat16\n"
        "    optimizer_cfg.main_params_dtype = torch.float32\n"
        "    optimizer_cfg.store_param_remainders = True\n",
    ),
    ("            grad_reduce_in_fp32=True,\n", "            grad_reduce_in_fp32=False,\n"),
    (
        '    cfg.mixed_precision = "bf16_mixed" if not args.fp32 else None\n',
        "    from megatron.bridge.training.mixed_precision import bf16_mixed as _bf16_mixed\n"
        "    _mp = _bf16_mixed()\n"
        "    _mp.grad_reduce_in_fp32 = False\n"
        "    cfg.mixed_precision = _mp if not args.fp32 else None\n",
    ),
    (
        '    if hasattr(standard_provider, "mtp_num_layers"):\n',
        '    if os.environ.get("MIMO_RECOMPUTE") == "1":\n'
        '        standard_provider.recompute_granularity = "full"\n'
        '        standard_provider.recompute_method = "uniform"\n'
        "        standard_provider.recompute_num_layers = 1\n"
        '    if hasattr(standard_provider, "mtp_num_layers"):\n',
    ),
]
for old, new in EDITS:
    n = src.count(old)
    if n != 1:
        sys.exit(f"patch_mimo: expected 1 match, found {n}: {old.strip()[:70]!r}")
    src = src.replace(old, new)
open(path, "w").write(src)
print("patch_mimo: 4 edits applied to", path)
