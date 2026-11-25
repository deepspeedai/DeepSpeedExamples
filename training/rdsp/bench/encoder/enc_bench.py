#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""Time one Qwen3.5 vision encoder alone on one GPU: Megatron-Bridge's (Transformer Engine
modules) or Hugging Face's, built from the same config with random weights, on the same images.

For each number of images per call (--images) and each mode (forward; forward + backward):
  timing pass  no hooks, no profiler: median over --passes of the time per image
  profile pass torch.profiler over --profile-images images, with labels on the encoder's parts
               (record_function from module hooks and wrappers around the rotary and attention
               functions; the encoders' code is not changed). From the trace: GPU kernels per
               image, GPU busy time per image, CPU time per part, kernels per part, GPU syncs.

  python enc_bench.py --impl bridge|hf [--hf-attn sdpa|varlen] --config <model dir or id>
         --grids grids.json --images 1,2,4,8 --out result.json [--trace-dir DIR]

Bridge runs inside the NeMo container (Megatron-Bridge v0.6.0 there); Hugging Face runs in the
container (same torch as Bridge) or in rdsp's venv (the stack the blog's rdsp runs used).
"""

import argparse
import bisect
import collections
import gzip
import json
import os
import platform
import statistics
import time
import types

import torch

PARTS = (
    "rotary",
    "core_attn",
    "linear",
    "norm",
    "attn_other",
    "mlp_other",
    "layer_other",
    "outside",
    "backward",
)


# ---------------------------------------------------------------- building the encoders


def build_bridge(config_dir, recompute=False, cuda_graph_len=0):
    """Megatron-Bridge's Qwen3VLVisionModel exactly as Qwen3VLModel builds it
    (modelling_qwen3_vl/model.py, v0.6.0): TE layer spec with Qwen3VLSelfAttention, TE merger,
    get_vision_model_config from the HF vision config; bf16 as Float16Module makes it.
    recompute: the language config's full/uniform/1, as our 27B Bridge runs (the vision config
    copies it). cuda_graph_len: vision CUDA graphs (vision_cuda_graph_impl="transformer_engine",
    every call padded to this many patches), captured by capture_bridge_graphs()."""
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29511")
    torch.distributed.init_process_group("nccl", rank=0, world_size=1)
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    parallel_state.initialize_model_parallel(1, 1)
    # Bridge's initialize.py: use_cudagraphable_rng when CUDA graphs are on (TE's tracker with them)
    model_parallel_cuda_manual_seed(
        0, te_rng_tracker=bool(cuda_graph_len), use_cudagraphable_rng=bool(cuda_graph_len)
    )
    from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.attention import Qwen3VLSelfAttention
    from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.transformer_config import (
        get_vision_model_config,
    )
    from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.utils import PatchMergerSubmodules
    from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.vision_model import Qwen3VLVisionModel
    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelLinear,
        TENorm,
        TERowParallelLinear,
    )
    from megatron.core.models.vision.vit_layer_specs import (
        get_vit_layer_with_transformer_engine_spec,
    )
    from transformers import AutoConfig

    hf_vision = AutoConfig.from_pretrained(config_dir).vision_config
    # the language config's fields get_vision_model_config copies: our Bridge runs' values
    # (no vision CUDA graphs; recompute off here: it repeats the forward, not a per-call cost)
    lang = types.SimpleNamespace(
        recompute_granularity="full" if recompute else None,
        recompute_method="uniform" if recompute else None,
        recompute_num_layers=1 if recompute else None,
        tensor_model_parallel_size=1,
        enable_cuda_graph=False,
        cuda_graph_use_single_mempool=False,
        cuda_graph_retain_backward_graph=False,
        cuda_graph_warmup_steps=3,
        external_cuda_graph=False,
        vision_cuda_graph_impl="transformer_engine" if cuda_graph_len else "none",
        vision_cuda_graph_scope=None,
    )
    if cuda_graph_len:
        lang.max_vision_cuda_graph_seq_length = cuda_graph_len
    cfg = get_vision_model_config(hf_vision, megatron_config=lang)
    cfg.pipeline_model_parallel_size = 1
    cfg.first_pipeline_num_layers = None
    cfg.bf16 = True
    cfg.params_dtype = torch.bfloat16
    spec = get_vit_layer_with_transformer_engine_spec()
    spec.submodules.self_attention.module = Qwen3VLSelfAttention
    merger = PatchMergerSubmodules(
        patch_norm=TENorm, linear_fc1=TEColumnParallelLinear, linear_fc2=TERowParallelLinear
    )
    from megatron.core.process_groups_config import ProcessGroupCollection

    # Qwen3VLModel passes its pg_collection; the vision model reads self.pg_collection.tp
    enc = Qwen3VLVisionModel(
        cfg,
        spec,
        merger,
        pre_process=True,
        post_process=True,
        pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
    )
    if os.environ.get("ENC_INT_MAX_SEQLEN") == "1":
        # experiment, not Bridge's behaviour: hand TE the longest image as a Python int instead of
        # Bridge's GPU tensor (seqlens.max()), to measure what the per-layer GPU reads cost
        build = enc.build_packed_seq_params

        def build_int(grid_thw):
            p = build(grid_thw)
            p.max_seqlen_q = p.max_seqlen_kv = int(p.max_seqlen_q)
            return p

        enc.build_packed_seq_params = build_int
    return enc.cuda().to(torch.bfloat16).train(), hf_vision


def capture_bridge_graphs(enc):
    """Capture the vision layers' CUDA graphs as Bridge's training loop does after its warmup
    steps (training/train.py: VisionTECudaGraphHelper, create_cudagraphs, manual hooks)."""
    from megatron.core import num_microbatches_calculator as nmc
    from megatron.core.transformer.cuda_graphs import (
        VisionTECudaGraphHelper,
        get_vision_cuda_graph_seq_length,
    )

    if nmc._GLOBAL_NUM_MICROBATCHES_CALCULATOR is None:  # training's setup does this
        nmc.init_num_microbatches_calculator(
            rank=0,
            rampup_batch_size=None,
            global_batch_size=1,
            micro_batch_size=1,
            data_parallel_size=1,
        )
    holder = torch.nn.Module()
    holder.vision_model = enc
    helper = VisionTECudaGraphHelper(
        model=[holder],
        vision_config=enc.config,
        vision_seq_length=get_vision_cuda_graph_seq_length(enc.config),
        micro_batch_size=1,
        num_microbatches=1,
    )
    helper.create_cudagraphs()
    helper.cuda_graph_set_manual_hooks()
    assert helper.graphs_created(), "vision CUDA graphs not created"
    return helper


def build_hf(config_dir, attn, device, recompute=False):
    from transformers import AutoConfig
    from transformers.models.qwen3_5 import modeling_qwen3_5

    hf_vision = AutoConfig.from_pretrained(config_dir).vision_config
    hf_vision._attn_implementation = "sdpa"
    torch.manual_seed(0)
    enc = modeling_qwen3_5.Qwen3_5VisionModel(hf_vision)
    if attn == "varlen":
        # rdsp's --vision-attn varlen (coded 2026-10-17, not in the blog's runs): set after the
        # model is built, as rdsp's use_varlen_vision_attention does
        from ray_deepspeed_pipeline import varlen_attention

        varlen_attention.register()
        for m in enc.modules():
            cfg = getattr(m, "config", None)
            if cfg is not None and hasattr(cfg, "_attn_implementation"):
                cfg._attn_implementation = varlen_attention.NAME
    elif attn != "sdpa":
        raise ValueError(attn)
    if recompute:
        # rdsp's recompute_blocks (partition.py): each block's forward under
        # torch.utils.checkpoint(use_reentrant=False)
        import functools

        from torch.utils import checkpoint as torch_checkpoint

        def checkpointed(forward, *a, **k):
            return torch_checkpoint.checkpoint(forward, *a, use_reentrant=False, **k)

        for blk in enc.blocks:
            blk.forward = functools.partial(checkpointed, blk.forward)
    return enc.to(device=device, dtype=torch.bfloat16).train(), hf_vision


class _BlockCall(torch.nn.Module):
    """One HF vision block with positional tensor inputs (what make_graphed_callables takes)."""

    def __init__(self, blk, length, recompute):
        super().__init__()
        self.blk, self.length, self.recompute = blk, length, recompute

    def _run(self, h, cu, cos, sin):
        return self.blk(h, cu_seqlens=cu, max_seqlen=self.length, position_embeddings=(cos, sin))

    def forward(self, h, cu, cos, sin):
        if self.recompute:
            from torch.utils import checkpoint as torch_checkpoint

            return torch_checkpoint.checkpoint(self._run, h, cu, cos, sin, use_reentrant=False)
        return self._run(h, cu, cos, sin)


class HFGraphed(torch.nn.Module):
    """Hugging Face's encoder with CUDA graphs, the way Bridge graphs its encoder: each layer's
    forward and backward captured once (torch.cuda.make_graphed_callables) at a fixed length,
    every call padded to it. Unlike Bridge's, the padding is an attention segment of its own, so
    images never attend to it or to each other (varlen attention, cu_seqlens with the padding
    appended). Patch embedding, position embeddings, rotary tables and merger run eagerly, as in
    Bridge. A prototype for rdsp, not code rdsp ships."""

    def __init__(self, enc, length, recompute):
        super().__init__()
        self.enc, self.length, self.recompute = enc, length, recompute
        self.graphed = None

    def _prepare(self, pixels, grid_thw):
        from transformers.vision_utils import (
            get_vision_cu_seqlens,
            get_vision_interpolation_indices_and_weights,
            get_vision_position_ids,
        )

        enc = self.enc
        idx, w = get_vision_interpolation_indices_and_weights(
            grid_thw,
            num_grid_per_side=enc.num_grid_per_side,
            mode=enc.interpolation_mode,
            align_corners=enc.interpolation_align_corners,
            spatial_merge_size=enc.config.spatial_merge_size,
            kwargs={},
        )
        pos_ids = get_vision_position_ids(grid_thw, enc.spatial_merge_size, kwargs={})
        cu = get_vision_cu_seqlens(grid_thw, kwargs={})
        h = enc.patch_embed(pixels)
        h = h + (enc.pos_embed(idx) * w[:, :, None]).sum(1).to(h.dtype)
        cos, sin = enc.rotary_pos_emb(h, pos_ids)
        s = h.shape[0]
        assert s < self.length, (s, self.length)
        pad = self.length - s
        h = torch.nn.functional.pad(h, (0, 0, 0, pad))
        cos = torch.nn.functional.pad(cos, (0, 0, 0, pad))
        sin = torch.nn.functional.pad(sin, (0, 0, 0, pad))
        cu = torch.cat([cu.to(torch.int32), cu.new_tensor([self.length], dtype=torch.int32)])
        return h, cu, cos, sin, s

    def forward(self, pixels, grid_thw):
        h, cu, cos, sin, s = self._prepare(pixels, grid_thw)
        if self.graphed is None:
            torch.set_grad_enabled(
                True
            )  # capture records backward too, even on a no_grad first call
            calls = tuple(_BlockCall(b, self.length, self.recompute) for b in self.enc.blocks)
            sample = []
            x = h.detach()
            for _ in calls:
                sample.append(
                    (
                        x.clone().requires_grad_(True),
                        cu.clone(),
                        cos.detach().clone(),
                        sin.detach().clone(),
                    )
                )
            self.graphed = torch.cuda.make_graphed_callables(calls, tuple(sample))
            print(f"captured HF vision CUDA graphs, length {self.length}", flush=True)
        for g in self.graphed:
            h = g(h, cu, cos, sin)
        return types.SimpleNamespace(pooler_output=self.enc.merger(h[:s]))


def encoder_output(impl, out):
    """The merged image embeddings each encoder hands to the language model."""
    return out[0] if impl == "bridge" else out.pooler_output


# ---------------------------------------------------------------- labels for the profile pass


class Labels:
    """record_function ranges around the encoder's parts, for the profile pass only."""

    def __init__(self):
        self.handles = []
        self.undo = []

    def module(self, mod, name):
        stack = []

        def pre(_m, _args, *_rest):
            r = torch.autograd.profiler.record_function(name)
            r.__enter__()
            stack.append(r)

        def post(_m, _args, *_rest):
            stack.pop().__exit__(None, None, None)

        self.handles.append(mod.register_forward_pre_hook(pre, with_kwargs=True))
        self.handles.append(mod.register_forward_hook(post, with_kwargs=True))

    def function(self, owner, attr, name):
        fn = getattr(owner, attr)

        def wrapped(*a, **k):
            with torch.autograd.profiler.record_function(name):
                return fn(*a, **k)

        setattr(owner, attr, wrapped)
        self.undo.append(lambda: setattr(owner, attr, fn))

    def remove(self):
        for h in self.handles:
            h.remove()
        for undo in self.undo:
            undo()
        self.handles, self.undo = [], []


def label_bridge(enc, labels):
    from megatron.bridge.models.qwen_vl.modelling_qwen3_vl import attention as bridge_attention

    labels.module(enc, "P:encoder")
    for layer in enc.decoder.layers:
        labels.module(layer, "P:layer")
        sa = layer.self_attention
        labels.module(sa, "P:attn")
        labels.module(sa.linear_qkv, "P:linear")  # TE fuses the layer norm into this linear
        labels.module(sa.core_attention, "P:core_attn")
        labels.module(sa.linear_proj, "P:linear")
        labels.module(layer.mlp, "P:mlp")
        labels.module(layer.mlp.linear_fc1, "P:linear")  # with the pre-MLP layer norm fused
        labels.module(layer.mlp.linear_fc2, "P:linear")
    labels.function(bridge_attention, "apply_rotary_pos_emb_absolute", "P:rotary")


def label_hf(enc, labels):
    from transformers import AttentionInterface
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.qwen3_5 import modeling_qwen3_5

    labels.module(enc, "P:encoder")
    for blk in enc.blocks:
        labels.module(blk, "P:layer")
        labels.module(blk.attn, "P:attn")
        labels.module(blk.attn.qkv, "P:linear")
        labels.module(blk.attn.proj, "P:linear")
        labels.module(blk.norm1, "P:norm")
        labels.module(blk.norm2, "P:norm")
        labels.module(blk.mlp, "P:mlp")
        for child in blk.mlp.children():
            if isinstance(child, torch.nn.Linear):
                labels.module(child, "P:linear")
    labels.function(modeling_qwen3_5, "apply_rotary_pos_emb_vision", "P:rotary")
    name = enc.config._attn_implementation
    fn = ALL_ATTENTION_FUNCTIONS[name]

    def core(*a, **k):
        with torch.autograd.profiler.record_function("P:core_attn"):
            return fn(*a, **k)

    AttentionInterface.register(name, core)
    labels.undo.append(lambda: AttentionInterface.register(name, fn))


# ---------------------------------------------------------------- reading the trace

SYNC_NAMES = ("cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaEventSynchronize")


def analyse_trace(path):
    """Kernels, GPU busy time, syncs, and CPU time and kernels per labelled part."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        events = json.load(f)["traceEvents"]
    ranges = collections.defaultdict(list)  # tid -> [(start, end, part)]
    launch_at = {}  # correlation -> (tid, ts)
    kernels, syncs, memcpy_d2h = [], 0, 0
    for e in events:
        if e.get("ph") != "X":
            continue
        cat, name = e.get("cat", ""), e.get("name", "")
        if cat == "user_annotation" and name.startswith("P:"):
            ranges[e["tid"]].append((e["ts"], e["ts"] + e["dur"], name[2:]))
        elif cat == "cuda_runtime" or cat == "cuda_driver":
            corr = e.get("args", {}).get("correlation")
            if corr is not None:
                launch_at[corr] = (e["tid"], e["ts"])
            if name in SYNC_NAMES:
                syncs += 1
        elif cat == "kernel":
            kernels.append(e)
        elif cat == "gpu_memcpy" and "DtoH" in name:
            memcpy_d2h += 1

    # innermost range containing a time, per thread
    index = {}
    for tid, rs in ranges.items():
        rs.sort()
        index[tid] = ([r[0] for r in rs], rs)

    def part_at(tid, ts):
        if tid not in index:
            return "backward"  # autograd's thread: no labels there
        starts, rs = index[tid]
        # ranges nest: of those containing ts, the one starting last is the innermost
        i = bisect.bisect_right(starts, ts) - 1
        while i >= 0:
            s, end, part = rs[i]
            if ts <= end:
                return part
            i -= 1
        return "outside_encoder"

    def category(part):
        return {
            "encoder": "outside",
            "layer": "layer_other",
            "attn": "attn_other",
            "mlp": "mlp_other",
        }.get(part, part)

    k_count = collections.Counter()
    k_time = collections.Counter()
    busy = 0.0
    for k in kernels:
        tid_ts = launch_at.get(k.get("args", {}).get("correlation"))
        part = category(part_at(*tid_ts)) if tid_ts else "unknown"
        k_count[part] += 1
        k_time[part] += k["dur"]
        busy += k["dur"]

    # CPU time per part, exclusive of nested labelled ranges (forward thread)
    cpu = collections.Counter()
    for rs in ranges.values():
        rs = sorted(rs, key=lambda r: (r[0], -r[1]))
        stack = []
        child = collections.defaultdict(float)
        for s, e, part in rs:
            while stack and stack[-1][1] <= s:
                stack.pop()
            if stack:
                child[stack[-1]] += e - s
            stack.append((s, e, part))
        for r in rs:
            cpu[category(r[2])] += (r[1] - r[0]) - child[r]
    return {
        "kernels": len(kernels),
        "gpu_busy_us": busy,
        "syncs": syncs,
        "memcpy_d2h": memcpy_d2h,
        "kernels_by_part": dict(k_count),
        "gpu_us_by_part": dict(k_time),
        "cpu_us_by_part": dict(cpu),
    }


# ---------------------------------------------------------------- running


def make_calls(impl, enc, hf_vision, grids, n, device):
    """Groups of n consecutive images: (pixel_values, grid_thw, output grad) per call."""
    patch_dim = hf_vision.in_channels * hf_vision.temporal_patch_size * hf_vision.patch_size**2
    g = torch.Generator().manual_seed(0)
    calls = []
    for i in range(0, len(grids) - n + 1, n):
        thw = torch.tensor(grids[i : i + n], dtype=torch.long)
        patches = int(thw.prod(-1).sum())
        pixels = torch.randn(patches, patch_dim, generator=g).to(device, torch.bfloat16)
        thw = thw.to(device)
        with torch.no_grad():
            out = encoder_output(
                impl, enc(pixels, grid_thw=thw) if impl == "hf" else enc(pixels, thw)
            )
        grad = torch.randn(out.shape, generator=g).to(device, out.dtype)
        calls.append((pixels, thw, grad))
    return calls


def run_call(impl, enc, call, backward):
    pixels, thw, grad = call
    out = enc(pixels, grid_thw=thw) if impl == "hf" else enc(pixels, thw)
    out = encoder_output(impl, out)
    if backward:
        out.backward(grad)


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--impl", choices=("bridge", "hf"), required=True)
    ap.add_argument("--hf-attn", default="sdpa", help="sdpa (the blog's rdsp runs) or varlen")
    ap.add_argument("--config", default="Qwen/Qwen3.5-27B")
    ap.add_argument("--grids", required=True, help="JSON list of [t, h, w] per image")
    ap.add_argument("--images", default="1,2,4,8")
    ap.add_argument("--modes", default="fwd,fwdbwd")
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--profile-images", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--trace-dir", default="traces")
    ap.add_argument(
        "--recompute",
        action="store_true",
        help="recompute every layer in backward, as both systems' 27B runs",
    )
    ap.add_argument(
        "--cuda-graphs",
        action="store_true",
        help="Bridge's vision CUDA graphs; one --images value; each call padded to "
        "the longest call's patches (rounded up to 256)",
    )
    ap.add_argument(
        "--hf-cuda-graphs",
        action="store_true",
        help="HFGraphed: Hugging Face's encoder with per-layer CUDA graphs (a prototype; "
        "needs --hf-attn varlen and one --images value)",
    )
    ap.add_argument("--label", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    grids = json.load(open(args.grids))
    device = args.device
    graph_len = 0
    if args.cuda_graphs or args.hf_cuda_graphs:
        assert "," not in args.images, "CUDA graphs: one --images value"
        assert args.impl == ("bridge" if args.cuda_graphs else "hf")
        n = int(args.images)
        longest = max(
            sum(t * h * w for t, h, w in grids[i : i + n]) for i in range(0, len(grids) - n + 1, n)
        )
        # HF's padding is a segment of its own: at least one patch longer than the longest call
        graph_len = -(-(longest + (1 if args.hf_cuda_graphs else 0)) // 256) * 256
    if args.impl == "bridge":
        enc, hf_vision = build_bridge(args.config, args.recompute, graph_len)
    elif args.hf_cuda_graphs:
        assert args.hf_attn == "varlen", "--hf-cuda-graphs needs --hf-attn varlen (no host sync)"
        enc, hf_vision = build_hf(args.config, args.hf_attn, device, recompute=False)
        enc = HFGraphed(enc, graph_len, args.recompute)
    else:
        enc, hf_vision = build_hf(args.config, args.hf_attn, device, args.recompute)
    label = args.label or (args.impl if args.impl == "bridge" else f"hf-{args.hf_attn}")
    graphs_captured = False
    os.makedirs(args.trace_dir, exist_ok=True)
    import transformers

    env = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "python": platform.python_version(),
        "gpu": torch.cuda.get_device_name() if device == "cuda" else "cpu",
        "params": sum(p.numel() for p in enc.parameters()),
    }
    try:
        import transformer_engine

        env["transformer_engine"] = transformer_engine.__version__
    except ImportError:
        pass
    print(json.dumps(env), flush=True)

    results = []
    for n in [int(x) for x in args.images.split(",")]:
        calls = make_calls(args.impl, enc, hf_vision, grids, n, device)
        images = len(calls) * n
        for mode in args.modes.split(","):
            bwd = mode == "fwdbwd"
            for call in calls[:3]:  # warm up: autotuning, allocator, first-call setup
                run_call(args.impl, enc, call, bwd)
            sync(device)
            if args.cuda_graphs and not graphs_captured:
                # Bridge captures after cuda_graph_warmup_steps (3) training steps
                t = time.perf_counter()
                capture_bridge_graphs(enc)
                print(
                    f"captured vision CUDA graphs, length {graph_len}, in "
                    f"{time.perf_counter() - t:.1f} s",
                    flush=True,
                )
                graphs_captured = True
                for call in calls[:3]:
                    run_call(args.impl, enc, call, bwd)
                sync(device)
            per_pass = []
            for _ in range(args.passes):
                t = time.perf_counter()
                for call in calls:
                    run_call(args.impl, enc, call, bwd)
                    sync(device)
                per_pass.append((time.perf_counter() - t) / images * 1e3)
            row = {
                "label": label,
                "impl": args.impl,
                "images_per_call": n,
                "mode": mode,
                "recompute": args.recompute,
                "cuda_graph_len": graph_len,
                "images_timed": images,
                "ms_per_image_passes": per_pass,
                "ms_per_image": statistics.median(per_pass),
            }
            if args.profile_images > 0:
                labels = Labels()
                if not args.hf_cuda_graphs:  # graph replay runs no Python inside the layers
                    (label_bridge if args.impl == "bridge" else label_hf)(enc, labels)
                pcalls = calls[: max(1, args.profile_images // n)]
                acts = [torch.profiler.ProfilerActivity.CPU]
                if device == "cuda":
                    acts.append(torch.profiler.ProfilerActivity.CUDA)
                with torch.profiler.profile(activities=acts) as prof:
                    t = time.perf_counter()
                    for call in pcalls:
                        run_call(args.impl, enc, call, bwd)
                        sync(device)
                    traced_ms = (time.perf_counter() - t) * 1e3
                labels.remove()
                path = os.path.join(args.trace_dir, f"{label}-n{n}-{mode}.json.gz")
                prof.export_chrome_trace(path)
                pimages = len(pcalls) * n
                a = analyse_trace(path)
                row.update(
                    {
                        "profiled_images": pimages,
                        "traced_ms_per_image": traced_ms / pimages,
                        "kernels_per_image": a["kernels"] / pimages,
                        "gpu_busy_ms_per_image": a["gpu_busy_us"] / pimages / 1e3,
                        "syncs_per_image": a["syncs"] / pimages,
                        "d2h_copies_per_image": a["memcpy_d2h"] / pimages,
                        "kernels_by_part_per_image": {
                            k: v / pimages for k, v in a["kernels_by_part"].items()
                        },
                        "gpu_ms_by_part_per_image": {
                            k: v / pimages / 1e3 for k, v in a["gpu_us_by_part"].items()
                        },
                        "cpu_ms_by_part_per_image": {
                            k: v / pimages / 1e3 for k, v in a["cpu_us_by_part"].items()
                        },
                    }
                )
            print(
                json.dumps(
                    {
                        k: (round(v, 3) if isinstance(v, float) else v)
                        for k, v in row.items()
                        if not isinstance(v, dict)
                    }
                ),
                flush=True,
            )
            results.append(row)
        del calls
        if device == "cuda":
            torch.cuda.empty_cache()
    json.dump({"env": env, "args": vars(args), "results": results}, open(args.out, "w"), indent=1)
    print("=== DONE", label, flush=True)


if __name__ == "__main__":
    main()
