# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Qwen3 / Qwen3.5 activation-offload benchmark (single GPU, full-parameter SFT).

Compares activation-checkpointing strategies:

  * gc                 : HF gradient checkpointing (use_reentrant=False), no offload
  * dss                : DSS install_activation_offload (pack-hooks + LIFO prefetch)
  * hf_offload         : gc + CheckpointHiddenStatesOffload (no prefetch)
  * hf_prefetch        : hf_offload with engine LIFO prefetch
  * ds_async           : native cpu_checkpointing via non_reentrant_checkpoint
  * ds_async_prefetch  : ds_async with cross-checkpoint LIFO prefetch
  * ds_blocking        : same native path, async engine disabled (blocking D2H/H2D)
  * hf_gc_offload      : HF gradient_checkpointing_enable(offload=True) (transformers#48444)

Each (mode, seqlen, mbs, gas) runs in an isolated child process so the CUDA
allocator peak is clean. The driver renders a summary table.

Metrics per config: ms/global step, tok/s/gpu = (mbs * gas * seqlen) / global_step,
tok/s = that times world_size, CUDA peak / max GB, CPU RSS delta GB.

Pass ``--fused_ce`` to use Liger fused linear+cross-entropy (no full logits tensor).
Pass ``--sum_loss`` to skip lm_head/CE (loss = last_hidden.float().sum()).
Pass ``--zero2`` for ZeRO-2 with GPU Adam. Pass ``--zero2_offload`` for ZeRO-2 + CPU Adam.
Pass ``--zero3`` for ZeRO-3 with GPU Adam.
Pass ``--nproc N`` to launch each child under torchrun (N GPUs per node).
Pass ``--nnodes 2 --hostfile /job/hostfile`` for a 2-node job (world size = nnodes * nproc).
Pass ``--autoep_size N`` for AutoEP + ZeRO (shards experts so GPU Adam can fit MoE).
Qwen3.5 loads the text config only (no vision encoder). AutoEP Qwen3.5 needs
``model_type=qwen3_5_moe_text`` and Transformers >= 5.2. TiledMLP (sequence-sharded SwiGLU) can run with AutoEP.
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time

MODES = ("gc", "dss", "hf_offload", "hf_prefetch", "ds_async", "ds_async_prefetch", "ds_blocking", "hf_gc_offload")
RESULT_PREFIX = "RESULT_JSON:"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--child", action="store_true", help="run a single (mode, seqlen) config")
    p.add_argument("--mode", choices=MODES, default=None)
    p.add_argument("--seqlen", type=int, default=None)
    p.add_argument("--modes", default=",".join(MODES), help="driver: comma list of modes")
    p.add_argument("--seqlens", default="8192,16384,32768", help="driver: comma list of seq lengths")
    p.add_argument("--model_name", default="Qwen/Qwen3-8B")
    p.add_argument("--num_layers", type=int, default=0, help="0 = model default; smaller for smoke runs")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=0, help="0 = auto (5 at >=32K else 10)")
    p.add_argument("--mbs", type=int, default=1, help="microbatch size (input batch dim)")
    p.add_argument("--gas", type=int, default=1, help="gradient accumulation steps (one global step)")
    p.add_argument("--fused_ce", action="store_true", help="Liger fused linear+CE (no materialized [B,S,V] logits)")
    p.add_argument("--sum_loss", action="store_true", help="Skip lm_head/CE; loss = last_hidden.float().sum()")
    p.add_argument("--zero2", action="store_true", help="ZeRO-2 with optimizer on GPU (Fused/GPU AdamW)")
    p.add_argument("--zero2_offload",
                   action="store_true",
                   help="ZeRO-2 with optimizer states offloaded to CPU (DeepSpeedCPUAdam)")
    p.add_argument("--zero3", action="store_true", help="ZeRO-3 with optimizer on GPU (Fused/GPU AdamW)")
    p.add_argument("--zero3_offload",
                   action="store_true",
                   help="ZeRO-3 with param+grad sharding AND optimizer states offloaded to CPU "
                   "(DeepSpeedCPUAdam)")
    p.add_argument("--nproc", type=int, default=1, help="GPUs per node (1 = single process; >1 uses torchrun)")
    p.add_argument("--nnodes", type=int, default=1, help="torchrun nnodes (1 = standalone; >1 needs --hostfile)")
    p.add_argument("--hostfile", default="", help="DeepSpeed-style hostfile; required when --nnodes>1")
    p.add_argument("--master_addr", default="", help="Rendezvous IP; default is the first hostfile entry")
    p.add_argument("--opt_pin_memory",
                   action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Pin ZeRO offload_optimizer CPU buffers (disable on tight host RAM)")
    p.add_argument("--cpu_optimizer",
                   choices=("AdamW", "SGD"),
                   default="AdamW",
                   help="CPU optimizer when --zero2_offload (SGD has no m/v states)")
    p.add_argument("--act_pin_memory",
                   action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Pin activation-offload CPU buffers (ds_async / hf_offload)")
    p.add_argument("--act_use_streams",
                   action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Use async copy stream for activation offload (ds_async); "
                        "--no-act_use_streams forces synchronous/blocking D2H+H2D copies "
                        "on the compute stream, still honoring --act_pin_memory")
    p.add_argument("--tiled_mlp_shards",
                   type=int,
                   default=0,
                   help="If >1, wrap SwiGLU MLPs with TiledMLP (ALST sequence tiling)")
    p.add_argument("--tiled_rematerialize_shards",
                   type=int,
                   default=0,
                   help="If >1, tile text decoder layers along batch (no-op when mbs<=1)")
    p.add_argument("--tiled_fwd",
                   action="store_true",
                   help="Shard the no_grad first forward along batch (default: full-batch fwd)")
    p.add_argument("--packed_seq",
                   action="store_true",
                   help="Build one packed (cu_seqlens) row from --pack_lens instead of a dense [mbs,seqlen] batch")
    p.add_argument("--pack_lens",
                   default="",
                   help="Comma-separated per-sample token lengths for one packed row (requires --packed_seq)")
    p.add_argument("--packed_tiled_remat_shards",
                   type=int,
                   default=0,
                   help="If >1, tile the self_attn/linear_attn mixer along tokens within the packed row "
                   "(requires --packed_seq)")
    p.add_argument("--packed_tiled_remat_budget",
                   type=int,
                   default=0,
                   help="If >0, size packed-seq mixer tiles by a token budget instead of a shard count "
                   "(requires --packed_seq)")
    p.add_argument("--seq_chain_budget",
                   type=int,
                   default=0,
                   help="If >0, rematerialize each linear_attn (GDN) as a sequential chain of this many "
                   "tokens. One dense sample (mbs=1, no --packed_seq). Full attention stays on --mode.")
    p.add_argument("--comms_logger",
                   action="store_true",
                   help="Enable DeepSpeed comms_logger and dump allocator retries after the timed loop")
    p.add_argument("--autoep_size", type=int, default=0, help="AutoEP degree (0=off). Requires ZeRO.")
    p.add_argument("--autoep_preset", default="auto", help="AutoEP preset or auto")
    p.add_argument("--mem_snapshot",
                   default="",
                   help="Rank-0 path for a CUDA allocator snapshot of the timed loop (use with --iters 1)")
    p.add_argument("--offload_reclaim_probe",
                   action="store_true",
                   help="Measure whether emptying an offloaded tiled hidden actually frees GPU bytes")
    p.add_argument("--mem_census",
                   action="store_true",
                   help="Attribute resident GPU bytes to params/optimizer/grads (finds duplicate weight copies)")
    p.add_argument("--mem_snapshot_early",
                   action="store_true",
                   help="Start the allocator recording before warmup so retained blocks keep their stacks")
    p.add_argument("--gc_reentrant",
                   action="store_true",
                   help="Use reentrant HF checkpointing; the non-reentrant path never recomputes on this model")
    p.add_argument("--gc_each_step",
                   action="store_true",
                   help="Run gc.collect() per step so cyclic garbage cannot inflate the resident baseline")
    p.add_argument("--layer_workspace_probe",
                   action="store_true",
                   help="During checkpoint rematerialize, record the allocator high-water "
                   "inside one full_attention mixer and one linear_attention mixer")
    return p.parse_args()


def _text_config(cfg):
    if hasattr(cfg, "get_text_config"):
        text = cfg.get_text_config()
        if text is not None:
            return text
    return cfg


def _is_qwen35(model_name, cfg):
    name = (model_name or "").lower()
    model_type = getattr(cfg, "model_type", "") or ""
    return ("qwen3.5" in name or "qwen3_5" in name
            or model_type in ("qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text"))


def _install_tiled_mlp(model, num_shards):
    from deepspeed.runtime.sequence_parallel.ulysses_sp import TiledMLP

    patched = []
    for mod in model.modules():
        if not (hasattr(mod, "gate_proj") and hasattr(mod, "up_proj") and hasattr(mod, "down_proj")):
            continue
        cls = type(mod)
        if cls in patched:
            continue
        orig_fwd = cls.forward

        def _make(orig, shards):

            def tiled_forward(self, hidden_states):
                compute_params = [self.gate_proj.weight, self.up_proj.weight, self.down_proj.weight]
                return TiledMLP.apply(orig, self, hidden_states, shards, compute_params)

            return tiled_forward

        cls.forward = _make(orig_fwd, num_shards)
        patched.append(cls)
    n = sum(1 for m in model.modules() if type(m) in patched)
    print(f"[child] tiled_mlp shards={num_shards} classes={[c.__name__ for c in patched]} modules={n}",
          file=sys.stderr)
    if n == 0:
        raise RuntimeError("tiled_mlp_shards>1 but no gate/up/down MLP modules found")
    return n


_RECLAIM_PROBE = {"calls": [], "enabled": False}


def _install_offload_reclaim_probe():
    """Wrap the tiled offload site to see if the emptied hidden is really freed.

    ``_offload_remat_input`` D2H-copies the hidden then drops its storage. If
    another reference survives (the leaf hypothesis), allocated bytes will not
    fall even though the copy happened.
    """
    import torch
    from deepspeed.runtime.activation_checkpointing import tiled_rematerialization as tr

    orig = tr._offload_remat_input

    def probed(ctx, x):
        nbytes = x.numel() * x.element_size()
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        orig(ctx, x)
        torch.cuda.synchronize()
        after = torch.cuda.memory_allocated()
        if _RECLAIM_PROBE["enabled"]:
            _RECLAIM_PROBE["calls"].append((nbytes, before - after))

    tr._offload_remat_input = probed
    print("[child] offload reclaim probe installed", file=sys.stderr)


def _dump_reclaim_probe(tag):
    calls = _RECLAIM_PROBE["calls"]
    if not calls:
        print(f"[reclaim] {tag} no calls recorded", file=sys.stderr, flush=True)
        return
    offloaded = sum(nb for nb, _ in calls)
    freed = sum(fr for _, fr in calls)
    fully = sum(1 for nb, fr in calls if fr >= 0.9 * nb)
    not_freed = sum(1 for nb, fr in calls if fr <= 0.1 * nb)
    print(
        f"[reclaim] {tag} calls={len(calls)} offloaded={offloaded/2**30:.2f}GiB "
        f"freed={freed/2**30:.2f}GiB ({freed/max(1, offloaded):.0%}) "
        f"full_free={fully} no_free={not_freed} "
        f"per_call_MiB={offloaded/len(calls)/2**20:.0f}",
        file=sys.stderr,
        flush=True,
    )


def _is_text_decoder_layer(mod):
    cls_name = type(mod).__name__
    if "Vision" in cls_name:
        return False
    if "DecoderLayer" in cls_name:
        return True
    has_mixer = hasattr(mod, "self_attn") or hasattr(mod, "linear_attn")
    return has_mixer and hasattr(mod, "mlp")


def _install_tiled_rematerialize(model, num_shards, tile_forward=False):
    import inspect

    import torch
    from deepspeed.runtime.activation_checkpointing.tiled_rematerialization import (
        classify_rematerialize_kwarg,
        tiled_rematerialize_compute,
    )

    for mod in model.modules():
        if isinstance(mod, torch.nn.Dropout):
            mod.p = 0.0
        dropout = getattr(mod, "attention_dropout", None)
        if isinstance(dropout, float):
            mod.attention_dropout = 0.0

    patched = []
    for mod in model.modules():
        if not _is_text_decoder_layer(mod):
            continue
        cls = type(mod)
        if cls in patched:
            continue
        orig_fwd = cls.forward
        sig = inspect.signature(orig_fwd)

        def _make(orig, shards, fwd_sig, tile_fwd):

            def tiled_forward(self, hidden_states, *args, **kwargs):
                bound = fwd_sig.bind_partial(self, hidden_states, *args, **kwargs)
                bound.apply_defaults()
                call_kw = {}
                for name, value in bound.arguments.items():
                    if name in ("self", "hidden_states"):
                        continue
                    param = fwd_sig.parameters.get(name)
                    if param is not None and param.kind == param.VAR_KEYWORD:
                        call_kw.update(value or {})
                    else:
                        call_kw[name] = value
                batch = hidden_states.shape[0]
                to_shard = {}
                to_pass = {}
                for name, value in call_kw.items():
                    lname = name.lower()
                    # Training does not use KV/GDN cache; a live Cache cannot be
                    # sharded and must not be shared across rows.
                    if lname in ("past_key_value", "past_key_values", "cache_params", "cache") or "cache" in lname:
                        to_pass[name] = None
                        continue
                    try:
                        kind = classify_rematerialize_kwarg(name, value, batch)
                    except TypeError:
                        if value is None:
                            to_pass[name] = None
                            continue
                        raise
                    if kind == "shard":
                        to_shard[name] = value
                    else:
                        to_pass[name] = value

                def _fn(h, **kw):
                    out = orig(self, h, **kw)
                    if torch.is_tensor(out):
                        return out
                    if isinstance(out, tuple) and out and torch.is_tensor(out[0]):
                        extra = [x for x in out[1:] if x is not None]
                        if extra:
                            raise TypeError(f"tiled rematerialize: {type(self).__name__} returned extra outputs")
                        return out[0]
                    raise TypeError(f"tiled rematerialize: {type(self).__name__} returned {type(out)}")

                params = [p for p in self.parameters() if p.requires_grad]
                return tiled_rematerialize_compute(_fn,
                                                   hidden_states,
                                                   shards,
                                                   params,
                                                   to_shard,
                                                   to_pass,
                                                   tile_forward=tile_fwd)

            return tiled_forward

        cls.forward = _make(orig_fwd, num_shards, sig, tile_forward)
        patched.append(cls)
    n = sum(1 for m in model.modules() if type(m) in patched)
    print(
        f"[child] tiled_rematerialize shards={num_shards} tiled_fwd={tile_forward} "
        f"classes={[c.__name__ for c in patched]} modules={n}",
        file=sys.stderr)
    if n == 0:
        raise RuntimeError("tiled_rematerialize_shards>1 but no text decoder layers found")
    return patched


# (kernel function, kwarg that resets state at packed boundaries, state, package providing it)
_GDN_BOUNDARY_RESET_KERNELS = (
    ("torch_chunk_gated_delta_rule", "cu_seqlens", "recurrent state", "fla (flash-linear-attention)"),
    ("causal_conv1d_fn", "seq_idx", "short-conv state", "causal-conv1d"),
)


def _hf_kernel_applicable_params(fn):
    """Kwarg names transformers' use_kernel_func_from_hub_with_fallback lets through, or None if not that wrapper.

    That wrapper keeps only kwargs *named* in the resolved implementation's
    signature, so the torch fallback silently drops cu_seqlens / seq_idx
    (a ``**kwargs`` catch-all does not count -- its name is just "kwargs").
    """
    import inspect

    while callable(fn):
        try:
            nonlocals = inspect.getclosurevars(fn).nonlocals
        except TypeError:
            return None
        if "applicable_params" in nonlocals:
            return nonlocals["applicable_params"]
        fn = getattr(fn, "__wrapped__", None)
    return None


def _gdn_boundary_reset_gaps(mixer_cls):
    """Reasons ``mixer_cls`` would NOT reset GDN state at every packed boundary; empty if it does.

    Whole-sample tiling only preserves numerics if the untiled packed forward
    already resets both states at every boundary; see "GDN state reset at
    packed boundaries" in tiled_remat/WRITEUP_packed_seq_tiled_remat.md.
    """
    module = sys.modules.get(mixer_cls.__module__)
    gaps = []
    for func_name, reset_kwarg, state, package in _GDN_BOUNDARY_RESET_KERNELS:
        params = _hf_kernel_applicable_params(getattr(module, func_name, None))
        if params is None:
            gaps.append(f"{state}: no transformers kernel-fallback wrapper named {func_name} in "
                        f"{mixer_cls.__module__}, so boundary resets cannot be verified")
        elif reset_kwarg not in params:
            gaps.append(f"{state}: {func_name} resolved to an implementation that drops '{reset_kwarg}' "
                        f"(install {package})")
    return gaps


def _install_seq_chain_tile(model, token_budget):
    """Chain one dense sample through each GDN mixer. Full attention is left alone.

    ``sequential_chain_rematerialize`` does not read cu_seqlens, so this is the
    single-sample path. A packed row must use the nested packed installer.
    """
    from deepspeed.runtime.activation_checkpointing.tiled_rematerialization import (
        qwen35_gdn_tile_forward,
        sequential_chain_rematerialize,
    )

    budget = int(token_budget)
    patched = 0
    for parent in model.modules():
        attn = getattr(parent, "linear_attn", None)
        if attn is None or not hasattr(attn, "in_proj_qkv"):
            continue
        if getattr(attn, "_seq_chain_patched", False):
            continue
        params = [p for p in attn.parameters() if p.requires_grad]

        def _tile(hidden, recurrent, conv, _attn=attn):
            return qwen35_gdn_tile_forward(_attn, hidden, recurrent, conv)

        def _wrapped(hidden_states, cache_params=None, attention_mask=None, _params=params, _tile=_tile, **kwargs):
            if cache_params is not None:
                raise RuntimeError("seq_chain_tile is the training forward (cache_params must be None)")
            if kwargs.get("seq_idx") is not None:
                raise RuntimeError("seq_chain_tile does not reset at packed boundaries; use packed tiled remat")
            if attention_mask is not None:
                if tuple(attention_mask.shape) != tuple(hidden_states.shape[:2]):
                    raise ValueError(f"seq_chain_tile attention_mask shape {tuple(attention_mask.shape)} "
                                     f"must be [B, T] = {tuple(hidden_states.shape[:2])}")
                # Mask once on the full row. The tile body is not given a per-tile mask.
                hidden_states = (hidden_states * attention_mask[:, :, None]).to(hidden_states.dtype)
            return sequential_chain_rematerialize(_tile, hidden_states, budget, compute_params=_params)

        attn.forward = _wrapped
        attn._seq_chain_patched = True
        patched += 1
    print(f"[child] seq_chain_tile budget={budget} linear_attn={patched}", file=sys.stderr)
    if patched == 0:
        raise RuntimeError("seq_chain_budget requested but no linear_attn GDN modules found")
    return patched


def _autoep_preset_name(model_name, cfg, preset):
    if preset and preset != "auto":
        return preset
    model_type = getattr(cfg, "model_type", "") or ""
    if _is_qwen35_moe(model_name, cfg) or "qwen3_5_moe" in model_type:
        return "qwen3_5_moe"
    if "qwen3_moe" in model_type or "qwen2_moe" in model_type:
        return "qwen3_moe"
    return "mixtral"


def _require_autoep_text_backbone(model_name, cfg, transformers_mod):
    from packaging.version import Version

    model_type = getattr(cfg, "model_type", "") or ""
    if not _is_qwen35_moe(model_name, cfg) and "qwen3_5_moe" not in model_type:
        return
    if Version(transformers_mod.__version__) < Version("5.2.0"):
        raise RuntimeError(f"AutoEP qwen3_5_moe needs Transformers >= 5.2.0, got {transformers_mod.__version__}")
    if model_type != "qwen3_5_moe_text":
        raise RuntimeError("AutoEP qwen3_5_moe needs text-backbone model_type='qwen3_5_moe_text', "
                           f"got {model_type!r} (use the text config, not the multimodal parent)")


def _is_qwen35_moe(model_name, cfg):
    if not _is_qwen35(model_name, cfg):
        return False
    name = (model_name or "").lower()
    model_type = getattr(cfg, "model_type", "") or ""
    return ("moe" in model_type or "a3b" in name or "moe" in name)


def _decoder_module(model):
    # CausalLM / ZeRO engine forward still runs lm_head and allocates [B,S,V].
    root = getattr(model, "module", model)
    inner = getattr(root, "model", None)
    if inner is not None and hasattr(inner, "layers"):
        return inner
    if inner is not None and hasattr(inner, "language_model"):
        return inner.language_model
    if hasattr(root, "language_model"):
        return root.language_model
    raise TypeError(f"sum_loss: no text decoder on {type(root).__name__}")


def _wrap_liger_ce_for_zero3():
    """Liger fused CE reads lm_head.weight without lm_head.forward, so Z3 never gathers it."""
    from liger_kernel.transformers.model import loss_utils
    from deepspeed.runtime.zero.linear import zero3_gathered_param

    orig = getattr(loss_utils, "LigerForCausalLMLoss", None)
    if orig is None or getattr(orig, "_ds_zero3_gathered", False):
        return

    def wrapped(hidden_states, lm_head_weight, *args, **kwargs):
        return orig(hidden_states, zero3_gathered_param(lm_head_weight), *args, **kwargs)

    wrapped._ds_zero3_gathered = True
    loss_utils.LigerForCausalLMLoss = wrapped


def _apply_fused_ce(model, model_name, cfg):
    # FLCE only: leave RMS/SwiGLU to HF (and AutoEP GroupedExperts).
    kwargs = dict(cross_entropy=False, fused_linear_cross_entropy=True, rms_norm=False, swiglu=False, model=model)
    if _is_qwen35_moe(model_name, cfg):
        try:
            from liger_kernel.transformers import apply_liger_kernel_to_qwen3_5_moe
        except ImportError as e:
            print("[child] fused_ce skip: liger has no apply_liger_kernel_to_qwen3_5_moe; "
                  f"{e}", file=sys.stderr)
            return False
        try:
            apply_liger_kernel_to_qwen3_5_moe(rope=False, **kwargs)
        except Exception as e:
            print(f"[child] fused_ce skip: {type(e).__name__}: {e}", file=sys.stderr)
            return False
        print(f"[child] fused_ce: liger qwen3_5_moe FLCE on {type(model).__name__}", file=sys.stderr)
        _wrap_liger_ce_for_zero3()
        return True
    if _is_qwen35(model_name, cfg):
        try:
            from liger_kernel.transformers import apply_liger_kernel_to_qwen3_5
        except ImportError as e:
            raise RuntimeError("fused_ce for Qwen3.5 requires liger apply_liger_kernel_to_qwen3_5; "
                               "upgrade liger-kernel") from e
        try:
            apply_liger_kernel_to_qwen3_5(rope=False, **kwargs)
        except TypeError:
            apply_liger_kernel_to_qwen3_5(**kwargs)
        except Exception as e:
            print(f"[child] fused_ce skip: {type(e).__name__}: {e}", file=sys.stderr)
            return False
        _wrap_liger_ce_for_zero3()
        return True
    name = (model_name or "").lower()
    model_type = getattr(cfg, "model_type", "") or ""
    is_qwen3_moe = ("moe" in model_type or "a3b" in name or "moe" in name)
    if is_qwen3_moe:
        try:
            from liger_kernel.transformers import apply_liger_kernel_to_qwen3_moe
        except ImportError:
            apply_liger_kernel_to_qwen3_moe = None
        if apply_liger_kernel_to_qwen3_moe is not None:
            try:
                apply_liger_kernel_to_qwen3_moe(rope=False, **kwargs)
            except TypeError:
                apply_liger_kernel_to_qwen3_moe(**kwargs)
            except Exception as e:
                print(f"[child] fused_ce skip: {type(e).__name__}: {e}", file=sys.stderr)
                return False
            print(f"[child] fused_ce: liger qwen3_moe FLCE on {type(model).__name__}", file=sys.stderr)
            _wrap_liger_ce_for_zero3()
            return True
    from liger_kernel.transformers import apply_liger_kernel_to_qwen3
    try:
        apply_liger_kernel_to_qwen3(rope=False, **kwargs)
    except TypeError:
        apply_liger_kernel_to_qwen3(**kwargs)
    _wrap_liger_ce_for_zero3()
    return True


def _host_mem_bytes():
    mem_total = mem_avail = None
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemTotal:"):
                mem_total = int(line.split()[1]) * 1024
            elif line.startswith("MemAvailable:"):
                mem_avail = int(line.split()[1]) * 1024
    return mem_total, mem_avail


def _log_host(stage, extra_msg=""):
    total, avail = _host_mem_bytes()
    rss = RssSampler._rss()

    def gb(n):
        return n / (1024**3) if n is not None else float("nan")

    extra = ""
    if extra_msg:
        extra = f" {extra_msg}"
    print(f"[child] host {stage}: rss={gb(rss):.1f}GiB avail={gb(avail):.1f}GiB total={gb(total):.1f}GiB{extra}",
          file=sys.stderr,
          flush=True)


class RssSampler(threading.Thread):

    def __init__(self, interval=0.02):
        super().__init__(daemon=True)
        self.interval = interval
        self._stop_evt = threading.Event()
        self.peak = self._rss()

    @staticmethod
    def _rss():
        try:
            import psutil
            return psutil.Process().memory_info().rss
        except Exception:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) * 1024
        return 0

    def run(self):
        while not self._stop_evt.is_set():
            self.peak = max(self.peak, self._rss())
            time.sleep(self.interval)

    def stop(self):
        self._stop_evt.set()
        self.join(timeout=1.0)
        self.peak = max(self.peak, self._rss())
        return self.peak


def _storage_bytes(tensors):
    """Sum distinct CUDA storages so flat buffers and their views count once."""
    import torch

    seen = {}
    for tensor in tensors:
        if not torch.is_tensor(tensor):
            continue
        try:
            if not tensor.is_cuda:
                continue
            storage = tensor.untyped_storage()
        except Exception:
            continue
        seen[storage.data_ptr()] = storage.nbytes()
    return seen


def _iter_tensors(obj, depth=0):
    import torch

    if depth > 3:
        return
    if torch.is_tensor(obj):
        yield obj
    elif isinstance(obj, dict):
        for value in obj.values():
            yield from _iter_tensors(value, depth + 1)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            yield from _iter_tensors(value, depth + 1)


def _install_checkpoint_probe(model):
    """Count checkpointable-layer forwards; recompute doubles the count when checkpointing works."""
    state = {"layers": 0, "enabled": 0, "calls": 0}

    def bump(*_args):
        state["calls"] += 1

    reported = set()
    for module in model.modules():
        if not hasattr(module, "gradient_checkpointing"):
            continue
        state["layers"] += 1
        state["enabled"] += int(bool(module.gradient_checkpointing))
        module.register_forward_hook(bump)
        cls = type(module).__name__
        if cls not in reported:
            reported.add(cls)
            func = getattr(module, "_gradient_checkpointing_func", None)
            print(
                f"[ckpt] class={cls} training={module.training} "
                f"gradient_checkpointing={module.gradient_checkpointing} "
                f"func={getattr(func, 'func', func)} "
                f"keywords={getattr(func, 'keywords', None)} "
                f"can_checkpoint_with_cache={getattr(module, '_can_checkpoint_with_cache', None)}",
                file=sys.stderr,
                flush=True)
    return state


def _install_layer_workspace_probe(model):
    """High-water of one full-attention mixer and one GDN mixer during rematerialize.

    The delta is peak allocated during that mixer forward minus bytes already
    live at entry. reset_peak_memory_stats would otherwise drop the process
    peak, so the caller must fold ``tracked_peak`` back into the reported max.
    """
    import torch

    state = {
        "in_backward": False,
        "tracked_peak": 0,
        "full_attention_bytes": 0,
        "linear_attention_bytes": 0,
        "full_attention_calls": 0,
        "linear_attention_calls": 0,
    }
    found = {}
    for mod in model.modules():
        if not _is_text_decoder_layer(mod):
            continue
        if hasattr(mod, "linear_attn") and "linear_attention" not in found:
            found["linear_attention"] = mod.linear_attn
        if hasattr(mod, "self_attn") and "full_attention" not in found:
            found["full_attention"] = mod.self_attn
    if not found:
        raise RuntimeError("layer_workspace_probe could not find a full_attention or linear_attention mixer")
    if "linear_attention" not in found:
        print("[layer-workspace] no linear_attention mixer; probing full attention only",
              file=sys.stderr,
              flush=True)

    def _wrap(mod, kind):
        orig = mod.forward
        bytes_key = f"{kind}_bytes"
        calls_key = f"{kind}_calls"

        def probed(*args, **kwargs):
            if not state["in_backward"]:
                return orig(*args, **kwargs)
            before = torch.cuda.memory_allocated()
            state["tracked_peak"] = max(state["tracked_peak"], torch.cuda.max_memory_allocated())
            torch.cuda.reset_peak_memory_stats()
            try:
                return orig(*args, **kwargs)
            finally:
                hi = torch.cuda.max_memory_allocated()
                delta = hi - before
                if delta > state[bytes_key]:
                    state[bytes_key] = delta
                state[calls_key] += 1
                state["tracked_peak"] = max(state["tracked_peak"], hi)

        mod.forward = probed
        print(
            f"[layer-workspace] probing {kind} module={type(mod).__name__}",
            file=sys.stderr,
            flush=True,
        )

    for kind, mod in found.items():
        _wrap(mod, kind)
    return state


def _dump_mem_census(tag, engine):
    """Break resident GPU bytes down by owner to expose duplicate weight copies."""
    import gc
    import torch
    import deepspeed.comm as dist
    from deepspeed.moe.utils import is_moe_param

    def gib(nbytes):
        return nbytes / (1024**3)

    def emit(msg):
        print(f"[census] {tag} {msg}", file=sys.stderr, flush=True)

    expert, dense = [], []
    for _, param in engine.module.named_parameters():
        (expert if is_moe_param(param) else dense).append(param.data)
    emit(f"params expert={gib(sum(_storage_bytes(expert).values())):.3f}GiB "
         f"dense={gib(sum(_storage_bytes(dense).values())):.3f}GiB")

    opt = getattr(engine, "optimizer", None)
    bit16_groups = getattr(opt, "bit16_groups", None)
    if bit16_groups is not None:
        moe_flags = getattr(opt, "is_moe_param_group", [])
        for idx, group in enumerate(bit16_groups):
            name = opt.optimizer.param_groups[idx].get("name", f"group{idx}")
            is_moe = bool(moe_flags[idx]) if idx < len(moe_flags) else False
            dp_world = dist.get_world_size(group=opt.real_dp_process_group[idx])
            flat = opt.bit16_groups_flat[idx]
            fp32 = opt.single_partition_of_fp32_groups[idx]
            emit(f"group{idx} name={name} moe={is_moe} dp_world={dp_world} params={len(group)} "
                 f"flat_bf16={gib(flat.numel() * flat.element_size()):.3f}GiB "
                 f"fp32_partition={gib(fp32.numel() * fp32.element_size()):.3f}GiB")

    inner = getattr(opt, "optimizer", None)
    if inner is not None:
        per_state = {}
        for state in inner.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    per_state.setdefault(key, []).append(value)
        for key in sorted(per_state):
            emit(f"optstate {key}={gib(sum(_storage_bytes(per_state[key]).values())):.3f}GiB")

    for attr in ("averaged_gradients", "grads_in_partition", "ipg_buffer", "bit16_groups_flat"):
        holder = getattr(opt, attr, None)
        if holder is None:
            continue
        total = sum(_storage_bytes(_iter_tensors(holder)).values())
        emit(f"buffer {attr}={gib(total):.3f}GiB")

    param_ptrs = set(_storage_bytes(expert + dense))
    live, by_shape, sample = {}, {}, {}
    for obj in gc.get_objects():
        try:
            if not torch.is_tensor(obj) or not obj.is_cuda:
                continue
            storage = obj.untyped_storage()
            ptr = storage.data_ptr()
            if ptr in live:
                continue
            nbytes = storage.nbytes()
            key = (tuple(obj.shape), str(obj.dtype))
        except Exception:
            continue
        live[ptr] = nbytes
        entry = by_shape.setdefault(key, [0, 0])
        entry[0] += 1
        entry[1] += nbytes
        if key not in sample and ptr not in param_ptrs:
            sample[key] = obj
    emit(f"live_cuda_storages={gib(sum(live.values())):.3f}GiB "
         f"memory_allocated={gib(torch.cuda.memory_allocated()):.3f}GiB")
    ranked = sorted(by_shape.items(), key=lambda kv: -kv[1][1])[:12]
    for (shape, dtype), (count, nbytes) in ranked:
        emit(f"top shape={shape} dtype={dtype} count={count} bytes={gib(nbytes):.3f}GiB")

    # Tensors with no Python referrer are either autograd-owned or trapped in an
    # uncollected reference cycle; a forced collection tells the two apart.
    before_collect = torch.cuda.memory_allocated()
    collected = gc.collect()
    after_collect = torch.cuda.memory_allocated()
    emit(f"gc_collect objects={collected} freed={gib(before_collect - after_collect):.3f}GiB "
         f"allocated_after={gib(after_collect):.3f}GiB")

    # Naming the holder is what separates a real leak from normal step-local reuse.
    for (shape, dtype), _ in ranked[:6]:
        tensor = sample.get((shape, dtype))
        if tensor is None:
            continue
        holders = []
        for referrer in gc.get_referrers(tensor)[:6]:
            if isinstance(referrer, dict):
                keys = [k for k, v in referrer.items() if v is tensor]
                owners = [type(o).__name__ for o in gc.get_referrers(referrer)[:3]]
                holders.append(f"dict{keys}@{owners}")
            else:
                holders.append(type(referrer).__name__)
        emit(f"holders shape={shape} dtype={dtype} -> {holders}")


def _dump_alloc_stats(tag):
    import torch
    stats = torch.cuda.memory_stats()
    allocated = stats.get("allocated_bytes.all.peak", 0) / (1024**3)
    reserved = stats.get("reserved_bytes.all.peak", 0) / (1024**3)
    print(
        f"[alloc] {tag} retries={stats.get('num_alloc_retries', 0)} "
        f"ooms={stats.get('num_ooms', 0)} "
        f"alloc_peak_gb={allocated:.2f} reserved_peak_gb={reserved:.2f} "
        f"frag_gb={reserved - allocated:.2f}",
        file=sys.stderr,
        flush=True,
    )
    return {
        "num_alloc_retries": int(stats.get("num_alloc_retries", 0) or 0),
        "num_ooms": int(stats.get("num_ooms", 0) or 0),
        "reserved_peak_gb": reserved,
        "frag_gb": reserved - allocated,
    }


def run_child(args):
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29556")

    import torch
    import deepspeed
    import deepspeed.runtime.activation_checkpointing.checkpointing as ds_ckpt
    from deepspeed.accelerator import get_accelerator
    from transformers import AutoConfig, AutoModelForCausalLM, GradientCheckpointingLayer
    from deepspeed.runtime.activation_checkpointing.offload_activations import (
        _ActivationOffloadEngine,
        get_checkpoint_hidden_states_offloading_ctx_manager,
    )

    mode = args.mode
    seqlen = args.seqlen
    mbs = int(args.mbs)
    gas = int(args.gas)
    if mbs < 1 or gas < 1:
        raise ValueError(f"mbs and gas must be >= 1, got mbs={mbs} gas={gas}")
    packed_seq = bool(args.packed_seq)
    pack_lens = []
    if packed_seq:
        if mbs != 1:
            raise ValueError("--packed_seq requires --mbs 1 (a packed row is a single physical row)")
        pack_lens = [int(x) for x in args.pack_lens.split(",") if x.strip()]
        if not pack_lens:
            raise ValueError("--packed_seq requires --pack_lens (comma-separated per-sample token lengths)")
        seqlen = sum(pack_lens)
    use_packed_tiled_remat = int(args.packed_tiled_remat_shards) > 1 or int(args.packed_tiled_remat_budget) > 0
    seq_chain_budget = int(args.seq_chain_budget)
    if use_packed_tiled_remat and not packed_seq:
        raise ValueError("--packed_tiled_remat_shards/--packed_tiled_remat_budget requires --packed_seq")
    if use_packed_tiled_remat and int(args.tiled_rematerialize_shards) > 1:
        raise ValueError("--packed_tiled_remat_* is mutually exclusive with --tiled_rematerialize_shards")
    if seq_chain_budget < 0:
        raise ValueError(f"--seq_chain_budget must be >= 0, got {seq_chain_budget}")
    if seq_chain_budget > 0 and (packed_seq or use_packed_tiled_remat):
        raise ValueError("--seq_chain_budget is the single dense sample; a packed row uses --packed_tiled_remat_*")
    if seq_chain_budget > 0 and mbs != 1:
        raise ValueError("--seq_chain_budget requires --mbs 1 (one sequence per row)")
    iters = args.iters or (5 if seqlen >= 32768 else 10)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank0 = local_rank == 0
    tokens_per_rank = mbs * gas * seqlen
    accel = get_accelerator()
    device = torch.device(accel.device_name(local_rank))
    accel.set_device(local_rank)

    dev_name = torch.cuda.get_device_name(local_rank)
    total_gb = torch.cuda.get_device_properties(local_rank).total_memory / (1024**3)
    if rank0:
        print(
            f"[child] mode={mode} seqlen={seqlen} mbs={mbs} gas={gas} iters={iters} "
            f"world_size={world_size} device={dev_name} total={total_gb:.1f}GiB",
            file=sys.stderr)

    native = mode in ("ds_async", "ds_async_prefetch", "ds_blocking")
    zero3_any = args.zero3 or args.zero3_offload
    if sum([args.zero2, args.zero2_offload, zero3_any]) > 1:
        raise ValueError("choose one of --zero2 / --zero2_offload / --zero3 / --zero3_offload")
    zero_stage = 3 if zero3_any else (2 if (args.zero2 or args.zero2_offload) else 0)
    use_zero = zero_stage > 0
    zero2 = zero_stage == 2
    offload_opt = bool(args.zero2_offload or args.zero3_offload)
    if native and not use_zero:
        deepspeed.init_distributed(dist_backend=accel.communication_backend_name(), auto_mpi_discovery=False)

    result = {
        "mode": f"{mode}+chain{seq_chain_budget}" if seq_chain_budget > 0 else mode,
        "seqlen": seqlen,
        "mbs": mbs,
        "gas": gas,
        "model_name": args.model_name,
        "fused_ce": bool(args.fused_ce),
        "sum_loss": bool(args.sum_loss),
        "zero2": zero2,
        "zero3": zero_stage == 3,
        "zero_stage": zero_stage,
        "zero2_offload": offload_opt,
        "nproc": world_size,
        "opt_pin_memory": bool(args.opt_pin_memory),
        "act_pin_memory": bool(args.act_pin_memory),
        "tiled_mlp_shards": int(args.tiled_mlp_shards),
        "tiled_rematerialize_shards": int(args.tiled_rematerialize_shards),
        "tiled_fwd": bool(args.tiled_fwd),
        "packed_seq": packed_seq,
        "pack_lens": pack_lens,
        "packed_tiled_remat_shards": int(args.packed_tiled_remat_shards),
        "packed_tiled_remat_budget": int(args.packed_tiled_remat_budget),
        "seq_chain_budget": seq_chain_budget,
        "gc_reentrant": bool(args.gc_reentrant),
        "autoep_size": int(args.autoep_size),
        "autoep_preset": args.autoep_preset,
        "cpu_optimizer": args.cpu_optimizer if offload_opt else ("GPU_AdamW" if use_zero else args.cpu_optimizer),
        "device": dev_name,
        "total_gb": round(total_gb, 1),
    }
    try:
        parent_cfg = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
        cfg = _text_config(parent_cfg)
        if args.num_layers > 0:
            cfg.num_hidden_layers = args.num_layers
        cfg.use_cache = False
        if hasattr(cfg, "output_router_logits"):
            cfg.output_router_logits = False
        # Note: setting cfg.attn_implementation here is a no-op on this
        # transformers version (it does not propagate to _attn_implementation);
        # the real switch is the attn_implementation= kwarg passed to
        # from_config below.
        if seqlen > cfg.max_position_embeddings:
            cfg.max_position_embeddings = seqlen
        print(
            f"[child] {args.model_name}: layers={cfg.num_hidden_layers} hidden={cfg.hidden_size} "
            f"intermediate={getattr(cfg, 'intermediate_size', None)} "
            f"vocab={cfg.vocab_size} max_pos={cfg.max_position_embeddings} "
            f"fused_ce={args.fused_ce} sum_loss={args.sum_loss} "
            f"zero_stage={zero_stage} offload_opt={offload_opt} "
            f"mbs={mbs} gas={gas} nproc={world_size} "
            f"opt_pin={args.opt_pin_memory} act_pin={args.act_pin_memory} "
            f"tiled_mlp={args.tiled_mlp_shards} "
            f"tiled_remat={args.tiled_rematerialize_shards} tiled_fwd={bool(args.tiled_fwd)} "
            f"packed_seq={packed_seq} pack_lens={pack_lens} "
            f"packed_tiled_remat={args.packed_tiled_remat_shards}/{args.packed_tiled_remat_budget} "
            f"seq_chain_budget={seq_chain_budget} "
            f"autoep={args.autoep_size}/{args.autoep_preset} "
            f"cpu_opt={args.cpu_optimizer} model_type={getattr(cfg, 'model_type', None)}",
            file=sys.stderr)
        _log_host("config")

        # Allocate bf16 on the GPU so we never hold a host fp32 copy (~102 GiB at 27B).
        from_cfg_kwargs = dict(trust_remote_code=True, attn_implementation="flash_attention_2")
        try:
            with torch.device(device):
                model = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16, **from_cfg_kwargs)
        except TypeError:
            with torch.device(device):
                model = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.bfloat16, **from_cfg_kwargs)
        _log_host("from_config")
        if next(model.parameters()).device != device:
            model = model.to(device=device, dtype=torch.bfloat16)
        _log_host("to_device")
        model.train()
        gc_kwargs = {"use_reentrant": bool(args.gc_reentrant)}
        use_tiled_remat = int(args.tiled_rematerialize_shards) > 1
        if use_tiled_remat:
            # TiledRematerialize wraps the whole decoder layer body, so it is
            # already a checkpoint for the layer. An outer HF/DS wrap would
            # rematerialize the layer body twice (rerun tiled forward, then
            # tiled backward remat).
            print("[child] skip outer gradient checkpointing; tiled rematerialize is the checkpoint", file=sys.stderr)
        elif mode == "hf_gc_offload":
            # transformers#48444: wrap checkpoint with save_on_cpu(pin_memory=True).
            try:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gc_kwargs, offload=True)
            except TypeError as e:
                raise RuntimeError("hf_gc_offload needs transformers with "
                                   "gradient_checkpointing_enable(..., offload=True) "
                                   "(huggingface/transformers#48444)") from e
        else:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gc_kwargs)
        if use_packed_tiled_remat:
            # Unlike TiledRematerialize, this only wraps self_attn/linear_attn,
            # not the whole decoder layer body (mlp/norms/residuals) -- so the
            # outer per-layer checkpoint above must stay enabled; packed-seq
            # tiling additionally shrinks the attention workspace inside each
            # layer's (outer-checkpoint) recompute.
            print(
                "[child] packed-seq tiled rematerialize wraps self_attn/linear_attn only; "
                "outer gradient checkpointing (if any, per --mode) stays enabled",
                file=sys.stderr)

        if args.sum_loss:
            args.fused_ce = False
        if args.fused_ce:
            if not _apply_fused_ce(model, args.model_name, cfg):
                raise RuntimeError("fused_ce/liger requested but did not apply")
        _log_host("fused_ce")
        autoep_size = int(args.autoep_size)
        if autoep_size < 0:
            raise ValueError("autoep_size must be >= 0")
        if autoep_size > 0:
            import transformers
            if not use_zero:
                raise ValueError("AutoEP requires --zero2, --zero2_offload, or --zero3")
            if int(args.tiled_mlp_shards) > 1:
                print("[child] AutoEP + tiled_mlp shards="
                      f"{int(args.tiled_mlp_shards)}", file=sys.stderr)
            if autoep_size > 1 and world_size % autoep_size != 0:
                raise ValueError(f"autoep_size={autoep_size} must divide world_size={world_size}")
            _require_autoep_text_backbone(args.model_name, cfg, transformers)
            if int(args.tiled_rematerialize_shards) > 1 and getattr(cfg, "output_router_logits", False):
                raise RuntimeError("tiled remat + output_router_logits is unsupported "
                                   "(aux logits are captured on the no_grad full forward)")
        if int(args.tiled_mlp_shards) > 1:
            _install_tiled_mlp(model, int(args.tiled_mlp_shards))
            _log_host("tiled_mlp")
        tiled_remat_classes = []
        if int(args.tiled_rematerialize_shards) > 1:
            tiled_remat_classes = _install_tiled_rematerialize(model,
                                                               int(args.tiled_rematerialize_shards),
                                                               tile_forward=bool(args.tiled_fwd))
            if args.offload_reclaim_probe:
                _install_offload_reclaim_probe()
            _log_host("tiled_rematerialize")
        packed_remat_classes = []
        if use_packed_tiled_remat:
            from deepspeed.runtime.activation_checkpointing.hf_tiled_rematerialization import (
                install_packed_seq_tiled_rematerialize)
            packed_remat_classes = install_packed_seq_tiled_rematerialize(
                model,
                token_budget=int(args.packed_tiled_remat_budget) or None,
                shards=int(args.packed_tiled_remat_shards) or None)
            if args.offload_reclaim_probe:
                _install_offload_reclaim_probe()
            _log_host("packed_seq_tiled_rematerialize")
        if seq_chain_budget > 0:
            # linear_attn only. The decoder-layer checkpoint from --mode stays:
            # the chain shrinks the GDN workspace inside that recompute.
            _install_seq_chain_tile(model, seq_chain_budget)
            print(
                "[child] seq_chain_tile wraps linear_attn only; "
                "outer gradient checkpointing (if any, per --mode) stays enabled",
                file=sys.stderr)
            _log_host("seq_chain_tile")
        if zero_stage == 3:
            from deepspeed.utils import set_z3_leaf_modules
            leaf_classes = []
            for mod in model.modules():
                name = type(mod).__name__
                if "GatedDeltaNet" in name or name.endswith("DeltaNet"):
                    if type(mod) not in leaf_classes:
                        leaf_classes.append(type(mod))
            # Required, not an optimization: the tiled forward calls each mixer
            # once per shard, so without leafing, ZeRO-3 re-runs its per-submodule
            # hooks per shard and registers the same param into a grad bucket
            # twice, tripping the ds_id uniqueness assert in stage3.py. Leafing
            # gathers once per layer entry, which also stops weight re-fetching.
            # Packed tiling calls each mixer once per tile, same ds_id hazard as
            # the batch-axis wrap. GatedDeltaNet is already name-matched above;
            # full-attention mixers are not, so keep the installer's classes too.
            for cls in list(tiled_remat_classes) + list(packed_remat_classes):
                if cls not in leaf_classes:
                    leaf_classes.append(cls)
            if leaf_classes:
                set_z3_leaf_modules(model, leaf_classes, raise_if_not_found=False)
                print(f"[child] z3 leaf classes={[c.__name__ for c in leaf_classes]}", file=sys.stderr)
            _log_host("z3_leaf")

        ds_engine = None
        if use_zero:
            if offload_opt and args.cpu_optimizer == "SGD":
                opt_block = {"type": "SGD", "params": {"lr": 1e-5}}
            else:
                opt_block = {"type": "AdamW", "params": {"lr": 1e-5, "betas": [0.9, 0.95]}}
            zero_opt = {
                "stage": zero_stage,
                "contiguous_gradients": True,
            }
            if offload_opt:
                zero_opt["offload_optimizer"] = {
                    "device": "cpu",
                    "pin_memory": bool(args.opt_pin_memory),
                }
            ds_config = {
                "train_micro_batch_size_per_gpu": mbs,
                "gradient_accumulation_steps": gas,
                "steps_per_print": 1000,
                "bf16": {
                    "enabled": True
                },
                "optimizer": opt_block,
                "zero_force_ds_cpu_optimizer": bool(offload_opt and args.cpu_optimizer == "AdamW"),
                "zero_allow_untested_optimizer": bool(offload_opt and args.cpu_optimizer != "AdamW"),
                "zero_optimization": zero_opt,
            }
            if autoep_size > 0:
                ds_config["expert_parallel"] = {
                    "enabled": True,
                    "autoep_size": autoep_size,
                    "preset_model": _autoep_preset_name(args.model_name, cfg, args.autoep_preset),
                    "load_balance_coeff": None,
                    "use_grouped_mm": True,
                }
                print(
                    f"[child] AutoEP preset={ds_config['expert_parallel']['preset_model']} "
                    f"size={autoep_size} grouped_mm=True",
                    file=sys.stderr)
            if args.comms_logger:
                ds_config["comms_logger"] = {
                    "enabled": True,
                    "verbose": False,
                    "prof_all": True,
                    "debug": False,
                }
            ds_engine, _, _, _ = deepspeed.initialize(model=model,
                                                      model_parameters=model.parameters(),
                                                      config=ds_config)
            model = ds_engine.module
            _log_host("initialize")
            if args.mem_census and rank0:
                _dump_mem_census("after_init", ds_engine)

        ctx_factory = None
        dss_manager = None
        hf_manager = None
        if mode == "dss":
            from autorun.vendor_dss.activation_offload import install_activation_offload
            dss_manager = install_activation_offload(model)
        elif mode in ("hf_offload", "hf_prefetch"):
            hf_manager = get_checkpoint_hidden_states_offloading_ctx_manager(prefetch=(mode == "hf_prefetch"),
                                                                             use_pin_memory=bool(args.act_pin_memory))
            ctx_factory = lambda: hf_manager  # noqa: E731
        elif native:
            if mode == "ds_async_prefetch":
                ds_ckpt._cpu_offload_prefetch = True
            if mode == "ds_blocking":
                # Force the pre-engine path: blocking copy_to_device / .cpu().
                ds_ckpt._get_cpu_offload_engine = lambda: None
            else:
                ds_ckpt._cpu_offload_engine = _ActivationOffloadEngine(
                    keep_last_count=1,
                    min_offload_bytes=0,
                    use_pin_memory=bool(args.act_pin_memory),
                    use_streams=bool(args.act_use_streams),
                    prefetch=bool(ds_ckpt._cpu_offload_prefetch),
                )
            deepspeed.checkpointing.configure(mpu_=None, checkpoint_in_cpu=True)
            if not use_tiled_remat:
                for m in model.modules():
                    if isinstance(m, GradientCheckpointingLayer):
                        m.gradient_checkpointing = True
                        m._gradient_checkpointing_func = ds_ckpt.non_reentrant_checkpoint

        opt = None
        if not use_zero:
            opt = torch.optim.AdamW(model.parameters(), lr=1e-5, betas=(0.9, 0.95))

        gen = torch.Generator().manual_seed(0)
        packed_kwargs = {}
        if packed_seq:
            ids = torch.randint(0, cfg.vocab_size, (1, seqlen), generator=gen, dtype=torch.long).to(device)
            labels = ids.clone()
            position_ids = torch.cat([torch.arange(n, dtype=torch.long) for n in pack_lens]).unsqueeze(0).to(device)
            cu = [0]
            for n in pack_lens:
                cu.append(cu[-1] + n)
            cu_seqlens = torch.tensor(cu, dtype=torch.int32, device=device)
            max_len_in_row = int(max(pack_lens))
            packed_kwargs = {
                "position_ids": position_ids,
                "cu_seq_lens_q": cu_seqlens,
                "cu_seq_lens_k": cu_seqlens,
                "max_length_q": max_len_in_row,
                "max_length_k": max_len_in_row,
            }
            linear_attn_classes = {type(m.linear_attn) for m in model.modules() if hasattr(m, "linear_attn")}
            if linear_attn_classes:
                # transformers never derives seq_idx from cu_seq_lens_q, and
                # causal_conv1d only resets the short conv between samples with it.
                seq_idx = torch.cat([torch.full((n, ), i, dtype=torch.int32) for i, n in enumerate(pack_lens)])
                packed_kwargs["seq_idx"] = seq_idx.unsqueeze(0).to(device)
            numerics_gaps = []
            for cls in linear_attn_classes:
                gaps = _gdn_boundary_reset_gaps(cls)
                if gaps:
                    numerics_gaps.extend(gaps)
                    print(f"[child] WARNING packed_seq: {cls.__name__} leaks state across packed samples "
                          "(numerics wrong, memory/throughput still measurable): " + "; ".join(gaps),
                          file=sys.stderr)
            result["numerics_valid"] = not numerics_gaps
            if numerics_gaps:
                result["numerics_gaps"] = numerics_gaps
        else:
            ids = torch.randint(0, cfg.vocab_size, (mbs, seqlen), generator=gen, dtype=torch.long).to(device)
            labels = ids.clone()

        import contextlib

        fwd = ds_engine if ds_engine is not None else model

        def _offload_msg():
            eng = None
            if dss_manager is not None:
                eng = dss_manager
            elif hf_manager is not None:
                eng = hf_manager
            elif native:
                eng = getattr(ds_ckpt, "_cpu_offload_engine", None)
            if eng is None or not hasattr(eng, "stats"):
                return ""
            st = eng.stats
            nbytes = getattr(st, "offloaded_bytes", 0) or 0
            return f"offloaded={st.offloaded_tensors} bytes={nbytes / (1024**3):.2f}GiB"

        decoder = _decoder_module(model) if args.sum_loss else None
        lm_head_calls = [0]
        if args.sum_loss and hasattr(model, "lm_head") and model.lm_head is not None:

            def _count_lm_head(_mod, _inp, _out):
                lm_head_calls[0] += 1

            model.lm_head.register_forward_hook(_count_lm_head)

        def global_step(trace_first=False):
            if args.gc_each_step:
                import gc as _gc
                _gc.collect()
            if trace_first:
                _log_host("step_start", _offload_msg())
            if ds_engine is None:
                opt.zero_grad(set_to_none=True)
            last_loss = None
            for _ in range(gas):
                ctx = ctx_factory() if ctx_factory is not None else contextlib.nullcontext()
                with ctx:
                    if args.sum_loss:
                        dec_out = decoder(input_ids=ids, use_cache=False, **packed_kwargs)
                        hs = dec_out.last_hidden_state
                        if trace_first:
                            hs_gb = hs.numel() * hs.element_size() / (1024**3)
                            sv_gb = ids.numel() * int(cfg.vocab_size) * 2 / (1024**3)
                            print(
                                f"[nologits] decoder={type(decoder).__name__} "
                                f"root={type(model).__name__} hs.shape={tuple(hs.shape)} "
                                f"dtype={hs.dtype} hs_gb={hs_gb:.3f} "
                                f"hidden={int(cfg.hidden_size)} vocab={int(cfg.vocab_size)} "
                                f"[B,S,V]_bf16={sv_gb:.2f}GiB",
                                file=sys.stderr,
                                flush=True,
                            )
                        if hs.shape[-1] != int(cfg.hidden_size) or hs.shape[-1] == int(cfg.vocab_size):
                            raise RuntimeError(f"sum_loss got last dim {hs.shape[-1]} "
                                               f"(hidden={cfg.hidden_size} vocab={cfg.vocab_size})")
                        last_loss = hs.float().sum()
                    else:
                        out = fwd(input_ids=ids, labels=labels, **packed_kwargs)
                        last_loss = out.loss
                    if trace_first:
                        _log_host("after_fwd", _offload_msg())
                    if ds_engine is not None:
                        if layer_probe is not None:
                            layer_probe["in_backward"] = True
                        try:
                            ds_engine.backward(last_loss)
                        finally:
                            if layer_probe is not None:
                                layer_probe["in_backward"] = False
                        if trace_first:
                            _log_host("after_bwd", _offload_msg())
                        ds_engine.step()
                        if trace_first:
                            _log_host("after_opt", _offload_msg())
                    else:
                        if layer_probe is not None:
                            layer_probe["in_backward"] = True
                        try:
                            (last_loss / gas).backward()
                        finally:
                            if layer_probe is not None:
                                layer_probe["in_backward"] = False
                        if trace_first:
                            _log_host("after_bwd", _offload_msg())
            if ds_engine is None:
                opt.step()
                if trace_first:
                    _log_host("after_opt", _offload_msg())
            if args.sum_loss and lm_head_calls[0] != 0:
                raise RuntimeError(f"sum_loss: lm_head ran {lm_head_calls[0]} times")
            return last_loss

        ckpt_probe = _install_checkpoint_probe(model) if (args.mem_census and rank0) else None
        layer_probe = _install_layer_workspace_probe(model) if (args.layer_workspace_probe and rank0) else None

        # Blocks allocated before recording starts carry no stack, so retained
        # state is only attributable if recording covers the first step.
        recording = bool(args.mem_snapshot) and rank0 and args.mem_snapshot_early
        if recording:
            torch.cuda.memory._record_memory_history(enabled="all",
                                                     context="all",
                                                     stacks="python",
                                                     max_entries=1000000)

        for i in range(args.warmup):
            global_step(trace_first=(i == 0))
            if i == 0:
                _log_host("first_warmup_step", _offload_msg())
                if ckpt_probe is not None:
                    # Not every module carrying the flag is a checkpointed layer (the text
                    # model carries it too), so any excess over one call each means recompute.
                    print(
                        f"[ckpt] layers={ckpt_probe['layers']} flag_enabled={ckpt_probe['enabled']} "
                        f"forward_calls={ckpt_probe['calls']} "
                        f"recompute_active={ckpt_probe['calls'] > ckpt_probe['layers']}",
                        file=sys.stderr,
                        flush=True)
        accel.synchronize()
        if rank0:
            _dump_alloc_stats("after_warmup")
            if args.mem_census and ds_engine is not None:
                _dump_mem_census("after_warmup", ds_engine)
        if args.comms_logger:
            import deepspeed.comm as dist
            dist.reset_log()
        if args.offload_reclaim_probe:
            _RECLAIM_PROBE["calls"].clear()
            _RECLAIM_PROBE["enabled"] = True

        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        snapshot_path = args.mem_snapshot if rank0 else ""
        if snapshot_path and not recording:
            torch.cuda.memory._record_memory_history(enabled="all",
                                                     context="all",
                                                     stacks="python",
                                                     max_entries=1000000)
        rss = RssSampler()
        rss_base = rss.peak
        rss.start()

        loss_val = None
        t0 = time.perf_counter()
        for _ in range(iters):
            loss_val = global_step()
        accel.synchronize()
        elapsed = time.perf_counter() - t0

        if args.offload_reclaim_probe:
            _RECLAIM_PROBE["enabled"] = False
            if rank0:
                _dump_reclaim_probe("timed")

        if snapshot_path:
            os.makedirs(os.path.dirname(snapshot_path) or ".", exist_ok=True)
            torch.cuda.memory._dump_snapshot(snapshot_path)
            torch.cuda.memory._record_memory_history(enabled=None)
            print(f"[child] mem snapshot -> {snapshot_path}", file=sys.stderr)

        peak = torch.cuda.max_memory_allocated()
        if layer_probe is not None:
            peak = max(peak, int(layer_probe["tracked_peak"]))
            full_gb = layer_probe["full_attention_bytes"] / (1024**3)
            linear_gb = layer_probe["linear_attention_bytes"] / (1024**3)
            print(
                f"[layer-workspace] full_attention={full_gb:.3f}GiB "
                f"calls={layer_probe['full_attention_calls']} "
                f"linear_attention={linear_gb:.3f}GiB "
                f"calls={layer_probe['linear_attention_calls']}",
                file=sys.stderr,
                flush=True,
            )
            result["layer_workspace_full_gb"] = full_gb
            result["layer_workspace_linear_gb"] = linear_gb
            result["layer_workspace_full_calls"] = layer_probe["full_attention_calls"]
            result["layer_workspace_linear_calls"] = layer_probe["linear_attention_calls"]
        rss_peak = rss.stop()

        offloaded = None
        prefetched = None
        if dss_manager is not None:
            offloaded = dss_manager.stats.offloaded_tensors
        elif hf_manager is not None:
            offloaded = hf_manager.stats.offloaded_tensors
            prefetched = hf_manager.stats.prefetched_tensors
        elif native:
            offload_eng = getattr(ds_ckpt, "_cpu_offload_engine", None)
            if offload_eng is not None:
                offloaded = offload_eng.stats.offloaded_tensors
                prefetched = offload_eng.stats.prefetched_tensors

        ms_global = (elapsed / iters) * 1e3
        tok_s_gpu = tokens_per_rank / (elapsed / iters)
        alloc_after = _dump_alloc_stats("after_timed") if rank0 else {}
        if args.mem_census and rank0 and ds_engine is not None:
            _dump_mem_census("after_timed", ds_engine)
        if args.comms_logger:
            import deepspeed.comm as dist
            comms = dist.log_summary(return_dict=True)
            if rank0 and comms:
                summary = comms.get("summary") or {}
                compact = {}
                for op, sizes in sorted(summary.items()):
                    count = 0
                    lat = 0.0
                    bytes_sum = 0.0
                    if isinstance(sizes, dict):
                        for sz, rec in sizes.items():
                            n = int((rec or {}).get("count", 0) or 0)
                            count += n
                            lat += float((rec or {}).get("total_latency_ms", 0) or 0)
                            bytes_sum += n * float(sz)
                    compact[op] = {"count": count, "total_lat_ms": lat, "bytes": bytes_sum}
                print(
                    "[comms] timed-loop " + ",".join(f"{k}:count={v['count']}:lat_ms={v['total_lat_ms']:.1f}"
                                                     for k, v in compact.items()),
                    file=sys.stderr,
                    flush=True,
                )
                result["comms_summary"] = compact
        result.update({
            "ok": True,
            "ms_step": ms_global,
            "ms_global": ms_global,
            "tok_s_gpu": tok_s_gpu,
            "tok_s": tok_s_gpu * world_size,
            "cuda_peak_delta_gb": (peak - base) / (1024**3),
            "cuda_max_allocated_gb": peak / (1024**3),
            "cpu_rss_delta_gb": max(0, rss_peak - rss_base) / (1024**3),
            "loss": float(loss_val.detach().item()),
            "offloaded_tensors": offloaded,
            "prefetched_tensors": prefetched,
            "num_alloc_retries": alloc_after.get("num_alloc_retries"),
            "num_ooms": alloc_after.get("num_ooms"),
            "reserved_peak_gb": alloc_after.get("reserved_peak_gb"),
            "frag_gb": alloc_after.get("frag_gb"),
        })
    except torch.cuda.OutOfMemoryError:
        result.update({"ok": False, "error": "OOM"})
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        result.update({"ok": False, "error": f"{type(e).__name__}: {e}"})

    if rank0:
        print(RESULT_PREFIX + json.dumps(result))
    return 0


def _hostfile_hosts(path):
    hosts = []
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            hosts.append(line.split()[0])
    return hosts


def _run_torchrun(cmd, nnodes, hosts):
    if nnodes <= 1:
        return subprocess.run(cmd, capture_output=True, text=True)
    if len(hosts) < nnodes:
        raise RuntimeError(f"hostfile has {len(hosts)} hosts, need nnodes={nnodes}")
    hosts = hosts[:nnodes]
    procs = []
    for i, host in enumerate(hosts):
        if i == 0:
            procs.append(subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
            continue
        env_cmd = [
            "env",
            f"PYTHONPATH={os.environ.get('PYTHONPATH', '')}",
            f"HF_HOME={os.environ.get('HF_HOME', '/data-fast/huggingface')}",
            "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
            "PYTORCH_ALLOC_CONF=expandable_segments:True",
        ]
        remote = ["ssh", "-o", "BatchMode=yes", "-n", host] + env_cmd + cmd
        procs.append(subprocess.Popen(remote, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
    stdout_parts = []
    stderr_parts = []
    rc = 0
    for proc in procs:
        out, err = proc.communicate()
        stdout_parts.append(out or "")
        stderr_parts.append(err or "")
        rc = rc or proc.returncode

    class _Done:
        pass

    done = _Done()
    done.returncode = rc
    done.stdout = stdout_parts[0]
    # Keep remote logs visible; RESULT_JSON is printed by rank0 on node 0.
    extra = []
    for i, chunk in enumerate(stdout_parts[1:], start=1):
        if chunk.strip():
            extra.append(f"[node{i} stdout]\n{chunk}")
    if extra:
        done.stdout = done.stdout + "\n" + "\n".join(extra)
    done.stderr = "".join(stderr_parts)
    return done


def run_driver(args):
    modes = [m for m in args.modes.split(",") if m]
    seqlens = [int(s) for s in args.seqlens.split(",") if s]
    results = []
    for seqlen in seqlens:
        for mode in modes:
            child = [
                os.path.abspath(__file__), "--child", "--mode", mode, "--seqlen",
                str(seqlen), "--model_name", args.model_name, "--num_layers",
                str(args.num_layers), "--warmup",
                str(args.warmup), "--iters",
                str(args.iters), "--mbs",
                str(args.mbs), "--gas",
                str(args.gas), "--cpu_optimizer", args.cpu_optimizer, "--nproc",
                str(args.nproc)
            ]
            if args.fused_ce:
                child.append("--fused_ce")
            if args.sum_loss:
                child.append("--sum_loss")
            if args.zero2:
                child.append("--zero2")
            if args.zero2_offload:
                child.append("--zero2_offload")
            if args.zero3:
                child.append("--zero3")
            if args.zero3_offload:
                child.append("--zero3_offload")
            if args.mem_snapshot:
                child.extend(["--mem_snapshot", args.mem_snapshot])
            if args.mem_snapshot_early:
                child.append("--mem_snapshot_early")
            if args.offload_reclaim_probe:
                child.append("--offload_reclaim_probe")
            if args.mem_census:
                child.append("--mem_census")
            if args.gc_each_step:
                child.append("--gc_each_step")
            if args.gc_reentrant:
                child.append("--gc_reentrant")
            if args.comms_logger:
                child.append("--comms_logger")
            child.extend(["--autoep_size", str(int(args.autoep_size))])
            child.extend(["--autoep_preset", args.autoep_preset])
            child.append("--opt_pin_memory" if args.opt_pin_memory else "--no-opt_pin_memory")
            child.append("--act_pin_memory" if args.act_pin_memory else "--no-act_pin_memory")
            child.append("--act_use_streams" if args.act_use_streams else "--no-act_use_streams")
            child.extend(["--tiled_mlp_shards", str(int(args.tiled_mlp_shards))])
            child.extend(["--tiled_rematerialize_shards", str(int(args.tiled_rematerialize_shards))])
            child.extend(["--seq_chain_budget", str(int(args.seq_chain_budget))])
            if args.packed_seq:
                child.append("--packed_seq")
            if args.pack_lens:
                child.extend(["--pack_lens", args.pack_lens])
            child.extend(["--packed_tiled_remat_shards", str(int(args.packed_tiled_remat_shards))])
            child.extend(["--packed_tiled_remat_budget", str(int(args.packed_tiled_remat_budget))])
            if args.layer_workspace_probe:
                child.append("--layer_workspace_probe")
            if args.tiled_fwd:
                child.append("--tiled_fwd")
            nproc = int(args.nproc)
            nnodes = int(args.nnodes)
            if nproc > 1:
                port = 29556 + (seqlen + args.mbs * 17 + args.gas * 31 + hash(mode)) % 1000
                cmd = [
                    sys.executable,
                    "-m",
                    "torch.distributed.run",
                    f"--nproc_per_node={nproc}",
                    f"--master_port={port}",
                ]
                hosts = []
                if nnodes > 1:
                    hostfile = args.hostfile or "/job/hostfile"
                    hosts = _hostfile_hosts(hostfile)
                    master = args.master_addr or hosts[0]
                    cmd.extend([
                        f"--nnodes={nnodes}",
                        "--rdzv_backend=c10d",
                        f"--rdzv_endpoint={master}:{port}",
                        f"--rdzv_id=bench{port}",
                    ])
                else:
                    cmd.append("--standalone")
                cmd = cmd + child
            else:
                cmd = [sys.executable] + child
                hosts = []
            tag = mode
            if args.sum_loss:
                tag += "+sum"
            if args.fused_ce:
                tag += "+lce"
            if int(args.tiled_rematerialize_shards) > 1:
                tag += f"+remat{int(args.tiled_rematerialize_shards)}"
                if args.tiled_fwd:
                    tag += "+tiledfwd"
            if args.zero2_offload:
                tag += "+z2cpu"
            elif args.zero3:
                tag += "+z3gpu"
            elif args.zero2:
                tag += "+z2gpu"
            if int(args.seq_chain_budget) > 0:
                tag += f"+chain{int(args.seq_chain_budget)}"
            if int(args.autoep_size) > 0:
                tag += f"+autoep{int(args.autoep_size)}"
            print(
                f"\n>>> running mode={tag} model={args.model_name} seqlen={seqlen} "
                f"mbs={args.mbs} gas={args.gas} nproc={nproc} nnodes={nnodes} "
                f"tiled_mlp={args.tiled_mlp_shards} tiled_remat={args.tiled_rematerialize_shards} "
                f"seq_chain={args.seq_chain_budget} "
                f"packed={int(bool(args.packed_seq))} pack_lens={args.pack_lens or '-'} "
                f"packed_budget={args.packed_tiled_remat_budget} "
                f"tiled_fwd={int(bool(args.tiled_fwd))} "
                f"autoep={args.autoep_size}",
                flush=True)
            if nproc > 1:
                proc = _run_torchrun(cmd, nnodes, hosts)
            else:
                proc = subprocess.run(cmd, capture_output=True, text=True)
            sys.stderr.write(proc.stderr)
            parsed = None
            for line in proc.stdout.splitlines():
                if line.startswith(RESULT_PREFIX):
                    parsed = json.loads(line[len(RESULT_PREFIX):])
                else:
                    print(line, flush=True)
            if parsed is None:
                chain_tag = f"+chain{int(args.seq_chain_budget)}" if int(args.seq_chain_budget) > 0 else ""
                parsed = {
                    "mode": f"{mode}{chain_tag}",
                    "seqlen": seqlen,
                    "mbs": args.mbs,
                    "gas": args.gas,
                    "model_name": args.model_name,
                    "ok": False,
                    "error": f"no result (rc={proc.returncode})",
                }
            results.append(parsed)

    _render_table(results, args.model_name)


def _render_table(results, model_name):
    header = (f"{'seq':>7}  {'mode':<18}  {'mbs':>3}  {'gas':>3}  {'nproc':>5}  {'lce':>3}  {'z2':>6}  "
              f"{'ms/global':>10}  {'tok/s':>8}  {'tok/s/gpu':>9}  {'CPU RSS dGB':>12}  "
              f"{'CUDA peak dGB':>13}  {'CUDA max GB':>12}")
    print("\n" + "=" * len(header))
    print(f"{model_name} activation-offload benchmark")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in results:
        if r.get("ok"):
            if r.get("sum_loss"):
                lce = "sum"
            else:
                lce = "on" if r.get("fused_ce") else "off"
            if r.get("zero2_offload"):
                z2 = "z2cpu"
            elif r.get("zero3") or r.get("zero_stage") == 3:
                z2 = "z3gpu"
            elif r.get("zero2"):
                z2 = "z2gpu"
            else:
                z2 = "off"
            max_gb = r.get("cuda_max_allocated_gb")
            max_s = f"{max_gb:12.2f}" if max_gb is not None else f"{'n/a':>12}"
            ms = r.get("ms_global", r.get("ms_step"))
            tok = r.get("tok_s")
            tok_gpu = r.get("tok_s_gpu", tok)
            nproc = r.get("nproc", 1)
            print(f"{r['seqlen']:>7}  {r['mode']:<18}  {r.get('mbs', 1):>3}  {r.get('gas', 1):>3}  "
                  f"{nproc:>5}  {lce:>3}  {z2:>6}  {ms:>10.1f}  {tok:>8.1f}  {tok_gpu:>9.1f}  "
                  f"{r['cpu_rss_delta_gb']:>12.2f}  {r['cuda_peak_delta_gb']:>13.2f}  {max_s}")
        else:
            print(f"{r['seqlen']:>7}  {r['mode']:<18}  {r.get('mbs', 1):>3}  {r.get('gas', 1):>3}  "
                  f"{r.get('error', 'FAIL')}")
    print("=" * len(header))
    print("\nRAW: " + json.dumps(results))


def main():
    args = parse_args()
    if args.child:
        return run_child(args)
    return run_driver(args)


if __name__ == "__main__":
    sys.exit(main())
