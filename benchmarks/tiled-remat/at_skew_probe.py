"""Find which rank diverges before the packed-sequence collective hang, and where.

Background. ArcticTraining packed-seq runs hang on a 1-element ALLREDUCE that never
completes, timing out at the 600s process-group deadline, with Qwen3-4B/gc passing at
max_length 389,120 and hanging at 393,216. It is not capacity (the two affected arms hang at
GPU occupancies 21 GiB apart, one with 25 GiB of HBM and 1.8 TB of host free), not the
tiling path (`gc` does no tiling), and not the packer's remainder-dropping (the hang persists
with `always_max_length: false`). A 1-element allreduce that hangs after dozens of completed
collectives means **control-flow skew**: some rank issues a different number of collectives
than its peers, so the odd one waits forever.

Ablation cannot go further -- the remaining suspect, `tiled_mlp_compute`, is also a memory
optimization, so disabling it OOMs before the run reaches the skew. This measures the
divergence directly instead.

Two probes, both env-gated on AT_SKEW_PROBE=1:

1. Per-rank dataloader length and training_horizon. `training_horizon` is
   `epochs * len(train_dataloader)` and `early_stop` fires on
   `global_step >= training_horizon`, so unequal lengths would make ranks stop at different
   steps. An all_gather reports MISMATCH/uniform directly, so this is one line to read
   rather than eight to compare. (Expected to come back uniform here, since
   `exit_iteration: 3` should stop every rank at the same step -- worth ruling out cheaply.)

2. An ordered per-rank log of every torch.distributed collective: index, op, tensor numel,
   and the first in-AT/DeepSpeed caller frame. When the job is killed, the LAST line of each
   rank file is the collective that rank was sitting in. Diffing the files finds the first
   index where ranks disagree -- which names the divergent call site. Each line is flushed,
   because the process dies by SIGKILL and anything buffered is lost.

Volume is small: the hang arrives around collective 43, so this writes tens of lines per
rank, not thousands.
"""

import os
import sys
import traceback

_LOGDIR = os.environ.get(
    "AT_SKEW_PROBE_DIR",
    "autorun/_skew3",
)

_state = {"n": 0, "fh": None, "rank": -1}
_PATCHED = []


def _interesting_frame():
    """The innermost caller that is AT or DeepSpeed, not torch internals or this probe."""
    for fr in reversed(traceback.extract_stack()[:-2]):
        f = fr.filename
        if "at_skew_probe" in f:
            continue
        if "/arctic_training/" in f or "/deepspeed/" in f:
            short = f.split("/arctic_training/")[-1].split("/deepspeed/")[-1]
            return f"{short}:{fr.lineno}:{fr.name}"
    return "?"


def _log(op, args):
    st = _state
    fh = st["fh"]
    if fh is None:
        return
    st["n"] += 1
    numel = ""
    try:
        import torch

        for a in args:
            if torch.is_tensor(a):
                numel = a.numel()
                break
            if isinstance(a, (list, tuple)) and a and torch.is_tensor(a[0]):
                numel = f"list[{len(a)}]x{a[0].numel()}"
                break
    except Exception:
        pass
    fh.write(f"{st['n']}\t{op}\tnumel={numel}\t{_interesting_frame()}\n")
    fh.flush()


def install(trainer):
    """Called from Trainer.train() under AT_SKEW_PROBE=1."""
    import torch
    import torch.distributed as dist

    rank = getattr(trainer, "global_rank", 0)
    _state["rank"] = rank
    os.makedirs(_LOGDIR, exist_ok=True)
    _state["fh"] = open(os.path.join(_LOGDIR, f"rank{rank}.log"), "w", buffering=1)

    # --- probe 1: does every rank agree on how long training is? ---
    try:
        n = len(trainer.train_dataloader) if trainer.train_dataloader is not None else -1
    except Exception as e:  # pragma: no cover - diagnostics must not add failures
        n = -1
        print(f"[skew] rank={rank} len(train_dataloader) raised {e!r}", file=sys.stderr, flush=True)
    try:
        horizon = trainer.training_horizon
    except Exception:
        horizon = -1
    cfg = trainer.config
    print(
        f"[skew] rank={rank} len_dataloader={n} training_horizon={horizon} "
        f"epochs={getattr(cfg, 'epochs', None)} gas={getattr(cfg, 'gradient_accumulation_steps', None)} "
        f"exit_iteration={getattr(cfg, 'exit_iteration', None)} "
        f"min_iterations={getattr(cfg, 'min_iterations', None)} "
        f"max_length={getattr(getattr(cfg, 'data', None), 'max_length', None)}",
        file=sys.stderr,
        flush=True,
    )
    if dist.is_available() and dist.is_initialized():
        try:
            dev = torch.cuda.current_device()
            t = torch.tensor([int(n), int(horizon)], device=dev, dtype=torch.long)
            gathered = [torch.zeros_like(t) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered, t)
            if rank == 0:
                lens = [int(x[0].item()) for x in gathered]
                hors = [int(x[1].item()) for x in gathered]
                print(
                    f"[skew] ACROSS RANKS len_dataloader={lens} "
                    f"{'MISMATCH' if len(set(lens)) > 1 else 'uniform'} | "
                    f"training_horizon={hors} "
                    f"{'MISMATCH' if len(set(hors)) > 1 else 'uniform'}",
                    file=sys.stderr,
                    flush=True,
                )
        except Exception as e:
            print(f"[skew] rank={rank} all_gather probe failed: {e!r}", file=sys.stderr, flush=True)

    # --- probe 2: order every collective this rank issues ---
    for name in (
        "all_reduce",
        "all_gather",
        "all_gather_into_tensor",
        "reduce_scatter",
        "reduce_scatter_tensor",
        "broadcast",
        "barrier",
        "reduce",
        "gather",
        "scatter",
        "all_to_all",
        "all_to_all_single",
    ):
        fn = getattr(dist, name, None)
        if fn is None or not callable(fn):
            continue

        def make(op, orig):
            def wrapped(*a, **k):
                _log(op, a)
                return orig(*a, **k)

            return wrapped

        setattr(dist, name, make(name, fn))
        _PATCHED.append(name)

    print(
        f"[skew] rank={rank} collective logging active -> {_LOGDIR}/rank{rank}.log "
        f"({len(_PATCHED)} ops wrapped)",
        file=sys.stderr,
        flush=True,
    )
