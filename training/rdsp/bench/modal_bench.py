# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""rdsp vs Megatron: pipeline-parallel training throughput on Modal H100s.

From the repo root:

    modal run bench/modal_bench.py                        # main grid, 1 repeat
    modal run bench/modal_bench.py --reps 3 --only headline
    modal run bench/modal_bench.py --only samebox         # see main() for all modes

Results are written to bench/results/ locally, one JSON file per run.

One NGC image serves both frameworks (modal_smoke.py checks it). Methodology:
- GPUs are requested as `H100!` (Modal otherwise may substitute H200s) and
  every container asserts the GPU name before doing any work. Identical work
  still varied by up to ~20% between machines of the same chip, so the
  comparisons that matter run both frameworks back to back in ONE container.
- Shapes run one after another: the workspace allows 10 concurrent GPUs.
- DeepSpeed ZeRO-0 bf16 sums microbatch gradients in bf16, Megatron in fp32 by
  default; the `grad_bf16` runs match Megatron to rdsp's precision.

Grid systems
  M-plain     Megatron-LM pretrain_gpt.py, optional fusions off
  M-defaults  pretrain_gpt.py with its standard fusions on
  R-local     rdsp (stage-local dispatch)
  *-fused     rdsp with Liger kernels matching Megatron's default fusions
Results recorded before 2026-09-26 also contain R-os / R-rdt cells: rdsp's
former driver-dispatched design (object store / Ray Direct Transport), since
removed from the package.
Model: Qwen3-0.6B shape (28 layers, untied), bf16, Adam lr 1e-5, no clipping,
no activation recomputation; 2048 tokens per microbatch (b=4 x 512 or b=1 x 2048).
"""

import itertools
import json
import os
import random
import sys
import time
from pathlib import Path

import modal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modal_smoke import image

app = modal.App("rdsp-vs-megatron", image=image)
cache = modal.Volume.from_name("rdsp-bench-cache", create_if_missing=True)
hf_cache = modal.Volume.from_name("rdsp-hf-cache", create_if_missing=True)
KW = dict(
    timeout=4 * 3600,
    cpu=16,
    memory=65536,
    volumes={"/cache": cache, "/root/.cache/huggingface": hf_cache},
)
BENCH = "/root/rdsp/bench"
HF = "/cache/qwen3-0.6b-untied"
RESULTS = Path(__file__).resolve().parent / "results"
WARMUP, STEPS = 10, 40


# --- helpers that run inside the containers -----------------------------------


def _sh(cmd):
    import subprocess

    return subprocess.run(cmd, shell=True, capture_output=True, text=True)


def _require_h100():
    """Fail before spending if Modal handed out anything but H100s."""
    gpus = _sh("nvidia-smi --query-gpu=name --format=csv,noheader").stdout
    names = [g for g in gpus.split("\n") if g.strip()]
    assert names and all("H100" in g for g in names), f"not an H100 machine: {gpus}"
    return gpus


def _install_rdsp():
    import subprocess

    subprocess.run(["pip", "install", "-e", "/root/rdsp"], check=True, capture_output=True)


def _load(path):
    with open(path) as f:
        return json.load(f)


def _tail(r, n=3000):
    return (r.stdout + r.stderr)[-n:]


def _megatron_hf(nproc, data, steps, out, **flags):
    """megatron_hf_loop.py on `nproc` GPUs; `flags` become --key 'value'."""
    extra = " ".join(f"--{k} '{v}'" for k, v in flags.items())
    return _sh(
        f"cd {BENCH} && CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --standalone "
        f"--nproc_per_node {nproc} megatron_hf_loop.py --hf {HF} --data {data} "
        f"--steps {steps} --out {out} {extra}"
    )


def _rdsp(pp, m, data, steps, out, **flags):
    """rdsp_bench.py with fused kernels, no warmup cut (callers drop warmup
    steps when summarizing)."""
    extra = " ".join(f"--{k} '{v}'" for k, v in flags.items())
    return _sh(
        f"cd {BENCH} && python rdsp_bench.py --hf {HF} --pp {pp} --m {m} --seq 512 "
        f"--steps {steps} --warmup 0 --data {data} --out {out} --fused 1 {extra}"
    )


def _megatron_checks(logs):
    """The per-rank layer lists and main_grad dtype that megatron_hf_loop.py
    prints: evidence of the layer split and gradient precision each run used."""
    return sorted(
        {
            line.strip()
            for line in logs.splitlines()
            if "main_grad dtype" in line or ": layers [" in line
        }
    )


def _pick(path, keys):
    return {k: v for k, v in _load(path).items() if k in keys}


# --- the grid: one cell per pretrain_gpt.py / rdsp_bench.py run ---------------


def _env_report():
    def run(cmd):
        return _sh(cmd).stdout.strip()

    return {
        "gpus": run("nvidia-smi --query-gpu=name,driver_version --format=csv,noheader"),
        "topo": run("nvidia-smi topo -m"),
        "versions": run(
            'python -c "import torch,deepspeed,ray,transformer_engine as te,'
            "megatron.core as mc;print(torch.__version__,deepspeed.__version__,"
            'ray.__version__,te.__version__,mc.__version__)"'
        ),
    }


def _nvml_peak(stop, peaks):
    import pynvml

    pynvml.nvmlInit()
    hs = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
    while not stop.is_set():
        for i, h in enumerate(hs):
            peaks[i] = max(peaks.get(i, 0), pynvml.nvmlDeviceGetMemoryInfo(h).used)
        time.sleep(0.05)


def _megatron_cell(c):
    import re

    iters = WARMUP + STEPS
    r = _sh(
        f"cd {BENCH} && PP={c['pp']} M={c['m']} SEQ={c['seq']} ITERS={iters} "
        f"VARIANT={c.get('variant', 'plain')} BALANCED={int(bool(c.get('balanced')))} "
        f"LAYOUT='{c.get('layout', '')}' "
        f"MEGATRON=/opt/Megatron-LM bash megatron_pretrain.sh"
    )
    out = r.stdout + r.stderr
    times = [
        float(x) / 1000 for x in re.findall(r"elapsed time per iteration \(ms\): ([\d.]+)", out)
    ]
    losses = [float(x) for x in re.findall(r"lm loss: ([\d.E+-]+)", out)]
    # keep the lines that explain a failure (rank tracebacks), not just the
    # launcher's generic epilogue
    errs = [
        line
        for line in out.splitlines()
        if re.search(r"Error|error|Assert|assert|Traceback|out of memory", line)
    ]
    return (
        r.returncode,
        {"step_s": times[WARMUP:], "loss": losses},
        "\n".join(errs[:60]) + "\n...\n" + out[-1500:],
    )


def _rdsp_cell(c):
    b = 4 if c["seq"] == 512 else 1
    out_path = f"/tmp/rdsp_{os.getpid()}_{random.randrange(10**9)}.json"
    r = _sh(
        f"cd {BENCH} && python rdsp_bench.py --hf {HF} "
        f"--pp {c['pp']} --m {c['m']} --seq {c['seq']} "
        f"--steps {c.get('warmup', WARMUP) + c.get('steps', STEPS)} "
        f"--warmup {c.get('warmup', WARMUP)} "
        f"--data /cache/batches_s{c['seq']}_b{b}_m{c['m']}.pt "
        f"--out {out_path} "
        f"--cuts '{c.get('cuts', '')}' --last_stage_gpus {c.get('last_stage_gpus', 1)} "
        f"--fused {int(bool(c.get('fused')))} "
        f"--timed {int(bool(c.get('timed')))} --timeline {int(bool(c.get('timeline')))}"
    )
    res = _load(out_path) if r.returncode == 0 and os.path.exists(out_path) else {}
    return r.returncode, res, _tail(r)


def _summarize_profile(c, res):
    """Compact profile summary computed in the container; the raw Ray
    timeline and per-stage CUDA timings go to the cache volume."""
    from collections import defaultdict

    prof = res.pop("profile", None)
    if not prof:
        return None
    os.makedirs("/cache/profile", exist_ok=True)
    raw = f"/cache/profile/rdsp_pp{c['pp']}_m{c['m']}.json"
    with open(raw, "w") as f:
        json.dump(prof, f)
    warm = c.get("warmup", WARMUP)
    windows = prof.get("step_windows", [])[warm:]
    n = max(len(windows), 1)
    out = {"raw": raw, "measured_steps": len(windows)}

    # per stage: GPU busy time per step from CUDA events
    stage_pid, gpu = {}, {}
    for fname, recs in prof.get("stage_ops", {}).items():
        recs = [r for r in recs if r["step"] >= warm]
        tot = defaultdict(float)
        for r in recs:
            for kind, _, ms in r["ops"]:
                tot[kind] += ms
        k = max(len(recs), 1)
        gpu[fname] = {kind: round(v / k, 2) for kind, v in tot.items()}
        if recs:
            stage_pid[recs[0]["pid"]] = fname
    out["gpu_ms_per_step_by_stage"] = gpu

    # Ray timeline within the measured steps: time per (process, event name)
    lo = windows[0][0] * 1e6 if windows else 0
    hi = windows[-1][1] * 1e6 if windows else float("inf")
    by = defaultdict(lambda: [0, 0.0])
    for e in prof.get("timeline", []):
        if e.get("ph") != "X" or not (lo <= e.get("ts", 0) <= hi):
            continue
        key = (e.get("pid"), e.get("name"), e.get("cat"))
        by[key][0] += 1
        by[key][1] += e.get("dur", 0) / 1000
    rows = sorted(
        ((v[1] / n, v[0] / n, str(k[0]), str(k[1]), str(k[2])) for k, v in by.items()), reverse=True
    )
    out["timeline_top"] = [
        {
            "ms_per_step": round(ms, 2),
            "count_per_step": round(cnt, 1),
            "pid": pid,
            "name": name,
            "cat": cat,
        }
        for ms, cnt, pid, name, cat in rows[:40]
    ]
    out["stage_pids"] = {str(k): v for k, v in stage_pid.items()}
    cache.commit()
    return out


def run_cells(cells):
    """Run grid cells in this container, in random order (unless profiling),
    sampling NVML peak memory alongside each."""
    import threading

    _install_rdsp()
    env = _env_report()
    assert "H200" not in env["gpus"] and "H100" in env["gpus"], (
        f"not an H100 machine: {env['gpus']}"
    )
    if not any(c.get("timeline") for c in cells):
        random.shuffle(cells)
    results = []
    for c in cells:
        stop, peaks = threading.Event(), {}
        th = threading.Thread(target=_nvml_peak, args=(stop, peaks), daemon=True)
        th.start()
        t0 = time.time()
        fn = _megatron_cell if c["system"].startswith("M-") else _rdsp_cell
        rc, res, tail = fn(c)
        prof_summary = _summarize_profile(c, res) if res else None
        stop.set()
        th.join()
        results.append(
            {
                "cell": c,
                "rc": rc,
                "wall_s": time.time() - t0,
                "nvml_peak_bytes": peaks,
                "result": res,
                "profile_summary": prof_summary,
                "tail": tail if rc else "",
                "env": env,
            }
        )
        steps = res.get("step_s") or []
        med = f"{sorted(steps)[len(steps) // 2]:.3f}s" if steps else "n/a"
        print(f"{c} rc={rc} median_step={med}", flush=True)
        if rc or not steps:
            print(f"--- failure tail ---\n{tail}", flush=True)
    return results


@app.function(gpu="H100!:1", **KW)
def h100x1(cells):
    return run_cells(cells)


@app.function(gpu="H100!:2", **KW)
def h100x2(cells):
    return run_cells(cells)


@app.function(gpu="H100!:4", **KW)
def h100x4(cells):
    return run_cells(cells)


@app.function(gpu="H100!:8", **KW)
def h100x8(cells):
    return run_cells(cells)  # also the 5-GPU capability cells (uses 5 of 8)


RUN = {1: h100x1, 2: h100x2, 4: h100x4, 8: h100x8}


# --- matched runs: same pretrained weights, same real text, same order --------

TEXT = "/cache/text_s512_b4_m8.pt"
LOSS_STEPS = 300
TEXT16 = "/cache/text_s512_b4_m16.pt"  # the same tokens regrouped: 150 steps x 16 microbatches
TEXT_B16 = "/cache/text_s512_b16_m4.pt"  # the same tokens: 4 microbatches x 16 rows x 512
TEXT_B16_M16 = "/cache/text_s512_b16_m16.pt"  # 37 steps x 16 microbatches x 16 rows x 512
PP8_LAYOUT = "Et*4|t*4|t*4|t*4|t*3|t*3|t*3|t*3,L"  # = rdsp's uniform split at 8 stages


@app.function(**KW)
def prepare_text():
    """WikiText-103 (real text), tokenized with Qwen3's tokenizer, packed into
    [steps, microbatches, rows, 513]; both frameworks read it identically."""
    import subprocess

    import torch

    if os.path.exists(TEXT):
        return "cached"
    subprocess.run(["pip", "install", "-q", "datasets"], check=True)
    import datasets
    import transformers

    tok = transformers.AutoTokenizer.from_pretrained(HF)
    need = LOSS_STEPS * 8 * 4 * 513
    ds = datasets.load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    ids, i = [], 0
    while len(ids) < need:
        chunk = "".join(ds[i : i + 2000]["text"])
        ids.extend(tok(chunk)["input_ids"])
        i += 2000
    t = torch.tensor(ids[:need], dtype=torch.int64).view(LOSS_STEPS, 8, 4, 513)
    torch.save(t, TEXT)
    cache.commit()
    return f"{need} tokens"


@app.function(**KW)
def prepare_text16():
    import torch

    if not os.path.exists(TEXT16):
        x = torch.load(TEXT)
        torch.save(x.reshape(x.shape[0] // 2, 16, *x.shape[2:]).clone(), TEXT16)
        cache.commit()
    return "ok"


def _ensure_b16_m16():
    import torch

    if not os.path.exists(TEXT_B16_M16):
        x = torch.load(TEXT).reshape(-1, 513)
        torch.save(x[: 37 * 256].reshape(37, 16, 16, 513).clone(), TEXT_B16_M16)


def _loss_compare(
    pp, text=TEXT, steps=LOSS_STEPS, layout="", systems=("megatron", "rdsp"), grad_bf16=0
):
    """HF reference loss on step 0, then Megatron-Core and rdsp from the same
    pretrained weights on the same batches, fusions on for both."""
    import subprocess

    import torch

    gpus = _require_h100()
    _install_rdsp()
    m = torch.load(text).shape[1]
    out = {"pp": pp, "m": m, "text": text, "layout": layout, "gpu": gpus.split("\n")[0]}
    # 1. reference: the plain HF model's loss on step 0's batch (no training)
    ref = subprocess.run(
        [
            "python",
            "-c",
            f"""
import torch, torch.nn.functional as F, transformers, json
m = transformers.AutoModelForCausalLM.from_pretrained(
    '{HF}', dtype=torch.bfloat16, attn_implementation='sdpa').cuda().eval()
X = torch.load('{text}')
with torch.no_grad():
    ls = [F.cross_entropy(m(X[0, k][:, :-1].cuda()).logits.float().flatten(0, 1),
                          X[0, k][:, 1:].cuda().flatten()).item() for k in range(X.shape[1])]
print(json.dumps(sum(ls) / len(ls)))
""",
        ],
        capture_output=True,
        text=True,
    )
    out["grad_bf16"] = grad_bf16
    out["hf_step0_loss"] = (
        json.loads(ref.stdout.strip().splitlines()[-1])
        if ref.returncode == 0
        else ref.stderr[-2000:]
    )
    # 2. Megatron-Core from the same HF weights, its standard fusions on
    r = _megatron_hf(pp, text, steps, "/tmp/meg_loss.json", layout=layout, grad_bf16=grad_bf16)
    logs = r.stdout + r.stderr
    out["megatron_layers"] = [
        line for line in logs.splitlines() if ": layers [" in line or "main_grad dtype" in line
    ]
    out["megatron"] = (
        _load("/tmp/meg_loss.json")
        if r.returncode == 0
        else {
            "error": "\n".join(
                line
                for line in logs.splitlines()
                if "rank0" in line or "Error" in line or "assert" in line
            )[-4000:]
            + "\n...\n"
            + logs[-2000:]
        }
    )
    if "rdsp" not in systems:
        out["rdsp"] = {"error": "skipped"}
        return out
    # 3. rdsp: same weights, data, optimizer, matching fused kernels
    r = _rdsp(pp, m, text, steps, "/tmp/rdsp_loss.json")
    out["rdsp"] = _load("/tmp/rdsp_loss.json") if r.returncode == 0 else {"error": _tail(r, 4000)}
    out["rdsp"].pop("profile", None)
    out["rdsp"].pop("executed", None)
    return out


def _same_box(pp, layout="", steps=60):
    """Megatron fp32-summing, Megatron bf16-summing and rdsp, back to back in
    ONE container on pinned H100s, same weights/data/kernels/steps."""
    _require_h100()
    _install_rdsp()
    q = (
        "nvidia-smi --query-gpu=index,name,pci.bus_id,clocks.max.sm,power.limit,"
        "temperature.gpu --format=csv,noheader"
    )
    out = {
        "pp": pp,
        "m": 16,
        "steps": steps,
        "layout": layout,
        "gpus": _sh(q).stdout,
        "topo": _sh("nvidia-smi topo -m").stdout,
        "order": ["megatron_fp32", "megatron_bf16", "rdsp"],
    }
    for name, bf16 in (("megatron_fp32", 0), ("megatron_bf16", 1)):
        r = _megatron_hf(pp, TEXT16, steps, f"/tmp/{name}.json", layout=layout, grad_bf16=bf16)
        out[name] = _load(f"/tmp/{name}.json") if r.returncode == 0 else {"error": _tail(r)}
        out[name + "_check"] = _megatron_checks(r.stdout + r.stderr)
    r = _rdsp(pp, 16, TEXT16, steps, "/tmp/rdsp.json")
    out["rdsp"] = _load("/tmp/rdsp.json") if r.returncode == 0 else {"error": _tail(r)}
    out["rdsp"].pop("profile", None)
    out["rdsp"].pop("executed", None)
    out["gpus_after"] = _sh(q).stdout
    return out


@app.function(gpu="H100!:1", **KW)
def gpubound_x1():
    """GPU-bound regime: 8,192-token microbatches. Megatron (native CE, TE CE)
    and rdsp back to back on one pinned H100, bf16 gradient summing on all,
    60 steps each; steps 20-29 profiled (excluded from timing)."""
    import glob
    import shutil

    import torch

    gpus = _require_h100()
    if not os.path.exists(TEXT_B16):
        x = torch.load(TEXT)  # [300, 8, 4, 513] -> consecutive rows regrouped
        torch.save(x.reshape(-1, x.shape[-1]).reshape(150, 4, 16, x.shape[-1]).clone(), TEXT_B16)
    _install_rdsp()
    out = {
        "gpu": gpus.strip(),
        "b": 16,
        "m": 4,
        "seq": 512,
        "steps": 60,
        "order": ["megatron_native_ce", "megatron_te_ce", "rdsp"],
    }
    for name, ce in (("megatron_native_ce", "native"), ("megatron_te_ce", "te")):
        r = _megatron_hf(
            1,
            TEXT_B16,
            60,
            f"/tmp/{name}.json",
            grad_bf16=1,
            ce_impl=ce,
            prof_out=f"/tmp/{name}_prof.json",
        )
        out[name] = (
            dict(_load(f"/tmp/{name}_prof.json"), **_load(f"/tmp/{name}.json"))
            if r.returncode == 0
            else {"error": _tail(r)}
        )
    shutil.rmtree("/tmp/rdsp_bench", ignore_errors=True)
    r = _rdsp(1, 4, TEXT_B16, 60, "/tmp/rdsp.json", b=16, prof=1)
    files = glob.glob("/tmp/rdsp_bench/prof_*.json")
    out["rdsp"] = (
        dict(_load(files[0]), **_pick("/tmp/rdsp.json", ("step_s", "loss", "b", "m")))
        if r.returncode == 0 and files
        else {"error": _tail(r)}
    )
    return out


def _gpubound_pipe(pp, layout=""):
    """Large-microbatch pipeline run: Megatron (TE CE, bf16 summing) and rdsp
    back to back on one pinned-H100 machine; 36 steps, timing only."""
    _require_h100()
    _ensure_b16_m16()
    _install_rdsp()
    out = {
        "pp": pp,
        "b": 16,
        "m": 16,
        "seq": 512,
        "steps": 36,
        "layout": layout,
        "gpus": _sh("nvidia-smi --query-gpu=index,name --format=csv,noheader").stdout,
        "order": ["megatron_te_ce", "rdsp"],
    }
    r = _megatron_hf(
        pp, TEXT_B16_M16, 36, "/tmp/meg.json", grad_bf16=1, ce_impl="te", layout=layout
    )
    out["megatron_te_ce"] = _load("/tmp/meg.json") if r.returncode == 0 else {"error": _tail(r)}
    out["megatron_te_ce_check"] = _megatron_checks(r.stdout + r.stderr)
    r = _rdsp(pp, 16, TEXT_B16_M16, 36, "/tmp/rdsp.json", b=16)
    out["rdsp"] = (
        _pick("/tmp/rdsp.json", ("step_s", "loss", "stage_tags"))
        if r.returncode == 0
        else {"error": _tail(r)}
    )
    return out


HETERO = [  # (name, system, pp, cuts or layout, last-stage GPUs)
    ("megatron_4_balanced", "megatron", 4, "Et*9|t*9|t*9|t,L", 1),
    ("rdsp_4_balanced", "rdsp", 4, "9,18,27", 1),
    ("megatron_5_balanced", "megatron", 5, "Et*7|t*7|t*7|t*6|t,L", 1),
    ("rdsp_5_balanced", "rdsp", 5, "7,14,21,27", 1),
    ("rdsp_4_stages_5_gpus", "rdsp", 4, "8,16,24", 2),  # the head stage on 2 GPUs
]


@app.function(gpu="H100!:8", **KW)
def hetero_x8():
    """Does giving the head stage a second GPU beat the best uniform layouts?
    Large microbatches, one pinned-H100 machine, back to back."""
    _require_h100()
    _ensure_b16_m16()
    _install_rdsp()
    out = {
        "gpus": _sh("nvidia-smi --query-gpu=index,name --format=csv,noheader").stdout,
        "b": 16,
        "m": 16,
        "seq": 512,
        "steps": 36,
        "order": [h[0] for h in HETERO],
        "configs": HETERO,
    }
    for name, system, pp, spec, last in HETERO:
        path = f"/tmp/{name}.json"
        if system == "megatron":
            r = _megatron_hf(pp, TEXT_B16_M16, 36, path, grad_bf16=1, ce_impl="te", layout=spec)
            out[name] = _load(path) if r.returncode == 0 else {"error": _tail(r)}
            out[name + "_check"] = _megatron_checks(r.stdout + r.stderr)
        else:
            r = _rdsp(pp, 16, TEXT_B16_M16, 36, path, b=16, cuts=spec, last_stage_gpus=last)
            out[name] = (
                _pick(path, ("step_s", "loss", "stage_tags", "last_stage_gpus", "b"))
                if r.returncode == 0
                else {"error": _tail(r)}
            )
    return out


FEWMB = [  # (name, data, microbatches, cuts, last-stage GPUs): rdsp only, same machine
    ("m4_5_stages", TEXT_B16, 4, "7,14,21,27", 1),
    ("m4_head_on_2_gpus", TEXT_B16, 4, "7,14,21", 2),
    ("m16_5_stages", TEXT_B16_M16, 16, "7,14,21,27", 1),
    ("m16_head_on_2_gpus", TEXT_B16_M16, 16, "7,14,21", 2),
]


@app.function(gpu="H100!:8", **KW)
def fewmb_x8():
    """Uneven stages vs a plain 5th stage at 4 and 16 microbatches, one pinned
    H100 machine. Prediction (stated before running): 2-GPU head stage wins,
    ~1.14x at 4 microbatches, ~1.05x at 16, minus its gradient all-reduce."""
    import torch

    _require_h100()
    x = torch.load(TEXT).reshape(-1, 513)
    if not os.path.exists(TEXT_B16):
        torch.save(x.reshape(150, 4, 16, 513).clone(), TEXT_B16)
    _ensure_b16_m16()
    _install_rdsp()
    out = {
        "gpus": _sh("nvidia-smi --query-gpu=index,name --format=csv,noheader").stdout,
        "order": [f[0] for f in FEWMB],
        "configs": FEWMB,
    }
    for name, data, m, cuts, last in FEWMB:
        pp = len(cuts.split(",")) + 1
        path = f"/tmp/{name}.json"
        r = _rdsp(pp, m, data, 36, path, b=16, cuts=cuts, last_stage_gpus=last)
        out[name] = (
            _pick(path, ("step_s", "loss", "stage_tags", "last_stage_gpus", "m", "b"))
            if r.returncode == 0
            else {"error": _tail(r)}
        )
    return out


@app.function(gpu="H100!:4", **KW)
def gpubound_x4():
    return _gpubound_pipe(4)


@app.function(gpu="H100!:8", **KW)
def gpubound_x8():
    return _gpubound_pipe(8, PP8_LAYOUT)


@app.function(gpu="H100!:1", **KW)
def profile_x1():
    """Kernel profile of matched-precision Megatron and rdsp, one pinned H100,
    same tool (torch.profiler) and window (steps 20-29) on both."""
    import glob

    gpus = _require_h100()
    _install_rdsp()
    out = {"gpu": gpus.strip()}
    r = _megatron_hf(1, TEXT16, 31, "/tmp/meg.json", grad_bf16=1, prof_out="/tmp/meg_prof.json")
    out["megatron"] = (
        dict(_load("/tmp/meg_prof.json"), step_s=_load("/tmp/meg.json")["step_s"])
        if r.returncode == 0
        else {"error": _tail(r)}
    )
    r = _rdsp(1, 16, TEXT16, 31, "/tmp/rdsp.json", prof=1)
    files = glob.glob("/tmp/rdsp_bench/prof_*.json")
    out["rdsp"] = (
        dict(_load(files[0]), step_s=_load("/tmp/rdsp.json")["step_s"])
        if r.returncode == 0 and files
        else {"error": _tail(r)}
    )
    return out


@app.function(gpu="H100!:1", **KW)
def samebox_x1():
    return _same_box(1, steps=100)  # longer: Megatron's 1-GPU step time drifts over ~70 steps


@app.function(gpu="H100!:4", **KW)
def samebox_x4():
    return _same_box(4)


@app.function(gpu="H100!:8", **KW)
def samebox_x8():
    return _same_box(8, PP8_LAYOUT)


@app.function(gpu="H100!:4", **KW)
def loss_compare():  # 4 stages, 8 microbatches, 300 steps
    return _loss_compare(4)


@app.function(gpu="H100!:1", **KW)
def matched_x1():
    return _loss_compare(1, TEXT16, 150)


@app.function(gpu="H100!:1", **KW)
def megatron_bf16_x1():  # Megatron only, gradients summed in bf16 like rdsp
    return _loss_compare(1, TEXT16, 150, systems=("megatron",), grad_bf16=1)


@app.function(gpu="H100!:4", **KW)
def megatron_bf16_x4():
    return _loss_compare(4, TEXT16, 150, systems=("megatron",), grad_bf16=1)


@app.function(gpu="H100!:8", **KW)
def megatron_bf16_x8():
    return _loss_compare(8, TEXT16, 150, PP8_LAYOUT, systems=("megatron",), grad_bf16=1)


@app.function(gpu="H100!:4", **KW)
def matched_x4():
    return _loss_compare(4, TEXT16, 150)


@app.function(gpu="H100!:8", **KW)
def matched_x8():
    return _loss_compare(8, TEXT16, 150, PP8_LAYOUT)


@app.function(**KW)
def prepare():
    """Untied Qwen3-0.6B checkpoint and the random-token grid files, once."""
    import torch
    import transformers

    if not os.path.exists(f"{HF}/config.json"):
        m = transformers.AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-0.6B", dtype=torch.bfloat16
        )
        m.lm_head.weight = torch.nn.Parameter(m.model.embed_tokens.weight.detach().clone())
        m.config.tie_word_embeddings = False
        m.save_pretrained(HF)
        transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B").save_pretrained(HF)
    for s, m_ in itertools.product((512, 2048), (4, 8, 16)):
        b = 4 if s == 512 else 1
        path = f"/cache/batches_s{s}_b{b}_m{m_}.pt"
        if not os.path.exists(path):
            g = torch.Generator().manual_seed(0)
            torch.save(torch.randint(0, 151936, (8, m_, b, s + 1), generator=g), path)
    cache.commit()
    hf_cache.commit()


def grid(only: str):
    """Cells for a grid run; the cell dicts are recorded with each result."""
    if only == "context":
        return [
            # Megatron as people run it (fusions on): full grid plus its own
            # PP=1 baselines for pipeline efficiency
            *[
                {"system": "M-defaults", "variant": "defaults", "pp": 1, "m": 8, "seq": seq}
                for seq in (512, 2048)
            ],
            *[
                {"system": "M-defaults", "variant": "defaults", "pp": pp, "m": m, "seq": seq}
                for pp in (2, 4)
                for m in (4, 8, 16)
                for seq in (512, 2048)
            ],
            # balanced split: move layers off the head-heavy last stage
            {
                "system": "R-local",
                "pp": 4,
                "m": 8,
                "seq": 512,
                "cuts": "9,18,27",
                "tag": "balanced",
            },
            {"system": "M-plain", "pp": 4, "m": 8, "seq": 512, "balanced": True, "tag": "balanced"},
            # a second GPU for the last stage (5 GPUs): not expressible in Megatron
            {
                "system": "R-local",
                "pp": 4,
                "m": 8,
                "seq": 512,
                "last_stage_gpus": 2,
                "tag": "capability",
                "shape": 8,
            },
        ]
    if only == "local":
        return [
            {"system": "R-local-fused", "fused": True, "pp": pp, "m": 16, "seq": 512, "shape": 8}
            for pp in (8, 4)
        ]
    if only == "profile":
        # one 8-GPU container: rdsp at 8 stages, then 4 for contrast
        base = {
            "system": "R-local-fused",
            "fused": True,
            "m": 16,
            "seq": 512,
            "timed": True,
            "timeline": True,
            "warmup": 5,
            "steps": 10,
            "shape": 8,
        }
        return [dict(base, pp=8), dict(base, pp=4)]
    if only == "pp8-megatron":
        return [c for c in grid("pp8") if c["system"] == "M-defaults"]
    if only == "pp8":
        # 8 stages; 28 layers split 4-4-4-4-3-3-3-3 on both sides (rdsp's uniform split)
        return [
            {
                "system": "M-defaults",
                "variant": "defaults",
                "pp": 8,
                "m": 16,
                "seq": 512,
                "layout": PP8_LAYOUT,
            },
            {"system": "R-local-fused", "fused": True, "pp": 8, "m": 16, "seq": 512},
        ]
    if only == "fused-smoke":
        return [
            {"system": "R-local-fused", "fused": True, "pp": 2, "m": 4, "seq": 512},
            {"system": "R-local", "pp": 2, "m": 4, "seq": 512},
        ]
    if only == "fused":
        cells = [
            {"system": "R-local-fused", "fused": True, "pp": 1, "m": 8, "seq": seq}
            for seq in (512, 2048)
        ]
        cells += [
            {"system": "R-local-fused", "fused": True, "pp": pp, "m": m, "seq": seq}
            for pp, m, seq in itertools.product((2, 4), (4, 8, 16), (512, 2048))
        ]
        return cells
    if only == "smoke":
        return [
            {"system": "M-plain", "pp": 2, "m": 4, "seq": 512},
            {"system": "R-local", "pp": 2, "m": 4, "seq": 512},
        ]
    if only not in ("", "headline"):
        raise ValueError(f"unknown mode {only!r}")
    # main grid ("" or "headline" = m=8 only), fusions off on both sides
    cells = []
    for seq in (512, 2048):  # PP=1 baselines (pipeline efficiency denominators)
        cells += [
            {"system": "M-plain", "pp": 1, "m": 8, "seq": seq},
            {"system": "R-local", "pp": 1, "m": 8, "seq": seq},
        ]
    for sysname, pp, m, seq in itertools.product(
        ("M-plain", "R-local"), (2, 4), (4, 8, 16), (512, 2048)
    ):
        if only == "headline" and m != 8:
            continue
        cells.append({"system": sysname, "pp": pp, "m": m, "seq": seq})
    return cells


# --- local entrypoint -----------------------------------------------------------


def _save(prefix, obj, indent=None):
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{prefix}{time.strftime('%Y%m%d-%H%M%S')}.json"
    with open(path, "w") as f:
        json.dump(obj, f, indent=indent)
    return path


def _stats(step_s, skip=10):
    """(median, p10, p90) step time after dropping the first `skip` steps."""
    t = sorted(step_s[skip:])
    return t[len(t) // 2], t[len(t) // 10], t[9 * len(t) // 10]


def _print_row(name, x, tokens_per_step, width):
    """One timing line for a result dict with step_s and loss, or its error."""
    if "step_s" not in x:
        print(f"  {name} FAILED:", x.get("error", "")[-2500:])
        return False
    med, p10, p90 = _stats(x["step_s"])
    print(
        f"  {name:{width}s} median {med * 1000:6.0f} ms = "
        f"{tokens_per_step / med / 1000:6.1f}k tok/s "
        f"(p10 {p10 * 1000:.0f}, p90 {p90 * 1000:.0f}) "
        f"loss {x['loss'][0]:.4f} -> {x['loss'][-1]:.4f}"
    )
    return True


def _print_loss_run(r, names):
    print(f"--- {r['pp']} GPU(s), {r['gpu']}: HF step-0 loss {r['hf_step0_loss']}")
    for line in r.get("megatron_layers", []):
        print("   ", line.strip()[-120:])
    for name in names:
        x = r[name]
        if "loss" not in x:
            print(f"  {name} FAILED:", x.get("error", "")[-3000:])
            continue
        med = _stats(x["step_s"])[0]
        print(
            f"  {name}: step0 {x['loss'][0]:.4f} last {x['loss'][-1]:.4f} "
            f"median {med * 1000:.0f} ms = {r['m'] * 4 * 512 / med / 1000:.1f}k tok/s"
        )


@app.local_entrypoint()
def main(reps: int = 1, only: str = ""):
    """--only picks the run: a grid name (see grid()), or one of loss, matched,
    samebox, samebox1, meg-bf16-{1,4,8}, profile1, gpubound1, gpubound48,
    hetero, fewmb."""
    prepare.remote()
    if only == "loss":
        print("text:", prepare_text.remote())
        res = loss_compare.remote()
        path = _save("loss_", res)
        print("HF reference step-0 loss:", res["hf_step0_loss"])
        for name in ("megatron", "rdsp"):
            r = res[name]
            if "loss" in r:
                med = _stats(r["step_s"])[0]
                print(
                    f"{name}: step0 {r['loss'][0]:.4f}  step{len(r['loss']) - 1} "
                    f"{r['loss'][-1]:.4f}  median step {med * 1000:.0f} ms"
                )
            else:
                print(name, "FAILED:", r.get("error", "")[-1500:])
        print("->", path)
        return
    if only == "fewmb":
        r = fewmb_x8.remote()
        path = _save("fewmb_", r)
        print(r["gpus"].strip())
        for name, _data, m, _cuts, _last in r["configs"]:
            x = r[name]
            if _print_row(name, x, m * 16 * 512, 20):
                print(f"      {x.get('stage_tags')} last_gpus={x.get('last_stage_gpus')}")
        print("->", path)
        return
    if only == "hetero":
        r = hetero_x8.remote()
        path = _save("hetero_", r)
        print(r["gpus"].strip())
        for name, _system, pp, _spec, last in r["configs"]:
            x = r[name]
            if _print_row(f"{name} ({pp + last - 1} GPUs)", x, 16 * 16 * 512, 32):
                print(f"      {x.get('stage_tags', '')} b={x.get('b', '')}")
            for c in r.get(name + "_check", []):
                print("       ", c[-100:])
        print("->", path)
        return
    if only == "gpubound48":
        res = []
        for fn in (gpubound_x8, gpubound_x4):  # one at a time: 10-GPU workspace limit
            r = fn.remote()
            res.append(r)
            print(f"=== {r['pp']} GPUs, one container ===\n{r['gpus'].strip()}")
            for name in r["order"]:
                _print_row(name, r[name], 16 * 16 * 512, 16)
            for c in r.get("megatron_te_ce_check", []):
                print("     ", c[-110:])
            print("   rdsp stages:", r["rdsp"].get("stage_tags"))
        print("->", _save("gpubound48_", res))
        return
    if only == "gpubound1":
        r = gpubound_x1.remote()
        path = _save("gpubound1_", r)
        print("GPU:", r["gpu"])
        for name in r["order"]:
            x = r[name]
            if "step_s" not in x:
                print(name, "FAILED:", x.get("error", "")[-2500:])
                continue
            # steps 20-30 carry the profiler
            t = sorted(x["step_s"][10:20] + x["step_s"][31:])
            med = t[len(t) // 2]
            kernel_ms = (
                sum(v[1] for k, v in x["kernels"].items() if not k.startswith("##"))
                / 1e3
                / x["steps"]
            )
            print(
                f"  {name:20s} median {med * 1000:5.0f} ms = {32768 / med / 1000:5.1f}k tok/s "
                f"| GPU kernels {kernel_ms:4.0f} ms/step "
                f"| loss {x['loss'][0]:.4f} -> {x['loss'][-1]:.4f}"
            )
        print("->", path)
        return
    if only == "profile1":
        r = profile_x1.remote()
        path = _save("profile1_", r)
        print("GPU:", r["gpu"])
        for name in ("megatron", "rdsp"):
            x = r[name]
            if "kernels" not in x:
                print(name, "FAILED:", x.get("error", "")[-2500:])
                continue
            busy = sum(v[1] for v in x["kernels"].values()) / 1e6
            print(
                f"{name}: window {x['wall_s']:.2f} s for 10 steps, kernel time {busy:.2f} s, "
                f"{len(x['kernels'])} distinct kernels"
            )
        print("->", path)
        return
    if only in ("samebox", "samebox1"):
        res = []
        # one at a time: 10-GPU workspace limit
        for fn in (samebox_x1,) if only == "samebox1" else (samebox_x8, samebox_x4):
            r = fn.remote()
            res.append(r)
            print(f"=== {r['pp']} GPUs, one container ===")
            print(r["gpus"].strip())
            for name in r["order"]:
                x = r[name]
                if _print_row(name, x, 16 * 2048, 14):
                    s = x["step_s"]
                    h = len(s) // 2
                    print(
                        f"      halves: steps 10-{h - 1} median "
                        f"{sorted(s[10:h])[(h - 10) // 2] * 1000:.0f} ms, "
                        f"steps {h}-{len(s) - 1} median "
                        f"{sorted(s[h:])[(len(s) - h) // 2] * 1000:.0f} ms"
                    )
                for c in r.get(name + "_check", []):
                    print("     ", c[-110:])
        print("->", _save("samebox_", res))
        return
    if only in ("meg-bf16-1", "meg-bf16-4", "meg-bf16-8"):
        fn = {"1": megatron_bf16_x1, "4": megatron_bf16_x4, "8": megatron_bf16_x8}[only[-1]]
        r = fn.remote()
        path = _save(f"{only}_", r)
        _print_loss_run(r, ("megatron",))
        print("->", path)
        return
    if only == "matched":
        # 1, 4 and 8 GPUs; same weights and text (16 microbatches x 2048 tokens
        # per step at every size), fusions on for both. 8+1 GPUs first, then 4.
        print("text:", prepare_text.remote(), prepare_text16.remote())
        h8, h1 = matched_x8.spawn(), matched_x1.spawn()
        res = [h1.get(), h8.get(), matched_x4.remote()]
        path = _save("matched_", res)
        for r in res:
            _print_loss_run(r, ("megatron", "rdsp"))
        print("->", path)
        return
    cells = grid(only)
    all_results = []
    for shape in (1, 2, 4, 8):  # one shape at a time: 10-GPU workspace limit
        mine = [c for c in cells if c.get("shape", c["pp"]) == shape]
        if not mine:
            continue
        for rep in range(reps):
            all_results.extend(RUN[shape].remote([dict(c, rep=rep) for c in mine]))
    path = _save("", all_results, indent=1)
    print(f"{len(all_results)} cells -> {path}")
