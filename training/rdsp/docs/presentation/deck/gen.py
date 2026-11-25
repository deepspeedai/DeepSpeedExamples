# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

import json, os

ROOT = os.path.dirname(os.path.abspath(__file__))
S = os.path.join(ROOT, "slides")

LIGHT, ALT, DARK = "#F6F5F1", "#EEECE6", "#16202B"
INK, BODY, MUTED, LINE = "#16202B", "#48525E", "#5F6A76", "#D6D2C8"
ACC = "#138A74"          # teal: rdsp stage-local / accent
MEG, DRV = "#4A67C9", "#C8742A"   # Megatron blue, rdsp driver-dispatch amber
SANS = "'IBM Plex Sans', Arial, sans-serif"
MONO = "'JetBrains Mono', 'Courier New', monospace"
CARD = "#FBFAF7"

slides = []


def sec(sid, inner, notes, bg=LIGHT, color=INK, extra=""):
    slides.append(sid)
    html = (f'<section id="{sid}" data-transition="fade" style="background:{bg}; color:{color}; '
            f'font-family:{SANS}; padding:128px; display:flex; flex-direction:column; '
            f'{"" if "gap:" in extra else "gap:40px"}{extra}">\n'
            f'{inner}\n<aside>{notes}</aside>\n</section>\n')
    open(os.path.join(S, sid + ".html"), "w").write(html)


def head(eyebrow, title, dark=False):
    c = "#EDEBE4" if dark else INK
    e = "#5CC7B1" if dark else ACC
    return (f'<div style="display:flex; flex-direction:column; gap:12px">'
            f'<p style="font-size:24px; font-weight:600; letter-spacing:2px; text-transform:uppercase; color:{e}">{eyebrow}</p>'
            f'<h2 style="font-size:60px; font-weight:600; line-height:1.15; color:{c}">{title}</h2></div>')


def card(title, text, fs=26, bg=CARD, pad=32, flex="flex:1"):
    return (f'<div style="{flex}; display:flex; flex-direction:column; gap:10px; background:{bg}; '
            f'padding:{pad}px; border:1px solid {LINE}; border-radius:14px">'
            f'<h3 style="font-size:30px; font-weight:600; line-height:1.2; color:{INK}">{title}</h3>'
            f'<p style="font-size:{fs}px; line-height:1.4; color:{BODY}">{text}</p></div>')


def table(headers, rows, widths, fs=24, first_bold=True, pad="10px 14px"):
    ths = "".join(f'<th style="width:{w}%; text-align:left">{h}</th>' for h, w in zip(headers, widths))
    trs = []
    for i, r in enumerate(rows):
        bg = f' style="background:{ALT}"' if i % 2 == 0 else ""
        tds = "".join(
            f'<td>{("<b>" + c + "</b>") if (j == 0 and first_bold) else c}</td>' for j, c in enumerate(r))
        trs.append(f"<tr{bg}>{tds}</tr>")
    return (f'<table style="font-size:{fs}px; color:{INK}; font-family:{SANS}; padding:{pad}">'
            f'<tr style="background:{DARK}">{ths.replace("<th ", "<th ")}</tr>{"".join(trs)}</table>')


# ---------------------------------------------------------------- 1 cover
sec("cover", f'''
<p style="font-size:26px; font-weight:600; letter-spacing:3px; text-transform:uppercase; color:#5CC7B1">rdsp · project review · September 2026</p>
<h1 style="font-size:104px; font-weight:600; line-height:1.05; color:#EDEBE4; width:1500px">Training one model across many GPUs, with Ray between the pieces</h1>
<div style="flex:1"></div>
<p style="font-size:32px; line-height:1.4; color:#B9C3CC; width:1400px">Pipeline parallelism on Ray + DeepSpeed: what we built, how it compares with NVIDIA Megatron, and where it falls short.</p>
''', "rdsp = ray_deepspeed_pipeline. External package: splits a Hugging Face model by depth into stages, each stage a group of Ray workers running an ordinary DeepSpeed engine. Deck covers design, correctness evidence, the Megatron benchmark, the profile that led to the stage-local redesign, and limitations.",
    bg=DARK, color="#EDEBE4", extra="; justify-content:space-between")

# ---------------------------------------------------------------- 2 goal
sec("goal", head("The goal", "Split a model by depth, and let each piece have its own GPUs") + f'''
<div style="display:flex; gap:32px">
{card("Split by depth", "Each stage owns a run of consecutive layers. Only activations cross between stages: once per boundary, per microbatch.")}
{card("Stages are not equal", "The stage holding the output head does about twice the work of a middle stage (85% vs 29% busy in our early runs). It should be allowed more GPUs.")}
{card("No forks", "DeepSpeed and Ray are used only through their public APIs: zero lines changed in either, checked against the pinned DeepSpeed source.")}
</div>
<p style="font-family:{MONO}; font-size:26px; line-height:1.55; color:#EDEBE4; background:{DARK}; padding:32px; border-radius:14px">engine, _, _, _ = rdsp.initialize(model=hf_model, config=ds_config,<br>&#160;&#160;&#160;&#160;pipeline_config=rdsp.PipelineConfig(stages=4), loss_fn=loss)<br>loss = engine.train_batch()&#160;&#160;&#160;# one call = one optimizer step</p>
''', "The user-facing API mirrors deepspeed.initialize. The optimizer and scheduler come from the ordinary DeepSpeed config and are built inside each stage, so initialize returns None for them rather than a look-alike object.")

# ---------------------------------------------------------------- 3 what we built
def tile(num, label):
    return (f'<div style="display:flex; flex-direction:column; gap:8px; background:{CARD}; padding:36px 40px; '
            f'border:1px solid {LINE}; border-radius:14px">'
            f'<p style="font-size:96px; font-weight:600; line-height:1.05; color:{ACC}">{num}</p>'
            f'<p style="font-size:28px; line-height:1.4; color:{BODY}">{label}</p></div>')

sec("built", head("Where it stands", "All nine phases are done; six of seven support rows pass on GPUs") + f'''
<div style="display:grid; grid-template-columns:1fr 1fr; gap:32px">
{tile("9 of 9", "phases complete, from the public API (P0) to stages of different shapes (P8)")}
{tile("6 of 7", "support rows validated on real GPUs; the 32-GPU row is blocked by account limits")}
{tile("32 / 32", "GPU tests pass on the final design (8×L4, NCCL); 290 CPU tests pass")}
{tile("0", "lines changed in DeepSpeed or Ray; everything lives in one external package")}
</div>
''', "Phases: P0 inputs, P1 API contract, P2 DeepSpeed reuse spike, P3 config/plan/partitioning, P4 schedule and coordinator, P5 Ray actors, P6 end-to-end parity, P7 checkpoint and recovery, P8 heterogeneous rows. Support rows: static boundary, resource mesh, data parallel + ZeRO, tensor parallel, sequence parallel, expert parallel + folding all supported; the design doc's 32-GPU four-stage topology is not run.")

# ---------------------------------------------------------------- 4 architecture
def pbox(l, t, w, h, title, lines, bg=CARD, border=LINE, tcol=INK):
    ps = "".join(f'<p style="font-size:24px; line-height:1.4; color:{BODY}">{x}</p>' for x in lines)
    return (f'<div style="position:absolute; left:{l}px; top:{t}px; width:{w}px; height:{h}px; display:flex; '
            f'flex-direction:column; gap:6px; background:{bg}; border:2px solid {border}; border-radius:14px; padding:20px 24px">'
            f'<h3 style="font-size:30px; font-weight:600; line-height:1.2; color:{tcol}">{title}</h3>{ps}</div>')

arch = f'''<div style="position:relative; width:1664px; height:630px">
{pbox(0, 0, 512, 160, "Your training script", ["Calls initialize(), then train_batch() in a loop"])}
{pbox(576, 0, 512, 160, "Plan compiler", ["Splits layers into stages; sets each stage's GPUs"])}
{pbox(1152, 0, 512, 160, "Coordinator (driver)", ["Sends each stage one command per step"], border=ACC)}
{pbox(0, 310, 368, 230, "Stage 0", ["Embedding + layers", "2 GPUs, data parallel, ZeRO-2"])}
{pbox(432, 310, 368, 230, "Stage 1", ["Layers", "2 GPUs, tensor parallel"])}
{pbox(864, 310, 368, 230, "Stage 2", ["Layers", "2 GPUs, sequence parallel"])}
{pbox(1296, 310, 368, 230, "Stage 3", ["Layers + output head", "4 GPUs, expert + tensor parallel"])}
<x-connector x1="512" y1="80" x2="576" y2="80" style="color:{MUTED}; border-width:3px"></x-connector>
<x-connector x1="1088" y1="80" x2="1152" y2="80" style="color:{MUTED}; border-width:3px"></x-connector>
<x-connector x1="1408" y1="160" x2="1408" y2="240" head="none" style="color:{ACC}; border-width:3px"></x-connector>
<x-connector x1="184" y1="240" x2="1480" y2="240" head="none" style="color:{ACC}; border-width:3px"></x-connector>
<x-connector x1="184" y1="240" x2="184" y2="310" style="color:{ACC}; border-width:3px"></x-connector>
<x-connector x1="616" y1="240" x2="616" y2="310" style="color:{ACC}; border-width:3px"></x-connector>
<x-connector x1="1048" y1="240" x2="1048" y2="310" style="color:{ACC}; border-width:3px"></x-connector>
<x-connector x1="1480" y1="240" x2="1480" y2="310" style="color:{ACC}; border-width:3px"></x-connector>
<x-connector x1="368" y1="425" x2="432" y2="425" head="both" style="color:{INK}; border-width:3px"></x-connector>
<x-connector x1="800" y1="425" x2="864" y2="425" head="both" style="color:{INK}; border-width:3px"></x-connector>
<x-connector x1="1232" y1="425" x2="1296" y2="425" head="both" style="color:{INK}; border-width:3px"></x-connector>
<p style="position:absolute; left:360px; top:186px; width:720px; font-size:24px; color:{ACC}; font-weight:600">one command per stage, per step (its whole 1F1B op list)</p>
<p style="position:absolute; left:0px; top:568px; width:1664px; font-size:26px; color:{BODY}; text-align:center">Each stage = one Ray worker per GPU, running a stock DeepSpeed engine. Neighbours exchange activations (→) and gradients (←) directly over NCCL.</p>
</div>'''
sec("arch", head("Architecture", "A driver that plans; stages that do the work") + arch,
    "The driver never holds weights or activations; only losses and ok/failed statuses come back to it. Each stage has its own torch.distributed world owned by DeepSpeed (for data/tensor/sequence/expert parallel inside the stage), plus two extra NCCL groups per GPU for stage-to-stage traffic, one per direction. The example shapes are illustrative: every stage's GPU count and parallelism is declared per stage.", extra="; gap:32px")

# ---------------------------------------------------------------- 5 one step
sec("step", head("One training step", "Each stage runs its own schedule; the driver only starts and finishes the step") + f'''
<div style="display:grid; grid-template-columns:1fr 1fr; gap:28px">
{card("1 · Hand out the work", "The driver takes exactly N microbatches and sends each stage one command: its forwards and backwards in 1F1B order.", pad=28)}
{card("2 · Stages work on their own", "Each GPU receives an activation, runs forward, sends it on; later receives a gradient, runs backward, sends it back.", pad=28)}
{card("3 · Nobody steps alone", "Only when every stage reports all backwards done does the driver tell all of them to run the optimizer, exactly once.", pad=28)}
{card("4 · Failures are clean", "Before the optimizer: a clean error, links rebuilt, retry is exact. During it: the pipeline locks until restored from a checkpoint.", pad=28)}
</div>
<p style="font-size:32px; font-weight:600; color:{ACC}">Ray calls per step at 8 stages: about 944 in the first design, about 16 now.</p>
''', "1F1B = one-forward-one-backward: each stage does a few warm-up forwards, then alternates, which bounds how many activations it keeps in memory. The first design had the driver send every forward and backward as a separate Ray call; the redesign sends one call per stage per step plus the optimizer apply.")

# ---------------------------------------------------------------- 6 shapes
rows = [["Data parallel (ZeRO 0/1/2)", "A different slice of the rows", "DeepSpeed ZeRO"],
        ["Tensor parallel", "The same rows; a slice of every weight matrix", "DeepSpeed AutoTP"],
        ["Sequence parallel", "The same rows; a different stretch of the sequence", "DeepSpeed Ulysses"],
        ["Expert parallel", "A different slice of the rows; some of the experts", "DeepSpeed AutoEP (+ folding)"]]
sec("shapes", head("Stages of different shapes", "Four ways a stage can use more GPUs") +
    table(["Way to add GPUs", "What each GPU gets", "Provided by"], rows, [30, 42, 28], fs=26) + f'''
<div style="display:flex; gap:32px">
{card("Changing shape at a boundary", "If one stage splits a batch in halves and the next in quarters, each receiving GPU takes exactly the pieces overlapping its slice. Nothing is gathered in one place.", fs=24, pad=28)}
{card("Keeping gradients right", "DeepSpeed averages gradients over data-parallel GPUs but adds them over sequence shards, so a gradient crossing a boundary is rescaled to match.", fs=24, pad=28)}
</div>
''', "Each stage declares its GPU count and tensor/sequence/expert degree; the data-parallel degree is derived. DeepSpeed reports each rank's coordinates at startup and rdsp refuses to run if they don't match the plan (mesh check).")

# ---------------------------------------------------------------- 7 vs megatron
vs = [["Who runs the schedule", "Every GPU process, inside one torchrun job", "Every stage's Ray workers, one command per step"],
      ["Stage shapes", "3D parallel (TP × PP × DP, + CP, EP), but every stage gets the same TP × DP", "Each stage its own GPU count and parallelism"],
      ["Inside a stage", "Megatron's own TP / CP / EP layers", "A stock DeepSpeed engine (ZeRO, AutoTP, Ulysses, AutoEP)"],
      ["Model code", "Megatron model definitions; HF weights converted", "The unmodified Hugging Face model, split by layer"],
      ["Between stages", "NCCL point-to-point", "NCCL point-to-point (first design: Ray-carried)"],
      ["Kernels", "Transformer Engine fusions", "HF layers + Liger fusions (in our benchmark)"],
      ["Failure handling", "Restart the job from a checkpoint", "Clean error and retry; lock + whole-pipeline restore"],
      ["Multi-machine", "Mature, production-scale", "Written; tested on one machine only"]]
sec("versus", head("rdsp vs Megatron", "Same pipelining idea, different ownership") +
    table(["", "NVIDIA Megatron-LM", "rdsp"], vs, [22, 39, 39], fs=24, pad="8px 14px"),
    "Megatron requires the world to factor as TP x PP x CP x DP with one uniform shape, so it cannot give the head-heavy last stage extra GPUs; its layout string can only move layers between stages. rdsp's reason to exist is per-stage shapes; the cost is an extra orchestration layer (Ray).")

# ---------------------------------------------------------------- 8 correctness
cr = [["First row (P6)", "1 · 1, Qwen3-0.6B, bf16", "Losses match within bf16 rounding"],
      ["Static boundary", "1 · 2 (rows split) · 1", "Losses within 1e-4"],
      ["Resource mesh", "2 · 4 · 2", "Losses within 1e-4"],
      ["Data parallel + ZeRO", "4 (ZeRO-2) · 2 (ZeRO-1) · 2 (ZeRO-0)", "Losses within 1e-4"],
      ["Tensor parallel", "1 · 2 (TP) · 4 (TP × DP)", "Every weight's update within 2.4e-5"],
      ["Sequence parallel", "2 (SP) · 4 (SP × DP) · 1", "Every weight's update within 2.4e-5"],
      ["Expert parallel + folding", "1 · 4 (EP) · 2 (EP folded with TP)", "Weight updates within 3e-4"],
      ["All four kinds combined", "2 (SP) · 2 (TP) · 2 (EP + TP) · 2 (TP)", "Weight updates within 2.4e-5"],
      ["Design doc's 32-GPU topology", "8 · 8 · 8 · 8 on 4 machines", "Not run (10-GPU account limit)"]]
sec("correct", head("Correctness", "Each row matches an unsplit, one-GPU model") +
    table(["Support row", "GPUs per stage", "Result (fp32 tiny models, SGD)"], cr, [30, 38, 32], fs=24) +
    f'<p style="font-size:24px; color:{MUTED}">Every passing row also passes a checkpoint round trip and a killed-GPU recovery. 8×L4, real DeepSpeed, stage-local design.</p>',
    "Parity uses SGD with momentum, not Adam: Adam ignores a uniformly mis-scaled gradient, so an early Adam-based test passed a deliberately broken build. Tests also refuse to run unless each step moves the loss by 20x the tolerance.", extra="; gap:32px")

# ---------------------------------------------------------------- 9 bugs
bugs = [("ZeRO-1/2 kept only the last microbatch", "DeepSpeed replaced instead of adding each microbatch's gradient. Fix: only the last backward is the accumulation boundary."),
        ("Qwen3 head norms under tensor parallel", "Each GPU saw only its own heads' share of the norm gradients, so GPUs drifted. Fix: sum across the group."),
        ("Sequence-parallel updates doubled", "DeepSpeed adds over sequence shards but averages over data-parallel GPUs; the first scale rule got it wrong."),
        ("Folded experts sent a partial gradient", "Each GPU held a different partial gradient below expert layers. Fix: average at the expert layer's input."),
        ("Failed steps left traces", "Gradients and random state from an abandoned step leaked into the retry. Fix: discard both; retries are now exact."),
        ("Silent per-stage clipping", "DeepSpeed clips by default, per stage rather than across the model. Now off; explicit clipping is rejected.")]
sec("bugs", head("What the strict tests caught", "Six real bugs, all fixed") +
    '<div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:24px">' +
    "".join(card(t, x, fs=24, pad=28) for t, x in bugs) + "</div>",
    "All six were found by the parity tests on real GPUs after tightening them (fp32, SGD, 1e-4 tolerance, sensitivity precondition, per-parameter update comparison). None needed a change to DeepSpeed.")

# ---------------------------------------------------------------- 10 checkpoints
sec("ckpt", head("Checkpoints and recovery", "A pipeline checkpoint is one consistent moment across every GPU") + f'''
<div style="display:grid; grid-template-columns:1fr 1fr; gap:28px">
{card("A manifest, written last", "Each stage saves with DeepSpeed's own checkpoint. A manifest listing every file's fingerprint is written only when all have finished.", pad=28)}
{card("Everything to continue exactly", "Weights, optimizer, schedule, random-number state and data position. A fresh pipeline that loads it matches the original bit for bit.", pad=28)}
{card("Checked before anything loads", "A wrong plan, a missing file or a changed file is rejected before any stage is touched; each stage re-checks its own files.", pad=28)}
{card("Recovery rebuilds everything", "If a worker died, loading tears the pipeline down and rebuilds it on the same machines, then loads.", pad=28)}
</div>
''', "A folder without a manifest is an unfinished save and is never loaded, so a crash mid-save can't produce a half-new checkpoint. Loading is the only way out of the locked state after a partial optimizer step.")

# ---------------------------------------------------------------- 11 setup
fus = [["Starting weights", "Pretrained Qwen3-0.6B, both"], ["Data", "WikiText-103, same batches, same order"],
       ["Layers per stage", "Identical (8 GPUs: 4-4-4-4-3-3-3-3)"],
       ["Fused kernels", "RoPE, SwiGLU, norm, cross-entropy, Adam, flash attention"],
       ["Optimizer", "Adam, lr 1e-5, no clipping, fp32 master weights"],
       ["Gradient summing", "rdsp bf16; Megatron fp32 (default) or bf16"]]
sec("setup", head("Benchmark setup", "Same weights, data, kernels and chip; compared only on the same machine") + f'''
<div style="display:flex; gap:48px">
<div style="flex:1; display:flex; flex-direction:column; gap:16px">
<h3 style="font-size:30px; font-weight:600">How it was run</h3>
<ul style="font-size:26px; line-height:1.45; color:{BODY}">
<li>Megatron-Core's own 1F1B schedule in a lean training loop, fusions on</li>
<li>rdsp, stage-local design, Liger fused kernels</li>
<li>150 steps of 16 microbatches × 2,048 tokens, at 1, 4 and 8 GPUs</li>
<li>Pinned H100 (checked on every GPU); all setups back to back on one machine per GPU count</li>
<li>One image: torch 2.9, CUDA 13, DeepSpeed (pinned), Megatron-Core 0.19.2</li>
</ul></div>
<div style="flex:1; display:flex; flex-direction:column; gap:16px">
<h3 style="font-size:30px; font-weight:600">Matched on both sides</h3>
{table(["Setting", "Value"], fus, [38, 62], fs=23, first_bold=True)}
</div></div>
''', "Megatron's HF weights were mapped by hand into its fused layout; every Megatron parameter is checked to receive exactly one weight, and the layer coverage across ranks is checked. The one remaining mismatch is gradient summing precision: DeepSpeed ZeRO-0 bf16 sums microbatch gradients in bf16; Megatron defaults to fp32 (its --grad-reduce-in-bf16 option matches rdsp). Megatron also keeps fused weight-gradient accumulation and disabled Python GC, which only help it.", extra="; gap:32px")

# ---------------------------------------------------------------- 12 headline chart
Y0, PX = 540, 500 / 120
groups = [("1 GPU", [22.1, 22.4, 27.8]), ("4 GPUs", [59.3, 57.6, 75.2]), ("8 GPUs", [86.5, 88.1, 113.6])]
cols = [MEG, "#C9D3F2", ACC]
rects, labels = [], []
for gi, (name, vals) in enumerate(groups):
    c = 90 + 161.7 * (2 * gi + 1)
    for bi, v in enumerate(vals):
        x = c - 100 + bi * 68
        if v is None:
            labels.append(f'<p style="position:absolute; left:{x - 12:.0f}px; top:{Y0 - 44}px; width:88px; font-size:22px; color:{MUTED}; text-align:center">H200*</p>')
            continue
        h = v * PX
        dash = ' stroke="#4A67C9" stroke-width="3" stroke-dasharray="8 5"' if bi == 1 else ""
        rects.append(f'<rect x="{x}" y="{Y0 - h:.1f}" width="64" height="{h:.1f}" rx="4" fill="{cols[bi]}"{dash}/>')
        labels.append(f'<p style="position:absolute; left:{x - 8:.0f}px; top:{Y0 - h - 40:.0f}px; width:80px; font-size:24px; font-weight:600; color:{INK}; text-align:center">{v:g}</p>')
    labels.append(f'<p style="position:absolute; left:{c - 80:.0f}px; top:{Y0 + 14}px; width:160px; font-size:24px; color:{BODY}; text-align:center">{name}</p>')
grid = "".join(f'<line x1="90" y1="{Y0 - k * PX:.1f}" x2="1060" y2="{Y0 - k * PX:.1f}" stroke="{LINE}" stroke-width="2"/>' for k in (40, 80, 120))
ylabels = "".join(f'<p style="position:absolute; left:0px; top:{Y0 - k * PX - 17:.0f}px; width:76px; font-size:24px; color:{MUTED}; text-align:right">{k}k</p>' for k in (0, 40, 80, 120))
svg = (f'<svg aria-label="Tokens per second on pinned H100: 1 GPU Megatron fp32 22.1k, bf16 22.4k, rdsp 27.8k; 4 GPUs 59.3k, 57.6k, 75.2k; 8 GPUs 86.5k, 88.1k, 113.6k" '
       f'style="position:absolute; left:0; top:0" width="1080" height="600" viewBox="0 0 1080 600">'
       f'{grid}<line x1="90" y1="{Y0}" x2="1060" y2="{Y0}" stroke="{MUTED}" stroke-width="2"/>{"".join(rects)}</svg>')

def sw(color, text, dashed=False):
    b = "border:3px dashed #4A67C9; box-sizing:border-box; " if dashed else ""
    return (f'<div style="display:flex; gap:12px; align-items:center"><div style="width:28px; height:28px; {b}background:{color}; border-radius:4px"></div>'
            f'<p style="font-size:24px; color:{INK}">{text}</p></div>')

legend = f'<div style="display:flex; gap:40px">{sw(MEG, "Megatron, fp32 summing (default)")}{sw("#C9D3F2", "Megatron, bf16 summing", True)}{sw(ACC, "rdsp, stage-local")}</div>'
side = (f'<div style="position:absolute; left:1150px; top:0px; width:514px; display:flex; flex-direction:column; gap:18px; background:{CARD}; '
        f'border:1px solid {LINE}; border-radius:14px; padding:30px">'
        f'<p style="font-size:26px; line-height:1.4; color:{BODY}"><b style="color:{INK}">Everything matched, same machine:</b> rdsp is <b style="color:{ACC}">1.24× faster at 1 GPU, 1.30× at 4, 1.29× at 8</b> than Megatron summing gradients in bf16 like rdsp.</p>'
        f'<p style="font-size:25px; line-height:1.4; color:{BODY}"><b style="color:{INK}">Caveat:</b> small (2,048-token) microbatches, where Python overhead dominates. In GPU-bound training: 1.10×, 1.11×, a tie at 8 (two slides on).</p>'
        '</div>')
sec("headline", head("Headline result · pinned H100, one machine per GPU count", "Tokens per second (thousands), fully matched") + legend +
    f'<div style="position:relative; width:1664px; height:600px">{svg}{"".join(labels)}{ylabels}{side}</div>',
    "4 and 8 GPUs: one pinned-H100 machine each (GPU name checked on every GPU before running), Megatron fp32, Megatron bf16 and rdsp back to back, 60 steps of 16 x 2,048 tokens, median of steps 10-59. Step times 8 GPUs: 379 / 372 / 288 ms; 4 GPUs: 553 / 569 / 436 ms; 1 GPU (100 steps, median of 10-99): 1,486 / 1,465 / 1,179 ms, no drift between halves. An earlier cross-machine 1-GPU estimate of 1.11x was wrong. Earlier headline numbers mixed H100 (Megatron) and H200 (rdsp) and are withdrawn.", extra="; gap:28px")

# ---------------------------------------------------------------- 12a why faster
GBARS = [("Small microbatches · 2,048 tokens", [("Megatron, default CE", 482, 713, MEG), ("rdsp", 437, 577, ACC)]),
         ("Large microbatches · 8,192 tokens", [("Megatron, default CE", 385, 202, MEG), ("Megatron, TE CE", 322, 163, MEG), ("rdsp", 342, 100, ACC)])]
gl, gpw, gmax, gbh = 330, 700, 1200, 48
gsv, glab, gy = [], [], 0
for title, rows in GBARS:
    glab.append(f'<p style="position:absolute; left:0; top:{gy}px; width:1100px; font-size:26px; font-weight:600; color:{INK}">{title}</p>')
    gy += 48
    for n, busy, idle, col in rows:
        wb, wi = busy / gmax * gpw, idle / gmax * gpw
        gsv.append(f'<rect x="{gl}" y="{gy}" width="{wb:.1f}" height="{gbh}" rx="4" fill="{col}"/>'
                   f'<rect x="{gl + wb + 3:.1f}" y="{gy}" width="{max(0, wi - 3):.1f}" height="{gbh}" rx="4" fill="#E4E1D8" stroke="{LINE}" stroke-width="2"/>')
        glab.append(f'<p style="position:absolute; left:0; top:{gy + 9}px; width:{gl - 20}px; font-size:24px; color:{BODY}; text-align:right">{n}</p>')
        glab.append(f'<p style="position:absolute; left:{gl + wb + wi + 14:.0f}px; top:{gy + 9}px; width:130px; font-size:24px; font-weight:600; color:{INK}">{busy + idle} ms</p>')
        gy += gbh + 16
    gy += 22
gsvg = (f'<svg aria-label="Step time split into GPU computing and idle. Small microbatches: Megatron 482+713, rdsp 437+577 ms. Large: Megatron default CE 385+202, TE CE 322+163, rdsp 342+100 ms." '
        f'style="position:absolute; left:0; top:0" width="1160" height="{gy}" viewBox="0 0 1160 {gy}">{"".join(gsv)}</svg>')
gside = (f'<div style="flex:1; display:flex; flex-direction:column; gap:20px">'
         f'<div style="display:flex; flex-direction:column; gap:10px">{sw(MEG, "Megatron: GPU computing")}{sw(ACC, "rdsp: GPU computing")}{sw("#E4E1D8", "GPU idle, waiting for the CPU")}</div>'
         f'<p style="font-size:25px; line-height:1.4; color:{BODY}"><b style="color:{INK}">Small microbatches:</b> the GPU idles 57–60% of each step; Python overhead sets the pace. That is where rdsp\'s 1.2–1.3× lead comes from.</p>'
         f'<p style="font-size:25px; line-height:1.4; color:{BODY}"><b style="color:{INK}">Large microbatches:</b> rdsp leads by only <b style="color:{ACC}">1.10×</b> against Megatron\'s TE cross-entropy, and Megatron\'s GPU work is slightly less (322 vs 342 ms).</p></div>')
sec("why", head("Why rdsp is faster on one GPU", "Mostly less idle time, not faster math") +
    f'''<div style="display:flex; gap:48px; align-items:flex-start">
<div style="position:relative; width:1160px; height:{gy}px">{gsvg}{"".join(glab)}</div>{gside}</div>''',
    "One pinned H100, torch.profiler on both frameworks over optimizer steps 20-29; idle = step time minus GPU kernel time. Small microbatches (profile1_20260923-152602): Megatron 1,195 ms, rdsp 1,014 ms per step; ~24.5k vs ~34.4k kernel launches per step; Megatron's default ('native') fused cross-entropy costs ~82 ms per step vs ~17 ms for Liger's. Large microbatches (gpubound1_20260923-153735): 16 rows x 512 tokens x 4 microbatches, same tokens per step; Megatron native CE 587 ms, TE CE 485 ms, rdsp 442 ms; losses agree within 0.1% median per step. The groups ran on different machines. Pipelined 4/8-GPU runs with large microbatches are not measured yet.", extra="; gap:28px")

# ---------------------------------------------------------------- 12a2 GPU-bound
GY0, GPX = 540, 500 / 300
ggroups = [("1 GPU", [67.6, 74.1], "1.10×"), ("4 GPUs", [174.2, 192.8], "1.11×"), ("8 GPUs", [248.4, 251.9], "1.02×")]
grects, glabels = [], []
for gi, (name, vals, ratio) in enumerate(ggroups):
    c = 90 + 161.7 * (2 * gi + 1)
    for bi, v in enumerate(vals):
        x = c - 70 + bi * 72
        hh = v * GPX
        grects.append(f'<rect x="{x}" y="{GY0 - hh:.1f}" width="68" height="{hh:.1f}" rx="4" fill="{[MEG, ACC][bi]}"/>')
        glabels.append(f'<p style="position:absolute; left:{x - 10:.0f}px; top:{GY0 - hh - 40:.0f}px; width:88px; font-size:24px; font-weight:600; color:{INK}; text-align:center">{v:g}</p>')
    glabels.append(f'<p style="position:absolute; left:{c - 100:.0f}px; top:{GY0 + 14}px; width:200px; font-size:24px; color:{BODY}; text-align:center">{name} · <b style="color:{ACC}">{ratio}</b></p>')
ggrid = "".join(f'<line x1="90" y1="{GY0 - k * GPX:.1f}" x2="1060" y2="{GY0 - k * GPX:.1f}" stroke="{LINE}" stroke-width="2"/>' for k in (100, 200, 300))
gyl = "".join(f'<p style="position:absolute; left:0px; top:{GY0 - k * GPX - 17:.0f}px; width:76px; font-size:24px; color:{MUTED}; text-align:right">{k}k</p>' for k in (0, 100, 200, 300))
gbsvg = (f'<svg aria-label="GPU-bound tokens per second: 1 GPU Megatron 67.6k vs rdsp 74.1k; 4 GPUs 174.2k vs 192.8k; 8 GPUs 248.4k vs 251.9k" '
         f'style="position:absolute; left:0; top:0" width="1080" height="600" viewBox="0 0 1080 600">{ggrid}'
         f'<line x1="90" y1="{GY0}" x2="1060" y2="{GY0}" stroke="{MUTED}" stroke-width="2"/>{"".join(grects)}</svg>')
gbside = (f'<div style="position:absolute; left:1150px; top:0px; width:514px; display:flex; flex-direction:column; gap:18px; background:{CARD}; '
          f'border:1px solid {LINE}; border-radius:14px; padding:30px">'
          f'<p style="font-size:26px; line-height:1.4; color:{BODY}"><b style="color:{INK}">GPU-bound training:</b> rdsp is <b style="color:{ACC}">on par with Megatron or faster</b>: 1.10× on one GPU, 1.11× at 4, a tie at 8.</p>'
          f'<p style="font-size:25px; line-height:1.4; color:{BODY}">rdsp\'s pipeline adds no measurable cost over Megatron\'s; Ray is invisible at this scale.</p>'
          f'<p style="font-size:22px; line-height:1.4; color:{MUTED}">Megatron at its fastest matched options: Transformer Engine cross-entropy, bf16 summing.</p></div>')
sec("gpubound", head("GPU-bound result · large microbatches, pinned H100", "Tokens per second (thousands), fully matched") +
    f'<div style="display:flex; gap:40px">{sw(MEG, "Megatron (TE cross-entropy, bf16 summing)")}{sw(ACC, "rdsp, stage-local")}</div>' +
    f'<div style="position:relative; width:1664px; height:600px">{gbsvg}{"".join(glabels)}{gyl}{gbside}</div>',
    "8,192-token microbatches. 1 GPU: 4 microbatches per step (gpubound1_20260923-153735); 4 and 8 GPUs: 16 microbatches per step, 36 steps, median of 10-35 (gpubound48_20260923-155657). Step times: 1 GPU 485 vs 442 ms, 4 GPUs 753 vs 680 ms, 8 GPUs 528 vs 520 ms. GPU name, gradient dtype and layer split checked on every rank. Final losses 2.7942 (Megatron) vs 2.7939-2.7949 (rdsp).", extra="; gap:28px")

# ---------------------------------------------------------------- 12b loss curves
MS = json.load(open(os.path.join(ROOT, "..", "matched_series.json")))
def smooth(a, w=10):
    return [sum(a[max(0, i - w + 1):i + 1]) / (i + 1 - max(0, i - w + 1)) for i in range(len(a))]
LW, LH, LL, LT, LPW, LPH = 1060, 560, 80, 20, 960, 470
def lx(i): return LL + i / 149 * LPW
def ly(v): return LT + (1 - (v - 2.5) / (3.4 - 2.5)) * LPH
lines = ""
for key, col, wdt in (("mf", MEG, 4), ("mb", "#C8742A", 4), ("rd", ACC, 4)):
    pts = " ".join(f"{lx(i):.1f},{ly(v):.1f}" for i, v in enumerate(smooth(MS["8"][key])))
    lines += f'<polyline fill="none" stroke="{col}" stroke-width="{wdt}" stroke-linejoin="round" points="{pts}"/>'
lgrid = "".join(f'<line x1="{LL}" y1="{ly(v):.1f}" x2="{LL + LPW}" y2="{ly(v):.1f}" stroke="{LINE}" stroke-width="2"/>' for v in (2.6, 2.8, 3.0, 3.2, 3.4))
lsvg = f'<svg aria-label="Training loss over 150 steps at 8 GPUs: Megatron fp32, Megatron bf16 and rdsp overlap, falling from 3.34 to about 2.65" style="position:absolute; left:0; top:0" width="{LW}" height="{LH}" viewBox="0 0 {LW} {LH}">{lgrid}{lines}</svg>'
lyl = "".join(f'<p style="position:absolute; left:0px; top:{ly(v) - 16:.0f}px; width:66px; font-size:22px; color:{MUTED}; text-align:right">{v:.1f}</p>' for v in (2.6, 2.8, 3.0, 3.2, 3.4))
lxl = "".join(f'<p style="position:absolute; left:{lx(i) - 40:.0f}px; top:{LT + LPH + 10}px; width:80px; font-size:22px; color:{MUTED}; text-align:center">{i}</p>' for i in (0, 50, 100, 149))
fl = [["1", "2.6481 / 2.6475", "2.6475"], ["4", "2.6474 / 2.6479", "2.6494"], ["8", "2.6474 / 2.6478", "2.6494"]]
sec("loss", head("Same training", "The loss curves overlap at every GPU count") +
    f'''<div style="display:flex; gap:40px; align-items:flex-start">
<div style="position:relative; width:{LW}px; height:{LH + 50}px">{lsvg}{lyl}{lxl}
<p style="position:absolute; left:{LL}px; top:{LT + LPH + 44}px; width:{LPW}px; font-size:22px; color:{BODY}; text-align:center">training step · 8 GPUs · 10-step average</p></div>
<div style="flex:1; display:flex; flex-direction:column; gap:22px">
<div style="display:flex; flex-direction:column; gap:10px">{sw(MEG, "Megatron, fp32 summing")}{sw("#C8742A", "Megatron, bf16 summing")}{sw(ACC, "rdsp")}</div>
{table(["GPUs", "Final: Megatron fp32 / bf16", "rdsp"], fl, [18, 52, 30], fs=22, pad="6px 10px")}
<p style="font-size:24px; line-height:1.4; color:{BODY}">Both start at the unsplit model's loss (3.338). Per step they differ by 0.05% (median), 1.5% at worst: bf16 rounding.</p>
</div></div>''',
    "Same pretrained weights and the same WikiText-103 batches in the same order. Plain Hugging Face model on step 0's batch: 3.3383; Megatron 3.3371, rdsp 3.3369 at every GPU count. Mean of the last 50 steps: Megatron 2.7285, rdsp 2.7289-2.7291. The bf16-summing Megatron curve is indistinguishable from the fp32 one, so the precision difference changes speed, not learning, at this scale.", extra="; gap:28px")

# ---------------------------------------------------------------- 13 detailed table
det = [["2", "4", "512", "20.7k", "25.0k", "28.1k"], ["2", "8", "512", "23.5k", "29.4k", "33.6k"],
       ["2", "16", "512", "23.7k", "31.8k", "37.5k"], ["4", "4", "512", "25.9k", "26.7k", "30.2k"],
       ["4", "8", "512", "32.4k", "33.5k", "37.9k"], ["4", "16", "512", "36.0k", "40.3k", "43.5k"],
       ["2", "4", "2048", "20.5k", "25.9k", "30.1k"], ["2", "8", "2048", "22.4k", "29.4k", "32.8k"],
       ["2", "16", "2048", "24.0k", "33.4k", "35.7k"], ["4", "4", "2048", "26.9k", "26.2k", "30.6k"],
       ["4", "8", "2048", "33.6k", "33.1k", "38.2k"], ["4", "16", "2048", "38.8k", "40.9k", "43.6k"]]
sec("detail", head("Appendix · earlier grid, first design vs pretrain_gpt.py, H100", "Tokens per second across the grid (separate machines)") +
    table(["Stages", "Microbatches", "Sequence", "Megatron", "rdsp, via Ray", "rdsp, via RDT"], det, [13, 17, 15, 18, 18, 19], fs=24, first_bold=False, pad="6px 14px"),
    "These cells were measured before the stage-local redesign: rdsp with driver dispatch, boundary tensors carried by Ray's object store or by Ray Direct Transport (RDT, GPU-to-GPU). rdsp is faster in 22 of 24 comparisons; the two exceptions are 4 stages with 4 microbatches at sequence 2048 (object store: 26.2k vs 26.9k) and 4 stages x 8 microbatches x 2048 (33.1k vs 33.6k), both within noise. 1-GPU baselines: Megatron 14.3k / 14.9k, rdsp 23.6k / 22.9k tokens/s.", extra="; gap:28px")

# ---------------------------------------------------------------- 14 scaling
sc = [["Machine A (matched run)", "441 ms", "321 ms", "1.37×"],
      ["Machine B (pinned rerun)", "553 ms", "436 ms", "1.27×"]]
sec("scaling", head("Why only same-machine ratios count", "Two H100 machines, same 4-GPU job: 25% apart, similar ratio") +
    table(["4 GPUs, Megatron fp32 summing vs rdsp", "Megatron", "rdsp", "rdsp faster by"], sc, [40, 20, 20, 20], fs=26) + f'''
<div style="display:flex; gap:32px">
{card("What it means", "Absolute speed depends on which machine you get, by up to ~25% here. The rdsp-to-Megatron ratio stayed in a narrow band, so ratios measured on one machine are what we report.", fs=24, pad=28)}
{card("Consequence", "Speed-up from 1 to 8 GPUs compares different machines, so we don't claim a scaling result. An earlier 'Megatron scales better' chart compared H200s against H100s and is withdrawn.", fs=24, pad=28)}
</div>
''', "Machine A: matched_20260923-133517 (4 GPUs). Machine B: samebox_20260923-144205 (4 GPUs). Both H100 80GB HBM3, same image, same data and settings. A same-machine repeat would tell run-to-run noise apart from machine-to-machine differences.")

# ---------------------------------------------------------------- 15 profile
pr = [["Step time", "742 ms", "831 ms"], ["Each GPU busy", "430–480 ms", "285–340 ms"],
      ["Each GPU idle", "270–315 ms (37–42%)", "495–550 ms (60–66%)"], ["Idle expected by 1F1B", "16%", "30%"],
      ["Layer compute per call", "12.8 ms", "8.4 ms"], ["Ray overhead per call", "4.7 ms", "4.9 ms"],
      ["Hop to next stage (median / slowest 10%)", "2.3 / 7.4 ms", "2.5 / 22.9 ms"]]
sec("profile", head("Why the first design stopped scaling", "Profile: the time went to Ray plumbing, not to the GPUs") + f'''
<div style="display:flex; gap:40px">
<div style="flex:3">{table(["First design, H200", "4 stages", "8 stages"], pr, [48, 26, 26], fs=24)}</div>
<div style="flex:2; display:flex; flex-direction:column; gap:16px">
<p style="font-size:26px; line-height:1.4; color:{BODY}">Every forward and backward was its own Ray call: <b style="color:{INK}">~944 per step</b> at 8 stages.</p>
<p style="font-size:26px; line-height:1.4; color:{BODY}">More stages halved each call's compute, but not the ~5 ms of overhead. A 1F1B step runs back-to-back slots, so the overhead sits on the critical path every time.</p>
<p style="font-size:26px; line-height:1.4; color:{BODY}">Slow hops (23 ms) stall every stage behind them.</p>
</div></div>
''', "Measured with CUDA-event timers around every engine call plus Ray's task timeline, 16 microbatches, sequence 512. The ~5 ms per call is about 2 ms receiving and unpacking the input, 1.3 ms storing the output, 1.5 ms for the RDT send/receive tasks. rdsp's compute per layer is the same at 1, 4 and 8 GPUs; the driver's submission loop (~70 ms/step) is off the critical path.")

# ---------------------------------------------------------------- 16 fix
fx = [["4 stages", "742 ms · 44.2k", "602 ms · 54.4k", "1.23×"],
      ["8 stages", "831 ms · 39.4k", "392 ms · 83.6k", "2.12×"]]
cpu = [["2 stages", "15.5 ms", "7.2 ms"], ["4 stages", "23.6 ms", "7.8 ms"], ["8 stages", "47.4 ms", "11.9 ms"]]
sec("fix", head("The fix", "Each stage runs its own schedule and talks to its neighbours directly") +
    table(["H200, 16 microbatches, seq 512", "First design", "Stage-local", "Faster by"], fx, [31, 23, 23, 23], fs=26) + f'''
<div style="display:flex; gap:40px">
<div style="flex:1; display:flex; flex-direction:column; gap:14px">
<h3 style="font-size:30px; font-weight:600">What changed</h3>
<p style="font-size:26px; line-height:1.4; color:{BODY}">One command per stage per step. Two extra NCCL groups per GPU, one per direction, carry activations and gradients. Numbers are bit-identical to the first design.</p></div>
<div style="flex:1; display:flex; flex-direction:column; gap:14px">
<h3 style="font-size:30px; font-weight:600">Overhead alone (CPU, tiny model)</h3>
{table(["", "First design", "Stage-local"], cpu, [34, 33, 33], fs=24)}</div></div>
''', "One group per direction avoids a stage's gradient receive queueing behind its own activation send on an NCCL stream. After a failed step the links are aborted and rebuilt immediately so a stuck GPU transfer can't later kill a worker. 8-stage throughput is 2.1x the first design; both designs measured on H200 machines (separate machines, allow ~20%). A Megatron column shown here earlier came from H100s and was removed.", extra="; gap:32px")

# ---------------------------------------------------------------- 17 pipelines drawn as boxes
TEAL_SOFT = "#D3ECE6"
def pipe(stages):
    """stages: list of (label, gpus). One box per stage; a 2-GPU stage is drawn teal with a badge."""
    out = []
    for label, g in stages:
        if g == 2:
            out.append(f'<div style="display:flex; flex-direction:column; align-items:center; justify-content:center; min-width:104px; height:50px; '
                       f'padding:0 10px; background:{TEAL_SOFT}; border:3px solid {ACC}; border-radius:8px">'
                       f'<p style="font-size:22px; font-weight:600; color:{INK}; line-height:1.1">{label}</p>'
                       f'<p style="font-size:16px; font-weight:600; color:{ACC}; line-height:1.1">2 GPUs</p></div>')
        else:
            out.append(f'<div style="display:flex; align-items:center; justify-content:center; min-width:60px; height:50px; padding:0 10px; '
                       f'background:{CARD}; border:2px solid {LINE}; border-radius:8px">'
                       f'<p style="font-size:22px; font-weight:600; color:{INK}">{label}</p></div>')
    arrow = f'<p style="font-size:20px; color:{MUTED}">→</p>'
    return f'<div style="display:flex; align-items:center; gap:6px">{arrow.join(out)}</div>'

EQ5 = [("7", 1), ("7", 1), ("7", 1), ("6", 1), ("1+H", 1)]
def prow(who, stages, step, tps, verdict, bold=False, shade=False):
    bg = f' style="background:{ALT}"' if shade else ""
    w = "600" if bold else "400"
    return (f'<tr{bg}><td style="font-weight:{w}">{who}</td><td>{pipe(stages)}</td><td class="r">{step}</td>'
            f'<td class="r">{tps}</td><td class="r" style="font-weight:600">{verdict}</td></tr>')
def grp(text):
    return f'<tr><td colspan="5" style="padding-top:10px; font-size:20px; font-weight:600; color:{ACC}; letter-spacing:1px; text-transform:uppercase">{text}</td></tr>'
core_tbl = (f'<table style="font-size:22px; color:{INK}; font-family:{SANS}">'
    f'<tr style="background:{DARK}"><th style="width:22%; text-align:left">Setup (5 GPUs)</th><th style="width:40%; text-align:left">Pipeline: layers per stage</th>'
    f'<th style="width:11%; text-align:right">Step</th><th style="width:12%; text-align:right">Tokens/s</th><th style="width:15%; text-align:right">vs 5 equal</th></tr>'
    + grp("Test 1 · 16 microbatches")
    + prow("rdsp, 5 equal stages", EQ5, "451 ms", "290.6k", "baseline", shade=True)
    + prow("rdsp, uneven", [("8", 1), ("8", 1), ("8", 1), ("4+H", 2)], "484 ms", "270.8k", "0.93×", bold=True)
    + grp("Test 2 · 16 microbatches, fairer layout")
    + prow("rdsp, 5 equal stages", EQ5, "456 ms", "287.3k", "baseline", shade=True)
    + prow("rdsp, uneven", [("7", 1), ("7", 1), ("7", 1), ("7+H", 2)], "501 ms", "261.8k", "0.91×", bold=True)
    + grp("Test 3 · 4 microbatches (extra stages cost more)")
    + prow("rdsp, 5 equal stages", EQ5, "187 ms", "174.9k", "baseline", shade=True)
    + prow("rdsp, uneven", [("7", 1), ("7", 1), ("7", 1), ("7+H", 2)], "193 ms", "169.9k", "0.97×", bold=True)
    + '</table>')
core_css = (f'<style>#nothelp td {{ padding:5px 12px; vertical-align:middle }} #nothelp th {{ padding:10px 12px; color:#EDEBE4 }} '
            f'#nothelp td.r {{ text-align:right }}</style>')
sec("nothelp", core_css + head("The core idea, tested fairly", "A 2-GPU busiest stage lost to a plain 5th stage, 3 of 3") +
    f'''<p style="font-size:22px; color:{MUTED}">Each box is one pipeline stage; the number is its layers (H = the output head). Teal = the stage given 2 GPUs. Same machine within each test.</p>
{core_tbl}
<p style="font-size:24px; line-height:1.4; color:{BODY}"><b style="color:{INK}">Why:</b> two GPUs made that stage only ~1.5× faster, not 2×: half a microbatch is too little work per GPU, plus a gradient sync and split data. It could still win on a much larger model, where one stage has enough work to split.</p>''',
    "hetero_20260923-160952 (test 1) and fewmb_20260923-164633 (tests 2 and 3); rdsp only, one 8xH100 machine per file, H100 checked on every GPU; 16 rows x 512 tokens per microbatch, 36 steps, median of 10-35. Tests 2 and 3 were predicted before running (uneven ~1.05x and ~1.14x faster); both predictions failed. All setups reach the same loss (median per-step difference <= 0.13%). Even the uneven setup beat Megatron's best 5-GPU layout (270.8k vs 247.5k, test 1).", extra="; gap:22px")

# ---------------------------------------------------------------- 17a balanced layouts vs Megatron
bl = [("4 GPUs", [("9", 1), ("9", 1), ("9", 1), ("1+H", 1)], "631 ms · 207.8k", "535 ms · 245.1k", "1.18×"),
      ("5 GPUs", EQ5, "530 ms · 247.5k", "451 ms · 290.6k", "1.18×")]
bl_rows = "".join(
    f'<tr{" style=&quot;background:" + ALT + "&quot;" if k == 0 else ""}><td style="font-weight:600">{g}</td><td>{pipe(st)}</td>'
    f'<td class="r">{m}</td><td class="r">{r}</td><td class="r" style="font-weight:600; color:{ACC}">{v}</td></tr>'
    for k, (g, st, m, r, v) in enumerate(bl)).replace("&quot;", '"')
bl_tbl = (f'<table style="font-size:24px; color:{INK}; font-family:{SANS}">'
          f'<tr style="background:{DARK}"><th style="width:10%; text-align:left">GPUs</th><th style="width:38%; text-align:left">Pipeline: layers per stage</th>'
          f'<th style="width:19%; text-align:right">Megatron</th><th style="width:19%; text-align:right">rdsp</th><th style="width:14%; text-align:right">rdsp faster</th></tr>'
          + bl_rows + '</table>')
bl_css = (f'<style>#balanced td {{ padding:12px 14px; vertical-align:middle }} #balanced th {{ padding:12px 14px; color:#EDEBE4 }} '
          f'#balanced td.r {{ text-align:right }}</style>')
sec("balanced", bl_css + head("Balanced layouts", "With the best layouts, rdsp is 1.18× faster than Megatron") +
    f'''<p style="font-size:24px; color:{MUTED}">Both frameworks use the same layout: fewer layers on the last stage, because it also carries the output head (H). Step time · tokens per second.</p>
{bl_tbl}
<div style="display:flex; gap:32px">
{card("Same machine, same settings", "One pinned-H100 machine, back to back: pretrained weights, same text, 16 microbatches of 8,192 tokens, bf16 gradient summing, Megatron with its faster cross-entropy option.", fs=23, pad=26)}
{card("The fastest setup overall", "rdsp with 5 equal stages: 290.6k tokens per second. The next slide tests whether giving one stage 2 GPUs does better.", fs=23, pad=26)}
</div>''',
    "hetero_20260923-160952. Layer layouts chosen so the last stage (which also runs the output head) gets 1 layer. Megatron: --pipeline-model-parallel-layout Et*9|t*9|t*9|t,L and Et*7|t*7|t*7|t*6|t,L; rdsp: ExplicitCuts 9,18,27 and 7,14,21,27. All four trained to the same loss.", extra="; gap:26px")

# ---------------------------------------------------------------- 17b conclusions
cc = [("It's correct", "Every configuration trains like the unsplit model; rdsp and Megatron produce the same loss curves at 1, 4 and 8 GPUs. Strict tests found and fixed six real bugs."),
      ("It's fast", "Matched setting by setting on one machine, rdsp is on par with Megatron or faster: 1.1–1.2× in GPU-bound training (a tie at 8 GPUs), up to 1.3× when Python overhead dominates."),
      ("Ray costs nothing measurable", "After the stage-local redesign, rdsp's pipeline adds no overhead beyond Megatron's up to 8 stages; Ray only starts and finishes each step."),
      ("Uneven stages: not on this model", "A second GPU for the busiest stage lost to a plain extra stage in all three fair tests. It needs a stage with enough work to split, i.e. a larger model.")]
sec("conclusions", head("Conclusions", "What the project showed") +
    '<div style="display:grid; grid-template-columns:1fr 1fr; gap:28px">' +
    "".join(card(t, x, fs=25, pad=30) for t, x in cc) + "</div>",
    "Correctness: parity rows within 1e-4 (fp32, SGD) and per-parameter update checks; loss curves within 0.05-0.13% median per step. Speed (same pinned-H100 machine): small microbatches 1.24/1.30/1.29x at 1/4/8 GPUs vs Megatron with bf16 summing; large microbatches 1.10/1.11/1.02x vs Megatron with TE cross-entropy; balanced 4/5-stage layouts 1.18x. Uneven stages: 0.93x, 0.91x, 0.97x vs rdsp's own 5-stage pipeline.")

# ---------------------------------------------------------------- 18 limitations
lim_l = ["One run per cell, and identical work varied by up to ~20% between machines: only same-machine comparisons are solid.",
         "Modal ran some 'H100' jobs on H200s; the first headline mixed them and is withdrawn. Runs now refuse non-H100 machines.",
         "One small model (0.6B) on one machine; larger models are untested.",
         "The design doc's 32-GPU, 4-machine topology never ran (Modal allows 10 GPUs at once). Multi-machine code is tested on one machine only.",
         "rdsp's lead is regime-dependent: 1.24–1.30× with small microbatches; GPU-bound it's 1.10×/1.11× and a tie at 8 GPUs."]
lim_r = ["Scaling across GPU counts isn't measured reliably (different machines); beyond 8 stages is unmeasured.",
         "A second GPU for the busy stage didn't pay off on Qwen3-0.6B: 0.91–0.97× vs a plain 5-stage pipeline in three fair tests.",
         "No gradient clipping across the whole pipeline.",
         "Sequence parallel can't be on the last stage; tensor and sequence parallel can't share a stage (DeepSpeed).",
         "Only plain layer stacks and Llama-style HF models split automatically. Checkpoints on local disks survive crashes, not machine loss."]
def ul(items):
    return (f'<ul style="font-size:26px; line-height:1.45; color:{BODY}; display:flex; flex-direction:column; gap:14px">'
            + "".join(f"<li>{i}</li>" for i in items) + "</ul>")
sec("limits", head("Limitations", "What these results do not show") + f'''
<div style="display:flex; gap:56px">
<div style="flex:1">{ul(lim_l)}</div>
<div style="flex:1">{ul(lim_r)}</div>
</div>
''', "Everything on these slides was measured on one pinned-H100 machine per comparison, with GPU type, gradient precision and layer split checked on every GPU. Total GPU spend for the benchmarking was roughly $40 on Modal.", bg=ALT)

# ---------------------------------------------------------------- 19 next
nx = [("Uneven stages on a larger model", "The one setting where the core idea could still pay off: a stage with enough work to split across GPUs."),
      ("Error bars", "Two or three same-machine repeats per cell, to put error bars on the 1.1–1.2× leads."),
      ("The 32-GPU topology", "Needs a higher Modal GPU limit or an AWS cluster; the harness is ready."),
      ("A larger model vs Megatron", "Check whether rdsp's lead holds beyond 0.6B parameters, where Megatron's kernels are tuned.")]
sec("next", head("Future work", "If the project continues") +
    '<div style="display:grid; grid-template-columns:1fr 1fr; gap:28px">' +
    "".join(card(t, x, pad=32) for t, x in nx) + "</div>",
    "Rough cost: repeats are a few dollars per cell on Modal; the larger-model rows and the 32-GPU run are the expensive ones.")

deck = {"v": 4, "createdOnFiles": {"v": 1, "at": "2026-09-23T15:20:00Z"},
        "title": "rdsp: Pipeline Parallelism on Ray + DeepSpeed",
        "order": ["cover", "goal", "built", "arch", "step", "shapes", "versus", "correct", "bugs", "ckpt",
                  "profile", "fix", "setup", "headline", "why", "gpubound", "loss", "scaling", "balanced", "nothelp",
                  "conclusions", "limits", "next", "detail"],
        "sections": {
            "intro": {"description": "What rdsp is for and where it stands", "start": "cover"},
            "design": {"description": "How it works and how it differs from Megatron", "start": "arch"},
            "correct": {"description": "Evidence that the split model trains like the unsplit one", "start": "correct"},
            "perf": {"description": "The profile, the fix, and the benchmark against Megatron", "start": "profile"},
            "close": {"description": "Conclusions, limitations and future work", "start": "conclusions"},
            "appendix": {"description": "Earlier first-design grid", "start": "detail"}},
        "faces": {"ibm-plex-sans": {"family": "IBM Plex Sans", "href": "https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600&display=swap"},
                  "jetbrains-mono": {"family": "JetBrains Mono", "href": "https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400&display=swap"}},
        "designSystems": []}
assert sorted(deck["order"]) == sorted(slides), "order must list every slide exactly once"
json.dump(deck, open(os.path.join(ROOT, "deck.json"), "w"), indent=1)
print(len(slides), slides)
