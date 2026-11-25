# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""Train a Qwen3-VL or Qwen3.5 model with rdsp on image-caption rows.

The driver holds only an empty model skeleton; each stage reads its own
weights from the downloaded checkpoint. Rows are synthetic images of coloured
shapes with their captions, or CORD-v2 receipts with their parse as JSON,
built by the model's own processor; the loss covers the caption tokens only.

    python train_vl.py --model Qwen/Qwen3-VL-2B-Instruct --stages 4 --check
    python train_vl.py --stages 3 --stage 0:gpus=2 --stage 1:recompute=1
    python train_vl.py --model Qwen/Qwen3.5-9B --dataset cord-v2 --seq 2048 \
        --stages 2 --stage 0:gpus=4,tp=2 --stage 1:gpus=4,tp=2 --colocated-vision
    python train_vl.py --model Qwen/Qwen3.5-9B --dataset exported:/data/cord_steps \
        --rows 2 --microbatches 32 --stages 2 --cuts 16 \
        --stage 0:gpus=4,tp=2,zero=1 --stage 1:gpus=4,tp=2,zero=1

`exported:DIR` reads samples written by bench/mimo/export_bridge_batches.py
(Megatron-Bridge's own SFT pipeline), one file per step, in order.
"""

import argparse
import dataclasses
import glob
import itertools
import json
import os
import random
import time

import torch
import torch.nn.functional as F

import ray_deepspeed_pipeline as rdsp
from ray_deepspeed_pipeline.config import MASTER_WEIGHTS_KEY, OPTIMIZER_GROUP_NUMEL_KEY
from train import parse_stage

COLORS = ["red", "green", "blue", "yellow", "purple", "orange", "white", "black"]
SHAPES = ["circle", "square", "triangle"]


def shape_image(size: int, rng: random.Random):
    """(PIL image, caption) of one shape on a plain background."""
    from PIL import Image, ImageDraw

    fg, bg = rng.sample(COLORS, 2)
    shape = rng.choice(SHAPES)
    image = Image.new("RGB", (size, size), bg)
    draw = ImageDraw.Draw(image)
    lo, hi = size // 5, size - size // 5
    if shape == "circle":
        draw.ellipse([lo, lo, hi, hi], fill=fg)
    elif shape == "square":
        draw.rectangle([lo, lo, hi, hi], fill=fg)
    else:
        draw.polygon([(size // 2, lo), (lo, hi), (hi, hi)], fill=fg)
    return image, f"A {fg} {shape} on a {bg} background."


def synthetic_rows(image_size: int, seed: int):
    rng = random.Random(seed)
    while True:
        yield shape_image(image_size, rng)


def cord_rows(max_pixels: int, seed: int):
    """CORD-v2 receipts (train split, shuffled, repeated): the image scaled
    to at most max_pixels, and its ground-truth parse as JSON text."""
    import datasets

    data = datasets.load_dataset("naver-clova-ix/cord-v2", split="train").shuffle(seed=seed)
    for row in itertools.cycle(data):
        image = row["image"].convert("RGB")
        scale = (max_pixels / (image.width * image.height)) ** 0.5
        if scale < 1:
            image = image.resize(
                (max(28, int(image.width * scale)), max(28, int(image.height * scale)))
            )
        caption = json.dumps(json.loads(row["ground_truth"])["gt_parse"], ensure_ascii=False)
        yield image, caption


def encode_row(processor, image, caption: str, seq: int) -> tuple[dict, torch.Tensor]:
    """One row: inputs padded to `seq` tokens, labels on the caption only;
    None if it does not fit."""
    user = {
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": "Describe the image."}],
    }
    answer = {"role": "assistant", "content": [{"type": "text", "text": caption}]}
    full = processor.apply_chat_template([user, answer], tokenize=False)
    prompt = processor.apply_chat_template([user], tokenize=False, add_generation_prompt=True)
    enc = processor(text=[full], images=[image], return_tensors="pt")
    n_prompt = processor(text=[prompt], images=[image], return_tensors="pt")["input_ids"].shape[1]
    ids = enc["input_ids"][0]
    if len(ids) > seq:
        return None
    pad = seq - len(ids)
    labels = ids.clone()
    labels[:n_prompt] = -100
    row = {
        "input_ids": F.pad(ids, (0, pad), value=processor.tokenizer.pad_token_id),
        "attention_mask": F.pad(enc["attention_mask"][0], (0, pad)),
        "mm_token_type_ids": F.pad(enc["mm_token_type_ids"][0], (0, pad)),
        "pixel_values": enc["pixel_values"],
        "image_grid_thw": enc["image_grid_thw"],
    }
    return row, F.pad(labels, (0, pad), value=-100)


def microbatches(processor, rows: int, seq: int, source, pad_multiple: int = 0):
    """Endless (inputs, labels) microbatches from (image, caption) pairs:
    row-shaped values stacked, images given per row (rdsp concatenates each
    rank's rows). Pairs too long for `seq` are skipped. pad_multiple: cut each
    microbatch to its longest row rounded up to this multiple, instead of
    `seq` (the pipeline then sends every boundary with its shape)."""
    encoded_rows = (encode_row(processor, image, caption, seq) for image, caption in source)
    fitting = (row for row in encoded_rows if row is not None)
    while True:
        encoded = [next(fitting) for _ in range(rows)]
        length = _padded_length([r for r, _ in encoded], pad_multiple, seq)
        inputs = {
            k: torch.stack([r[k][:length] for r, _ in encoded])
            for k in ("input_ids", "attention_mask", "mm_token_type_ids")
        }
        inputs["pixel_values"] = [r["pixel_values"] for r, _ in encoded]
        inputs["image_grid_thw"] = [r["image_grid_thw"] for r, _ in encoded]
        yield inputs, torch.stack([labels[:length] for _, labels in encoded])


_EXPORTED_KEYS = ("input_ids", "attention_mask", "mm_token_type_ids")


def exported_microbatches(
    directory: str, rows: int, pad_multiple: int = 0, pad_per_microbatch: bool = False
):
    """(inputs, labels) microbatches of `rows` consecutive samples from an
    exported directory, step file by step file: the order and grouping
    Megatron's sequential sampler gives a global microbatch. pad_multiple:
    cut each step's padding to its longest row rounded up to this multiple
    (a step's microbatches share one length). pad_per_microbatch: cut each
    microbatch to its own longest row instead, as Megatron's collate does
    (the pipeline then sends every boundary with its shape)."""
    for path in sorted(glob.glob(os.path.join(directory, "step_*.pt"))):
        samples = torch.load(path, weights_only=False)
        if len(samples) % rows:
            raise ValueError(
                f"{path}: {len(samples)} samples do not split into microbatches of {rows} rows"
            )
        exported = samples[0]["input_ids"].shape[1]
        length = _padded_length(samples, pad_multiple, exported)
        for start in range(0, len(samples), rows):
            group = samples[start : start + rows]
            if pad_per_microbatch:
                length = _padded_length(group, pad_multiple, exported)
            inputs = {k: torch.cat([s[k][:, :length] for s in group]) for k in _EXPORTED_KEYS}
            inputs["pixel_values"] = [s["pixel_values"] for s in group]
            inputs["image_grid_thw"] = [s["image_grid_thw"] for s in group]
            yield inputs, torch.cat([s["labels"][:, :length] for s in group])


def drop_padding_mask(inputs: dict) -> dict:
    """inputs without their attention mask, for rows padded only at the end:
    causal attention already keeps real tokens off later (padding) positions,
    and without a mask attention runs its causal flash kernel. Raises
    ValueError when a row has padding before a real token."""
    mask = inputs["attention_mask"]
    lengths = mask.sum(dim=1, keepdim=True)
    if not torch.equal(mask.bool(), torch.arange(mask.shape[1]) < lengths):
        raise ValueError("--drop-padding-mask needs rows padded on the right only")
    return {k: v for k, v in inputs.items() if k != "attention_mask"}


def _padded_length(samples, pad_multiple: int, exported: int) -> int:
    """The samples' longest real row rounded up to pad_multiple, at most the
    exported length; the exported length when pad_multiple is 0."""
    if not pad_multiple:
        return exported
    longest = max(int(s["attention_mask"].sum()) for s in samples)
    return min(exported, -(-longest // pad_multiple) * pad_multiple)


# summed next-token loss over the caption tokens of some rows
token_loss_sum = rdsp.next_token_loss_sum


def token_loss_sum_liger(logits, labels):
    """token_loss_sum with Liger's cross entropy: computed in fp32 chunk by
    chunk from bf16 logits, gradient written in place, so no fp32 copy of
    the (vocabulary-wide) logits."""
    from liger_kernel.transformers import LigerCrossEntropyLoss

    return LigerCrossEntropyLoss(ignore_index=-100, reduction="sum")(
        logits[:, :-1].reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1).to(logits.device)
    )


def apply_liger(model):
    """Liger's fused RMSNorm and SwiGLU kernels, patched into this model's
    module instances: the patched forwards travel with the stage modules to
    their workers (a class-level patch would stay in this process)."""
    from liger_kernel.transformers import _apply_liger_kernel_to_instance

    _apply_liger_kernel_to_instance(
        model=model,
        rms_norm=True,
        swiglu=True,
        rope=False,
        cross_entropy=False,
        fused_linear_cross_entropy=False,
    )


def restore_hf_experts(model) -> int:
    """Undo Liger's fused mixture-of-experts kernel (LigerExperts) on the routed experts, so
    they run Hugging Face's own implementation (grouped matmuls on GPU) again. Liger binds
    its forward onto each module instance; removing that binding restores the class's.
    Returns how many modules were restored."""
    restored = 0
    for module in model.modules():
        name = module.__dict__.get("_get_name")
        if "forward" in module.__dict__ and name is not None and name() == "LigerExperts":
            del module.__dict__["forward"], module.__dict__["_get_name"]
            restored += 1
    return restored


def token_count(labels) -> int:
    return int((labels[:, 1:] != -100).sum())


# what training averages over; either way the reported step loss is the
# mean over every caption token of the step, as Megatron logs it
LOSSES = {
    "token-mean": False,  # every token of the step (calculate_per_token_loss)
    "microbatch-mean": True,  # each rank's microbatch, then the mean of those
    # (Megatron's default)
}


def partition_policy(cuts: str, ratio):
    """--cuts as a partition policy. ratio() measures the vision encoder's cost
    per text token from the first step; only 'balanced' (and 'measured',
    whose calibration runs with the balanced cut) need it."""
    if cuts in ("balanced", "measured"):
        return rdsp.BalancedTransformerBlocks(vision_token_ratio=ratio())
    if cuts in ("0,balanced", "0,measured"):  # encoder alone on stage 0, decoder over the rest
        return rdsp.BalancedTransformerBlocks(vision_stage=True)
    if cuts == "even":
        return rdsp.UniformTransformerBlocks()
    return rdsp.ExplicitCuts(tuple(int(c) for c in cuts.split(",")))


def calibrated_policy(args, engine, data, estimate):
    """--cuts measured: profile --calibration-steps steps of `engine` (built
    with the balanced cut and a learning rate of 0, so the weights stay as
    loaded), shut it down, and return the balanced policy re-cut by what the
    stages measured (BalancedTransformerBlocks.measured). The steps read
    their own copy of the run's first batches."""
    steps = args.calibration_steps

    def batches():
        for _ in range(steps + 1):  # one more: prefetch reads a step ahead
            entries = [next(data) for _ in range(args.microbatches)]
            if args.drop_padding_mask:
                entries = [(drop_padding_mask(x), y) for x, y in entries]
            yield from entries

    try:
        measured = engine.measure_stages(batches(), steps=steps, skip=1)
    finally:
        engine.shutdown()
    print(f"measured: {measured}", flush=True)
    return dataclasses.replace(estimate, measured=measured)


def loss_function(args):
    """The stage loss for --loss / --liger / --sharded-loss."""
    if args.sharded_loss:
        token_sum = rdsp.next_token_loss_sum
    elif args.liger:
        token_sum = token_loss_sum_liger
    else:
        token_sum = token_loss_sum
    return rdsp.TokenMeanLoss(token_sum, token_count, per_microbatch=LOSSES[args.loss])


def unsplit_loss(weights: str, batches) -> float:
    """Token-mean loss of the whole model, unsplit: the reference for --check.
    Spread over the GPUs when it does not fit one (32B); freed afterwards."""
    import transformers

    model = transformers.AutoModelForImageTextToText.from_pretrained(
        weights, dtype=torch.bfloat16, device_map="auto"
    ).eval()
    sums, counts = [], []
    with torch.no_grad():
        for inputs, labels in batches:
            full = {k: torch.cat(v) if isinstance(v, list) else v for k, v in inputs.items()}
            full = {k: v.to(model.device) for k, v in full.items()}
            sums.append(float(token_loss_sum(model(**full, use_cache=False).logits, labels)))
            counts.append(max(token_count(labels), 1))
    del model
    torch.cuda.empty_cache()
    return sum(sums) / sum(counts)


def ds_config(args, lr: float | None = None) -> dict:
    offload = any(o.offload_optimizer for o in args.stage)
    beta1, beta2 = (float(b) for b in args.betas.split(","))
    conf = {
        "train_batch_size": args.rows * args.microbatches,
        "gradient_accumulation_steps": args.microbatches,
        "bf16": {
            "enabled": True,
            **({"immediate_grad_update": False} if args.keep_bf16_grads else {}),
        },
        # ZeRO's default 500M-element communication buckets cost ~2 GB each in
        # fp32; smaller buckets change how gradients are batched, not the math
        "zero_optimization": {
            "stage": args.zero,
            "reduce_bucket_size": int(5e7),
            "allgather_bucket_size": int(5e7),
        },
        "gradient_clipping": 0.0,
        # DeepSpeed's CPU Adam under optimizer offload, PyTorch's AdamW otherwise
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.lr if lr is None else lr,
                "betas": [beta1, beta2],
                "eps": args.eps,
                "weight_decay": args.weight_decay,
                "torch_adam": not offload,
                # one kernel per step; the default foreach AdamW also allocates a
                # temporary as large as the fp32 master weights
                **({"fused": True} if args.fused_adam else {}),
            },
        },
        "steps_per_print": 10**9,
    }
    if args.grad_dtype == "fp32":
        # accumulate and all-reduce gradients in fp32 (Megatron's
        # grad_reduce_in_fp32 with fp32 main grads)
        conf["data_types"] = {"grad_accum_dtype": "fp32"}
        conf["communication_data_type"] = "fp32"
    if args.optimizer_group_numel:
        conf[OPTIMIZER_GROUP_NUMEL_KEY] = args.optimizer_group_numel
    if args.master_dtype == "bf16":
        # DeepSpeed's low-precision model states: the ZeRO optimizer keeps its
        # master weights, gradients and Adam's two moments in bf16, about 10-12
        # bytes per parameter instead of 18
        conf["bf16"].update(bf16_master_weights_and_grads=True, bf16_optimizer_states=True)
    if args.master_dtype == "remainder":
        # exact fp32 master weights at 2 bytes per parameter: the bf16 weights
        # plus a 16-bit remainder (the stage adapter swaps in RemainderAdamW)
        conf[MASTER_WEIGHTS_KEY] = "remainder"
    return conf


# what rdsp's actors read from the environment: rdsp's own switches, NCCL and its network
# plugin, the Hugging Face cache, the CUDA allocator, DeepSpeed's op builds
_WORKER_ENV_PREFIXES = ("RDSP_", "NCCL_", "FI_", "HF_", "PYTORCH_", "TORCH_", "DS_")
_WORKER_ENV_NAMES = ("CUDA_HOME", "LD_LIBRARY_PATH", "TRANSFORMERS_OFFLINE")
# never passed: Ray keeps a job's runtime_env with the job, readable by whoever can see the cluster
_CREDENTIAL_MARKS = ("TOKEN", "KEY", "SECRET", "PASSWORD")


def worker_env(environ=None) -> dict:
    """The driver's settings rdsp's actors read, for Ray to give every worker of this job.
    A cluster started with `ray start` (two nodes) runs workers in that command's
    environment, not the driver's; a cluster ray.init() starts here already has it. Ray
    sets CUDA_VISIBLE_DEVICES per actor itself, so it is never passed; nor is anything
    named like a credential (HF_TOKEN): the workers read weights the driver downloaded."""
    environ = os.environ if environ is None else environ
    return {
        k: v
        for k, v in environ.items()
        if (k.startswith(_WORKER_ENV_PREFIXES) or k in _WORKER_ENV_NAMES)
        and not any(mark in k for mark in _CREDENTIAL_MARKS)
    }


def optimizer_problem(args) -> str | None:
    """Why --master-dtype bf16/remainder or --fused-adam cannot apply as asked,
    or None. DeepSpeed reads the bf16 states only in its ZeRO-1/2 optimizer,
    which it picks for bf16 gradient accumulation; elsewhere the keys are
    ignored and the states stay fp32 without a word. The remainder master
    lives in that same optimizer. Under offload the optimizer is DeepSpeed's
    CPU Adam, which has no fused option."""
    overrides = {o.stage: o for o in args.stage}
    offloaded = [s for s, o in sorted(overrides.items()) if o.offload_optimizer]
    if args.fused_adam and offloaded:
        return f"--fused-adam is PyTorch's AdamW; stage {offloaded[0]} offloads to CPU Adam"
    if args.master_dtype == "fp32":
        return None
    flag = f"--master-dtype {args.master_dtype}"
    if args.grad_dtype != "bf16":
        return f"{flag} needs --grad-dtype bf16"
    if args.colocated_vision:
        return f"{flag}: the replicated encoder keeps fp32 states"
    for s in range(args.stages):
        o = overrides.get(s)
        zero = o.zero_stage if o is not None and o.zero_stage is not None else args.zero
        if zero not in (1, 2):
            return f"{flag} needs ZeRO 1 or 2 on every stage; stage {s} has {zero}"
        if o is not None and o.offload_optimizer:
            return f"{flag} is not supported with offload (stage {s})"
    return None


def experts_problem(args) -> str | None:
    """Why --moe-experts hf cannot apply as asked, or None: it undoes Liger's
    expert kernel, which only --liger installs."""
    if args.moe_experts == "hf" and not args.liger:
        return (
            "--moe-experts hf needs --liger: without it the experts already run "
            "Hugging Face's implementation"
        )
    return None


def _at_least(minimum: int):
    """argparse type: a whole number >= minimum."""

    def parse(text: str) -> int:
        value = int(text)
        if value < minimum:
            raise argparse.ArgumentTypeError(f"must be at least {minimum}, got {value}")
        return value

    return parse


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    p.add_argument("--stages", type=int, default=2)
    p.add_argument(
        "--cuts",
        default="balanced",
        help="'balanced', 'measured' (balanced, then re-cut by the stage "
        "times and memory --calibration-steps profiled steps measure), "
        "'0,balanced' (vision encoder alone on stage 0), '0,measured' (that, "
        "the decoder stages then re-cut as measured does), 'even', or e.g. 9,18",
    )
    p.add_argument(
        "--calibration-steps",
        type=_at_least(2),
        default=3,
        help="--cuts measured: steps profiled at learning rate 0, the first not counted",
    )
    p.add_argument(
        "--stage",
        type=parse_stage,
        action="append",
        default=[],
        help=(
            "per-stage layout, as in train.py; recompute_blocks=N recomputes the "
            "stage's first N decoder layers, and the first stage, which holds the "
            "vision encoder, always recomputes the encoder (even at N=0)"
        ),
    )
    p.add_argument("--microbatches", type=int, default=8)
    p.add_argument("--rows", type=int, default=4, help="rows per microbatch")
    p.add_argument("--seq", type=int, default=256)
    p.add_argument("--image-size", type=int, default=448, help="synthetic images' side")
    p.add_argument(
        "--dataset",
        default="synthetic",
        help="synthetic, cord-v2, or exported:DIR (export_bridge_batches.py)",
    )
    p.add_argument(
        "--loss",
        choices=tuple(LOSSES),
        default="token-mean",
        help="what training averages over; the step loss printed is the token mean either way",
    )
    p.add_argument(
        "--liger", action="store_true", help="Liger fused RMSNorm/SwiGLU kernels and cross entropy"
    )
    p.add_argument(
        "--moe-experts",
        choices=("liger", "hf"),
        default="liger",
        help="mixture-of-experts models under --liger: the routed experts in Liger's "
        "fused kernel (liger) or Hugging Face's grouped matmuls (hf)",
    )
    p.add_argument(
        "--pad-multiple",
        type=int,
        default=0,
        help="exported data: trim each step's padding to its longest row, "
        "rounded up to this multiple (0 keeps the exported length); with "
        "--pad-per-microbatch, also applies to cord-v2 and synthetic rows",
    )
    p.add_argument("--betas", default="0.9,0.999", help="AdamW betas")
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument(
        "--grad-dtype",
        choices=("bf16", "fp32"),
        default="fp32",
        help="gradient accumulation and reduction dtype",
    )
    p.add_argument(
        "--master-dtype",
        choices=("bf16", "fp32", "remainder"),
        default="fp32",
        help="dtype of the optimizer's master weights and Adam's moments; "
        "remainder: fp32 master weights stored as the bf16 weights plus a "
        "16-bit remainder (2 bytes per parameter instead of 4, same math), "
        "fp32 moments; bf16 and remainder need --grad-dtype bf16 and ZeRO 1 "
        "or 2 on every stage",
    )
    p.add_argument(
        "--optimizer-group-numel",
        type=_at_least(0),
        default=0,
        help="split each stage's parameters into optimizer groups of at most "
        "this many elements (0: one group); DeepSpeed's ZeRO step then "
        "copies one group's gradients at a time: less peak memory, same "
        "math; stages with ep= keep DeepSpeed's own groups",
    )
    p.add_argument(
        "--fused-adam",
        action="store_true",
        help="PyTorch's fused AdamW instead of its default foreach AdamW, "
        "which needs a temporary as large as the fp32 master weights",
    )
    p.add_argument(
        "--attn", default="sdpa", help="attention implementation (sdpa, flash_attention_2, ...)"
    )
    p.add_argument(
        "--vision-attn",
        choices=("same", "varlen"),
        default="same",
        help="vision encoder attention: same as --attn, or varlen: PyTorch's "
        "variable-length flash attention, one kernel per layer for all "
        "images and no host sync per layer (CUDA)",
    )
    p.add_argument(
        "--max-pixels",
        type=int,
        default=512 * 512,
        help="cord-v2: scale images down to at most this many pixels",
    )
    p.add_argument(
        "--untie-embeddings",
        action="store_true",
        help="train the input embedding and the output head as two matrices; "
        "needed when a tied pair would land on different stages",
    )
    p.add_argument(
        "--keep-bf16-grads",
        action="store_true",
        help="keep DeepSpeed's per-parameter bf16 gradients (accumulated into "
        "fp32 after each backward) instead of freeing each one as it lands",
    )
    p.add_argument(
        "--pad-per-microbatch",
        action="store_true",
        help="with --pad-multiple: pad each microbatch to its own longest row "
        "(as Megatron does) instead of each step to its longest",
    )
    p.add_argument(
        "--drop-padding-mask",
        action="store_true",
        help="leave out the attention mask of right-padded rows, so attention "
        "can use its causal flash kernel",
    )
    p.add_argument(
        "--sharded-loss",
        action="store_true",
        help="token-mean loss over vocab-parallel logits: tensor-parallel "
        "stages skip gathering the full logits",
    )
    p.add_argument(
        "--prefetch",
        action="store_true",
        help="read and ship the next step's batches while this step runs",
    )
    p.add_argument(
        "--colocated-vision",
        action="store_true",
        help="run the vision encoder on every GPU (rdsp.ColocatedVision)",
    )
    p.add_argument(
        "--vision-per-microbatch",
        action="store_true",
        help="with --colocated-vision: encode and backpropagate images per "
        "microbatch inside the pipeline (less memory, can be slower)",
    )
    p.add_argument(
        "--vision-compile",
        action="store_true",
        help="with --colocated-vision: torch.compile the encoder's blocks",
    )
    p.add_argument(
        "--vision-recompute",
        action="store_true",
        help="with --colocated-vision: recompute the encoder's blocks",
    )
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--zero", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--check",
        action="store_true",
        help="first compare the pipeline's loss with the unsplit model on one GPU",
    )
    p.add_argument(
        "--profile",
        action="store_true",
        help="print per-stage phase times (RDSP_PROFILE=1; slows steps)",
    )
    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if problem := optimizer_problem(args) or experts_problem(args):
        parser.error(problem)

    import accelerate
    import huggingface_hub
    import ray
    import transformers

    weights = huggingface_hub.snapshot_download(
        args.model, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"]
    )
    processor = transformers.AutoProcessor.from_pretrained(weights)

    def new_data():
        """The run's microbatches from the start (a calibration reads its own)."""
        if args.dataset.startswith("exported:"):
            return exported_microbatches(
                args.dataset.split(":", 1)[1], args.rows, args.pad_multiple, args.pad_per_microbatch
            )
        source = (
            cord_rows(args.max_pixels, args.seed)
            if args.dataset == "cord-v2"
            else synthetic_rows(args.image_size, args.seed)
        )
        return microbatches(
            processor,
            args.rows,
            args.seq,
            source,
            args.pad_multiple if args.pad_per_microbatch else 0,
        )

    data = new_data()
    first = [next(data) for _ in range(args.microbatches)]

    reference = unsplit_loss(weights, first) if args.check else None

    config = transformers.AutoConfig.from_pretrained(weights)
    if args.untie_embeddings:  # tied parameters must sit on one stage
        config.tie_word_embeddings = config.get_text_config().tie_word_embeddings = False
    config._attn_implementation = args.attn
    with accelerate.init_empty_weights():
        skeleton = transformers.AutoModelForImageTextToText.from_config(
            config, dtype=torch.bfloat16
        )
    if args.vision_attn == "varlen":
        from ray_deepspeed_pipeline.varlen_attention import use_varlen_vision_attention

        use_varlen_vision_attention(skeleton)
    from ray_deepspeed_pipeline.partition import vision_token_ratio

    partition = partition_policy(args.cuts, ratio=lambda: vision_token_ratio(config, first))

    if args.profile:
        os.environ["RDSP_PROFILE"] = "1"  # before ray.init: actors inherit it
    loss_fn = loss_function(args)
    if args.liger:
        apply_liger(skeleton)
        if args.moe_experts == "hf":
            restore_hf_experts(skeleton)
    ray.init(ignore_reinit_error=True, runtime_env={"env_vars": worker_env()})

    def build(partition, conf):
        pipeline_config = rdsp.PipelineConfig(
            stages=args.stages,
            partition=partition,
            stage_overrides=tuple(args.stage),
            colocated_vision=(
                rdsp.ColocatedVision(
                    recompute=args.vision_recompute,
                    compile=args.vision_compile,
                    encode_per_microbatch=args.vision_per_microbatch,
                )
                if args.colocated_vision
                else None
            ),
            prefetch=args.prefetch,
        )
        engine, _, _, _ = rdsp.initialize(
            model=skeleton,
            config=conf,
            loss_fn=loss_fn,
            weights=weights,
            pipeline_config=pipeline_config,
        )
        print(f"stages (blocks): {engine.stage_blocks}", flush=True)
        return engine

    if args.cuts in ("measured", "0,measured"):
        partition = calibrated_policy(
            args, build(partition, ds_config(args, lr=0.0)), new_data(), partition
        )
    engine = build(partition, ds_config(args))

    if reference is not None:
        piped = float(engine.eval_batch(data_iter=iter(first)))
        print(
            f"check: pipeline {piped:.5f}  unsplit {reference:.5f}  "
            f"rel diff {abs(piped - reference) / reference:.2e}",
            flush=True,
        )

    padded = args.rows * args.seq * args.microbatches
    counts = {}  # step -> (real, supervised), filled as its batches are read

    def stream():
        """Every step's batches through one iterator (prefetch reads ahead)."""
        for step in range(args.steps):
            entries = first if step == 0 else [next(data) for _ in range(args.microbatches)]
            # real tokens: what the attention mask covers (padding excluded)
            counts[step] = (
                sum(int(x["attention_mask"].sum()) for x, _ in entries),
                sum(token_count(y) for _, y in entries),
            )
            if args.drop_padding_mask:
                entries = [(drop_padding_mask(x), y) for x, y in entries]
            yield from entries

    batches = stream()
    for step in range(args.steps):
        start = time.perf_counter()
        loss = float(engine.train_batch(data_iter=batches))
        real, supervised = counts.pop(step)
        ms = (time.perf_counter() - start) * 1e3
        print(
            f"step {step} loss {loss:.4f} {ms:.0f} ms {real / ms * 1e3:.0f} real tok/s "
            f"{padded / ms * 1e3:.0f} padded tok/s real {real} supervised {supervised}",
            flush=True,
        )
    return engine


if __name__ == "__main__":
    main()
