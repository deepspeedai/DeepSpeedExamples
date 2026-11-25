# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""Deterministic contiguous model partitioning and stage-module builders.

Policies: explicit cuts, uniform transformer blocks, uniform sequential, and
cost-balanced blocks (parameter counts as the cost estimate). Stages are
described by fully qualified parameter names, never driver parameter objects.
"""

import functools
from dataclasses import dataclass

import torch.nn as nn

from ray_deepspeed_pipeline.config import (
    BalancedTransformerBlocks,
    ExplicitCuts,
    UniformSequential,
    UniformTransformerBlocks,
)
from ray_deepspeed_pipeline.errors import ValidationError


@dataclass(frozen=True)
class StagePartition:
    index: int
    parameter_names: tuple[str, ...]
    block_start: int
    block_stop: int


def vision_config(model):
    """The model's vision encoder config (HF `config.vision_config`), or None."""
    return getattr(getattr(model, "config", None), "vision_config", None)


def find_block_list(model: nn.Module) -> tuple[str, nn.ModuleList]:
    """(name, module) of the longest ModuleList whose children share one
    class. Raises ValidationError if there is none, or a tie."""
    candidates = [
        (name, m)
        for name, m in model.named_modules()
        if isinstance(m, nn.ModuleList) and len(m) >= 2 and len({type(c) for c in m}) == 1
    ]
    # a vision-language model's pipeline cuts its decoder, never the encoder
    # (which may be as deep: Qwen3.5-2B has 24 of each)
    encoder = _vision_encoder_path(model)
    if encoder is not None:
        candidates = [(n, m) for n, m in candidates if not n.startswith(encoder + ".")]
    if not candidates:
        raise ValidationError(
            "no transformer block list found: expected a ModuleList of >=2 "
            "same-class blocks; use ExplicitCuts on a model this heuristic "
            "cannot read"
        )
    longest = max(len(m) for _, m in candidates)
    winners = [(n, m) for n, m in candidates if len(m) == longest]
    if len(winners) > 1:
        raise ValidationError(
            f"ambiguous block lists {[n for n, _ in winners]}: partitioning refuses to guess"
        )
    return winners[0]


def _vision_encoder_path(model: nn.Module) -> str | None:
    if vision_config(model) is None:
        return None
    from ray_deepspeed_pipeline.vision import find_vision_encoder

    try:
        return find_vision_encoder(model)
    except ValidationError:
        return None


def _cuts_for(policy, n_blocks: int, stages: int, vision: bool = False) -> list[int]:
    """vision: the model has a vision encoder, which a first cut at 0 leaves
    alone on the first stage (with the embeddings)."""
    if isinstance(policy, ExplicitCuts):
        cuts = list(policy.cuts)
        if len(cuts) != stages - 1:
            raise ValidationError(
                f"ExplicitCuts needs {stages - 1} cuts for {stages} stages, got {len(cuts)}"
            )
    elif isinstance(policy, (UniformTransformerBlocks, UniformSequential)):
        base, extra = divmod(n_blocks, stages)
        sizes = [base + (1 if i < extra else 0) for i in range(stages)]
        cuts, acc = [], 0
        for size in sizes[:-1]:
            acc += size
            cuts.append(acc)
    else:
        raise ValidationError(f"unknown partition policy {type(policy)}")

    if cuts and cuts[0] == 0 and not vision:
        raise ValidationError(
            "a first cut at 0 leaves the first stage no blocks, which only a model "
            "with a vision encoder can use (a vision-only stage)"
        )
    if any(c < 0 or c >= n_blocks for c in cuts) or 0 in cuts[1:]:
        raise ValidationError(f"cuts must lie in (0, {n_blocks}), got {cuts}")
    if any(a >= b for a, b in zip(cuts, cuts[1:])):
        raise ValidationError(f"cuts must be strictly increasing, got {cuts}")
    return cuts


def vision_injection_depth(model: nn.Module) -> int:
    """Number of leading blocks that receive vision features straight from
    the vision encoder (Qwen3-VL's deepstack), which must share the first
    stage with it; 0 for other models."""
    vision = vision_config(model)
    return len(getattr(vision, "deepstack_visual_indexes", None) or ())


def _compute_costs(
    model: nn.Module, blocks_name: str, blocks, exclude: str | None = None
) -> tuple[int, list[int], int]:
    """(before the blocks, per block, after the blocks) cost estimates in
    parameters, embeddings and the `exclude` module excluded, each shared
    parameter counted once."""
    named = [n for n, _ in model.named_parameters()]
    prefix = blocks_name + "."
    first = min(i for i, n in enumerate(named) if n.startswith(prefix))
    position = {n: i for i, n in enumerate(named)}
    pre = post = 0
    per_block = [0] * len(blocks)
    seen = set()
    for module_name, module in model.named_modules():
        if isinstance(module, nn.Embedding):
            continue
        for leaf, param in module.named_parameters(recurse=False):
            name = f"{module_name}.{leaf}" if module_name else leaf
            if id(param) in seen or name not in position or _under(name, exclude):
                continue
            seen.add(id(param))
            if name.startswith(prefix):
                per_block[int(name[len(prefix) :].split(".", 1)[0])] += param.numel()
            elif position[name] < first:
                pre += param.numel()
            else:
                post += param.numel()
    return pre, per_block, post


def vision_token_ratio(hf_config, sample) -> float:
    """The vision encoder's cost per text token, for
    BalancedTransformerBlocks(vision_token_ratio=...), measured from a step's
    batch. sample: [(inputs, labels)], inputs with input_ids and
    image_grid_thw (one grid per row; empty for a row without an image).

    Text tokens are those actually computed (each microbatch at its own
    length, not a configured maximum). Each image's patches are weighted up
    for the encoder's attention over them: per patch, attention costs about
    4 x patches x width multiply-adds against 2 x a layer's parameters for
    its matrix products, so its share grows with the patches per image."""
    vision = getattr(hf_config, "vision_config", None)
    if vision is None:  # no encoder: the ratio does not matter
        return 1.0
    width = vision.hidden_size
    layer = 4 * width * width + 2 * width * vision.intermediate_size
    tokens = weighted = 0
    for inputs, _ in sample:
        tokens += inputs["input_ids"].numel()
        for grid in inputs.get("image_grid_thw") or []:
            for image in grid.reshape(-1, 3):
                patches = int(image.prod())
                weighted += patches * (1 + 2 * patches * width / layer)
    return weighted / max(tokens, 1)


def _balanced_cuts(
    costs: tuple, stages: int, min_first: int, stage_gpus: tuple[int, ...], extra=None, fits=None
) -> list[int]:
    """Contiguous split minimising the most expensive stage, a stage's cost
    divided by its GPU count; then, among splits that achieve it, the sum of
    squared stage costs (so no stage is left nearly idle). Ties go to splits
    whose earlier stages take more. extra: a fixed cost per stage, added to
    its blocks'. fits(s, j, i): whether stage s may hold blocks j..i-1."""
    pre, per_block, post = costs
    n = len(per_block)
    if n - (stages - 1) < max(min_first, 1):
        raise ValidationError(
            f"{n} blocks cannot fill {stages} stages with at least {min_first} on the first"
        )
    total = [0]
    for c in per_block:
        total.append(total[-1] + c)

    def stage_cost(s, j, i):
        return (
            total[i]
            - total[j]
            + (pre if s == 0 else 0)
            + (post if s == stages - 1 else 0)
            + (extra[s] if extra else 0)
        ) / stage_gpus[s]

    def allowed(s, j, i, limit):
        return stage_cost(s, j, i) <= limit and (fits is None or fits(s, j, i))

    def solve(score, limit):
        # best[s][i]: score of blocks[:i] on stages 0..s; back: where stage s starts
        inf = float("inf")
        best = [[inf] * (n + 1) for _ in range(stages)]
        back = [[0] * (n + 1) for _ in range(stages)]
        for i in range(max(min_first, 1), n + 1):
            if allowed(0, 0, i, limit):
                best[0][i] = score(0, stage_cost(0, 0, i))
        for s in range(1, stages):
            for i in range(s + 1, n + 1):
                for j in range(i - 1, s - 1, -1):
                    if best[s - 1][j] == inf or not allowed(s, j, i, limit):
                        continue
                    value = score(best[s - 1][j], stage_cost(s, j, i))
                    if value < best[s][i]:
                        best[s][i], back[s][i] = value, j
        return best[stages - 1][n], back

    slowest, _ = solve(max, float("inf"))
    if slowest == float("inf"):
        raise ValidationError(
            f"no cut of {n} blocks into {stages} stages fits every "
            f"stage's memory; give the stages more GPUs, recompute "
            f"their blocks, or use more stages"
        )
    _, back = solve(lambda acc, c: acc + c * c, slowest)
    cuts, i = [], n
    for s in range(stages - 1, 0, -1):
        i = back[s][i]
        cuts.append(i)
    return cuts[::-1]


def measured_extras(
    per_block: list, post: float, stage_gpus: tuple[int, ...], measured
) -> list[float]:
    """Each stage's measured time beyond its blocks', in the blocks' cost
    units (parameters): BalancedTransformerBlocks.measured. post: the cost of
    what follows the last block (head and loss)."""
    n = len(measured.seconds)
    ranges = measured.block_ranges(len(per_block))
    if any(start >= stop for start, stop in ranges) or n != len(stage_gpus):
        raise ValidationError(
            f"measured cuts {measured.cuts} do not fit {len(per_block)} blocks on "
            f"{len(stage_gpus)} stages; measure with this model and this number of stages"
        )
    blocks = [sum(per_block[start:stop]) for start, stop in ranges]
    middle = range(1, n - 1)
    if middle:  # stages holding blocks only
        rate = sum(measured.seconds[s] for s in middle) / sum(
            blocks[s] / stage_gpus[s] for s in middle
        )
    else:  # the last stage, its head and loss at their parameter-count estimate
        rate = measured.seconds[-1] / ((blocks[-1] + post) / stage_gpus[-1])
    return [max(0.0, measured.seconds[s] * stage_gpus[s] / rate - blocks[s]) for s in range(n)]


def measured_cuts(
    per_block: list, post: float, stage_gpus: tuple[int, ...], min_first: int, measured
) -> list[int]:
    """The balanced cut with measured fixed costs per stage, and memory: a
    stage gaining blocks gains their weights, gradients and optimizer state.
    The activations of gained blocks are not counted: the reserve in
    MeasuredStages.limit_gib has to cover them."""
    extra = measured_extras(per_block, post, stage_gpus, measured)  # checks the measured cuts
    total = [0]
    for c in per_block:
        total.append(total[-1] + c)
    held = [total[stop] - total[start] for start, stop in measured.block_ranges(len(per_block))]

    def fits(s, j, i):
        gained = total[i] - total[j] - held[s]
        return gained <= 0 or (
            measured.peak_gib[s] + gained * measured.bytes_per_param[s] / 2**30
            <= measured.limit_gib
        )

    return _balanced_cuts(
        (0, per_block, 0), len(stage_gpus), min_first, stage_gpus, extra=extra, fits=fits
    )


def _decoder_stages(measured):
    """A vision-only first stage's measurements left out: the decoder stages' own, for
    re-cutting the decoder alone. The vision stage keeps its GPUs and its cut at 0."""
    import dataclasses

    if not measured.cuts or measured.cuts[0] != 0:
        raise ValidationError(
            "vision_stage measurements must come from a run with the vision "
            f"encoder alone on stage 0 (first cut at 0), got cuts {measured.cuts}"
        )
    return dataclasses.replace(
        measured,
        cuts=measured.cuts[1:],
        seconds=measured.seconds[1:],
        peak_gib=measured.peak_gib[1:],
        bytes_per_param=measured.bytes_per_param[1:],
    )


def _under(name: str, module: str | None) -> bool:
    return module is not None and name.startswith(module + ".")


def partition_parameters(
    model: nn.Module,
    policy,
    stages: int,
    stage_gpus: tuple[int, ...] | None = None,
    exclude: str | None = None,
) -> tuple[StagePartition, ...]:
    """Assign every parameter name to exactly one stage. Blocks map by cut
    range; parameters before the block list go to the first stage, those
    after it to the last. Rejects parameters tied across stages: each stage
    would hold its own copy and see only part of the gradient. exclude: a
    module whose parameters go to no stage (a colocated vision encoder)."""
    if stages < 1:
        raise ValidationError(f"stages must be >= 1, got {stages}")
    blocks_name, blocks = find_block_list(model)
    if isinstance(policy, BalancedTransformerBlocks) and stages > 1:
        pre, per_block, post = _compute_costs(model, blocks_name, blocks, exclude)
        gpus = stage_gpus or (1,) * stages
        if policy.vision_stage:
            if vision_config(model) is None or exclude is not None or vision_injection_depth(model):
                raise ValidationError(
                    "vision_stage puts the vision encoder alone on the first stage: it needs "
                    "a model with a vision encoder whose output enters only the input "
                    "embeddings, not colocated vision"
                )
            if policy.measured is not None:
                cuts = [0] + measured_cuts(
                    per_block, post, gpus[1:], 0, _decoder_stages(policy.measured)
                )
            else:
                cuts = [0] + _balanced_cuts((0, per_block, post), stages - 1, 0, gpus[1:])
        elif policy.measured is not None:
            cuts = measured_cuts(
                per_block, post, gpus, vision_injection_depth(model), policy.measured
            )
        else:
            # what precedes the blocks, embeddings aside, is the vision encoder
            pre *= policy.vision_token_ratio
            cuts = _balanced_cuts(
                (pre, per_block, post), stages, vision_injection_depth(model), gpus
            )
    else:
        vision = vision_config(model) is not None
        cuts = _cuts_for(policy, len(blocks), stages, vision) if stages > 1 else []
    bounds = [0] + cuts + [len(blocks)]

    def stage_of_block(b: int) -> int:
        for s in range(stages):
            if bounds[s] <= b < bounds[s + 1]:
                return s
        raise AssertionError(b)

    # declaration order stands in for topology on the sequential models
    # supported; remove_duplicate=False keeps tied names visible
    named = [
        (n, p) for n, p in model.named_parameters(remove_duplicate=False) if not _under(n, exclude)
    ]
    prefix = blocks_name + "."
    block_positions = [i for i, (n, _) in enumerate(named) if n.startswith(prefix)]
    if not block_positions:
        raise ValidationError(f"block list {blocks_name} has no parameters")
    first_block, last_block = block_positions[0], block_positions[-1]

    # a vision-only first stage (first cut at 0) holds only the vision
    # encoder; the embeddings go to the first decoder stage
    encoder = _vision_encoder_path(model) if stages > 1 and bounds[1] == 0 else None
    assignment: dict[str, int] = {}
    by_id: dict[int, list[str]] = {}
    for i, (name, param) in enumerate(named):
        if name.startswith(prefix):
            block_index = int(name[len(prefix) :].split(".", 1)[0])
            stage = stage_of_block(block_index)
        elif i < first_block:
            stage = 1 if encoder is not None and not _under(name, encoder) else 0
        elif i > last_block:
            stage = stages - 1
        else:
            raise ValidationError(
                f"parameter {name} is interleaved with the block list; "
                f"v1 supports strictly sequential layouts only"
            )
        assignment[name] = stage
        by_id.setdefault(id(param), []).append(name)

    for names in by_id.values():
        stage_set = {assignment[n] for n in names}
        if len(stage_set) > 1:
            raise ValidationError(
                f"tied parameters {names} land on stages {sorted(stage_set)}: "
                f"cross-stage tied parameters are unsupported in v1 — untie "
                f"them (e.g. tie_word_embeddings=False) or change the partition"
            )

    return tuple(
        StagePartition(
            index=s,
            parameter_names=tuple(n for n, _ in named if assignment[n] == s),
            block_start=bounds[s],
            block_stop=bounds[s + 1],
        )
        for s in range(stages)
    )


class GenericSequentialStage(nn.Module):
    """Pre-modules, a block slice and post-modules chained sequentially. For
    models whose blocks take only the previous output."""

    def __init__(self, pre, blocks, post):
        super().__init__()
        self.pre = nn.ModuleList(pre)
        self.blocks = nn.ModuleList(blocks)
        self.post = nn.ModuleList(post)

    def forward(self, x):
        for module in (*self.pre, *self.blocks, *self.post):
            x = module(x)
        return x

    def local_blocks(self):
        return list(self.blocks)


def _checkpointed(forward, *args, **kwargs):
    import torch.utils.checkpoint

    return torch.utils.checkpoint.checkpoint(forward, *args, use_reentrant=False, **kwargs)


def recompute_blocks(stage: nn.Module, first: int | None = None) -> None:
    """Keep only each of the stage's own blocks' inputs for backward and
    recompute the rest there. Wraps each block's forward, not the block, so
    parameter names (checkpoints, weight files) stay the same, and module
    hooks still run once, outside the recomputed part. first: only the first
    `first` of the stage's pipeline blocks, as Megatron's block method; the
    blocks of an encoder the stage holds always (little compute for the
    activations they keep)."""
    blocks = stage.local_blocks()
    if first is not None:
        encoder = stage.encoder_blocks() if hasattr(stage, "encoder_blocks") else []
        own = [b for b in blocks if not any(b is e for e in encoder)]
        blocks = own[:first] + encoder
    for block in blocks:
        block.forward = functools.partial(_checkpointed, block.forward)


class _CompiledForward:
    """A block's forward, compiled (shapes dynamic) on its first call. Pickles
    as the plain forward, so each process compiles its own."""

    def __init__(self, forward):
        self.forward, self._compiled = forward, None

    def __call__(self, *args, **kwargs):
        if self._compiled is None:
            import torch

            self._compiled = torch.compile(self.forward, dynamic=True)
        return self._compiled(*args, **kwargs)

    def __getstate__(self):
        return {"forward": self.forward, "_compiled": None}


def compile_blocks(stage: nn.Module, encoder_only: bool = False) -> None:
    """Compile each of the stage's own blocks' forward: fuses the small
    elementwise ops between the matmuls. Like recompute_blocks, wraps the
    forward, not the block; apply it after recompute_blocks. encoder_only:
    only the blocks of the encoder the stage holds (its vision encoder)."""
    for block in stage.encoder_blocks() if encoder_only else stage.local_blocks():
        block.forward = _CompiledForward(block.forward)


def build_stage_module(
    model: nn.Module, block_start: int, block_stop: int, parameter_names: tuple[str, ...]
) -> nn.Module:
    """One stage as a GenericSequentialStage of deep copies, safe to ship to
    an actor."""
    import copy

    blocks_name, blocks = find_block_list(model)
    prefix = blocks_name + "."
    named = [n for n, _ in model.named_parameters(remove_duplicate=False)]
    first_block = min(i for i, n in enumerate(named) if n.startswith(prefix))

    pre_paths, post_paths = [], []
    for name in parameter_names:
        if name.startswith(prefix):
            continue
        path = name.rsplit(".", 1)[0]
        bucket = pre_paths if named.index(name) < first_block else post_paths
        if path not in bucket:
            bucket.append(path)

    return GenericSequentialStage(
        pre=[copy.deepcopy(model.get_submodule(p)) for p in pre_paths],
        blocks=[copy.deepcopy(blocks[i]) for i in range(block_start, block_stop)],
        post=[copy.deepcopy(model.get_submodule(p)) for p in post_paths],
    )


class CausalLMStage(nn.Module):
    """Stage of an HF llama-style causal LM (Qwen, Llama). Each stage carries
    its own copy of the parameter-free rotary embedding and recomputes (cos,
    sin) locally rather than shipping them between stages."""

    def __init__(self, emb, layers, norm, head, rotary):
        super().__init__()
        self.emb = emb
        self.layers = nn.ModuleList(layers)
        self.norm = norm
        self.head = head
        self.rotary = rotary

    def forward(self, x, position_ids=None):
        import torch

        if self.emb is not None:
            x = self.emb(x)
        if position_ids is None:
            position_ids = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        cos_sin = self.rotary(x, position_ids)
        for layer in self.layers:
            x = layer(x, position_ids=position_ids, position_embeddings=cos_sin)
        if self.norm is not None:
            x = self.norm(x)
        if self.head is not None:
            x = self.head(x)
        return x

    def local_blocks(self):
        return list(self.layers)


def build_causal_lm_stage(
    model: nn.Module, block_start: int, block_stop: int, parameter_names: tuple[str, ...]
) -> nn.Module:
    """One stage of an HF llama-style causal LM as a CausalLMStage."""
    import copy

    blocks_name, blocks = find_block_list(model)  # e.g. "model.layers"
    parent_path = blocks_name.rsplit(".", 1)[0]
    parent = model.get_submodule(parent_path)
    names = set(parameter_names)

    def owns(sub):
        return any(n.startswith(f"{parent_path}.{sub}.") for n in names)

    stage = CausalLMStage(
        emb=copy.deepcopy(parent.embed_tokens) if owns("embed_tokens") else None,
        layers=[copy.deepcopy(blocks[i]) for i in range(block_start, block_stop)],
        norm=copy.deepcopy(parent.norm) if owns("norm") else None,
        head=copy.deepcopy(model.lm_head) if any(n.startswith("lm_head.") for n in names) else None,
        rotary=copy.deepcopy(parent.rotary_emb),
    )
    attach_tp_plan(stage, getattr(getattr(model, "config", None), "base_model_tp_plan", None))
    # attention implementation, head counts and MoE routing for the engine.
    # Not named `config`: DeepSpeed would prefer HF's unfiltered TP plan.
    stage._rdsp_hf_config = getattr(model, "config", None)
    return stage


_CAUSAL_LM_PARTS = ("embed_tokens", "norm", "rotary_emb")
# model types whose forward CausalLMStage reproduces, checked on GPU with
# every intra-stage layout; other llama-shaped models differ in details it
# would silently drop (Gemma's embedding scale, Gemma2's logit soft-capping)
_CAUSAL_LM_TYPES = ("llama", "qwen3", "qwen3_moe")


def select_stage_builder(model: nn.Module):
    """The stage builder a model needs, decided from its structure.

    HF causal LMs of a _CAUSAL_LM_TYPES type with the llama-style layout
    (`embed_tokens`, `norm` and `rotary_emb` next to the block list, `lm_head`
    at the top) get build_causal_lm_stage, unless they use eager attention:
    that stage passes no mask, which SDPA and flash attention read as causal
    but eager attention reads as none. Other HF models (a `config`
    attribute) get build_hf_stage, which runs the model's own forward.
    Everything else gets build_stage_module. A rotary embedding outside an HF
    model raises ValidationError, since the generic chain would silently drop
    the positions."""
    blocks_name, _ = find_block_list(model)
    parent_path = blocks_name.rsplit(".", 1)[0] if "." in blocks_name else ""
    parent = model.get_submodule(parent_path) if parent_path else model
    present = {p for p in _CAUSAL_LM_PARTS if isinstance(getattr(parent, p, None), nn.Module)}
    has_head = isinstance(getattr(model, "lm_head", None), nn.Module)
    config = getattr(model, "config", None)
    model_type = getattr(config, "model_type", None)
    eager = getattr(config, "_attn_implementation", None) == "eager"
    if (
        model_type in _CAUSAL_LM_TYPES
        and not eager
        and present == set(_CAUSAL_LM_PARTS)
        and has_head
    ):
        return build_causal_lm_stage
    if hasattr(model, "config"):
        from ray_deepspeed_pipeline.hf_stage import build_hf_stage  # imports this module

        return build_hf_stage
    if "rotary_emb" in present:
        missing = [p for p in _CAUSAL_LM_PARTS if p not in present]
        if not has_head:
            missing.append("lm_head")
        raise ValidationError(
            f"{type(model).__name__} has a rotary embedding next to its block "
            f"list {blocks_name!r} but not the rest of the llama-style causal-LM "
            f"layout (missing {missing}); no built-in stage builder can pass it "
            f"positions correctly"
        )
    return build_stage_module


_TP_STYLES = ("colwise", "rowwise", "colwise_gather_output", "qkv_colwise")


def attach_tp_plan(stage: nn.Module, plan: dict | None) -> None:
    """Carry an HF tensor-parallel plan onto a stage module.

    Kept: colwise, rowwise, colwise_gather_output (split the output, then
    gather it on every rank: Qwen3.5's linear-attention projections) and
    qkv_colwise (a fused q/k/v projection split by thirds). Anything else
    makes AutoTP fall back to model-type presets, which do not recognize a
    stage module. `replicated_with_grad_allreduce` entries (Qwen3's
    q_norm/k_norm) stay replicated but see per-rank heads, so they are listed
    in `_rdsp_tp_grad_allreduce` for the adapter's TP gradient all-reduce."""
    plan = dict(plan or {})
    stage._tp_plan = {k: v for k, v in plan.items() if v.lower() in _TP_STYLES}
    stage._rdsp_tp_grad_allreduce = tuple(
        k for k, v in plan.items() if v.lower() == "replicated_with_grad_allreduce"
    )
