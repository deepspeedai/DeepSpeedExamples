# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""Pipeline stages of a Hugging Face model that run the model's own forward.

A stage is a copy of the whole module tree in which everything it does not
own is swapped out: blocks outside its range pass their input through, and
other modules holding parameters it does not own become stand-ins. Calling the
model then runs its own setup code (masks, positions, rotary tables, vision
merge) and only this stage's blocks, so no architecture needs a hand-written
stage forward.

Besides the hidden state, a boundary carries the tensor arguments the model
passes to every downstream block (rotary tables, masks, ...): they may depend
on inputs only stage 0 sees (Qwen3-VL's image positions), and may differ per
block type (Gemma3's local and global rotary tables). Stage 0's loop still
calls the pass-through blocks after its range, so it records each one's
arguments; blocks sharing the same tensors are sent as one group.
"""

import copy
import json
import os

import torch
import torch.nn as nn

from ray_deepspeed_pipeline.errors import ValidationError
from ray_deepspeed_pipeline.partition import (
    attach_tp_plan,
    find_block_list,
    vision_injection_depth,
)
from ray_deepspeed_pipeline.vision import PrecomputedVision, VisionTower, find_vision_encoder


class _PassThrough(nn.Module):
    """A block another stage owns. Returns its input shaped like the real
    blocks' output: a tensor, or a tuple whose first item is the hidden state
    (Bloom). `form` is shared with the stage, which learns it from its own
    blocks; until then a 1-tuple, which the model's loop either indexes or
    hands on unchanged until the stage's first block replaces it."""

    def __init__(self, form: list):
        super().__init__()
        self.form = form  # [None | 0 for a tensor | tuple length]

    def forward(self, hidden_states, *args, **kwargs):
        n = self.form[0]
        if n == 0:
            return hidden_states
        return (hidden_states, *[None] * ((n or 1) - 1))


class _Unowned(nn.Module):
    """A module whose parameters another stage owns. Returns its input, or
    zeros for an embedding (a later stage's setup may still add position
    embeddings). Keeps meta-device stand-ins of its parameters, outside the
    parameter list, for model code that reads their dtype or shape."""

    def __init__(self, module: nn.Module):
        super().__init__()
        self._embedding_dim = getattr(module, "embedding_dim", None)
        for name, p in module.named_parameters(recurse=False):
            object.__setattr__(self, name, torch.empty(p.shape, dtype=p.dtype, device="meta"))

    def forward(self, x, *args, **kwargs):
        if self._embedding_dim is None:
            return x
        dtype = getattr(self, "weight", x).dtype
        return torch.zeros(*x.shape, self._embedding_dim, dtype=dtype, device=x.device)


def _block_tensors(args: tuple, kwargs: dict) -> dict:
    """A block call's tensor arguments besides the hidden state: positional
    ones as #i, keyword ones by name, tuples of tensors as name.i."""
    out = {f"#{i}": a for i, a in enumerate(args) if i and torch.is_tensor(a)}
    for name, value in kwargs.items():
        if torch.is_tensor(value):
            out[name] = value
        elif isinstance(value, tuple) and value and all(torch.is_tensor(v) for v in value):
            out.update({f"{name}.{i}": v for i, v in enumerate(value)})
    return out


def _with_block_tensors(args: tuple, kwargs: dict, tensors: dict) -> tuple[tuple, dict]:
    """(args, kwargs) with the entries of a _block_tensors() dict put back in."""
    args, kwargs, tuples = list(args), dict(kwargs), {}
    for key, value in tensors.items():
        name, _, index = key.partition(".")
        if key.startswith("#"):
            args[int(key[1:])] = value
        elif index:
            tuples.setdefault(name, {})[int(index)] = value
        else:
            kwargs[name] = value
    for name, parts in tuples.items():
        kwargs[name] = tuple(parts[i] for i in range(len(parts)))
    return tuple(args), kwargs


class _Recorded(Exception):
    """Ends a non-last stage's run of the model once the last block's
    arguments are recorded."""


_FORM_KEY = "@block_form"
_TOKEN_IDS_KEY = "@input_ids"


def _pack(per_block: dict, rows: int, block_form: int | None) -> dict:
    """{block: {arg: tensor}} as boundary extras keyed "6+7+9/arg": blocks
    receiving the same tensors share one entry. Tensors are laid out rows
    first like the hidden state, so the boundary code can split them per
    rank; a broadcast row is expanded. Tuple-returning blocks also send their
    tuple length, which the next stage's pass-throughs need before any of its
    own blocks has run."""
    groups = {}
    for block, tensors in sorted(per_block.items()):
        signature = tuple(sorted((k, id(v)) for k, v in tensors.items()))
        groups.setdefault(signature, ([], tensors))[0].append(str(block))
    out = {}
    for blocks, tensors in groups.values():
        for name, t in tensors.items():
            if t.dim() >= 1 and t.shape[0] == 1 and rows > 1:
                t = t.expand(rows, *t.shape[1:])
            # rows first, possibly folded with another dim (Bloom: rows x heads)
            if t.dim() < 2 or t.shape[0] % rows:
                raise ValidationError(
                    f"block argument {name!r} of shape {tuple(t.shape)} has no leading "
                    f"row dimension of size {rows}, so it cannot cross a stage boundary"
                )
            out[f"{'+'.join(blocks)}/{name}"] = t.detach()
    if block_form:
        out[_FORM_KEY] = torch.full((rows, 1), block_form, dtype=torch.int64)
    return out


def _unpack(extras: dict) -> dict:
    per_block = {}
    for key, t in extras.items():
        if key.startswith("@"):
            continue
        blocks, _, name = key.partition("/")
        for block in blocks.split("+"):
            per_block.setdefault(int(block), {})[name] = t
    return per_block


class HFModelStage(nn.Module):
    """forward(x=None, position_ids=None, **kwargs). Stage 0: the model's
    inputs, as x (input ids) or as keyword tensors. Later stages: x is the
    hidden state and the keyword tensors are the block arguments from
    upstream (their keys contain "/"). Returns the logits on the last stage,
    otherwise (hidden, extras) for the next stage."""

    def __init__(
        self,
        model,
        blocks_name: str,
        block_start: int,
        block_stop: int,
        is_first: bool,
        is_last: bool,
        block_form: list,
        sends_token_ids: bool = False,
        embeds_token_ids: bool = False,
    ):
        super().__init__()
        self.model = model
        self._block_form = block_form  # shared with the _PassThrough blocks
        self._blocks_name = blocks_name
        self._local = (block_start, block_stop)
        self.is_first, self.is_last = is_first, is_last
        # a vision-only first stage sends the token ids; the next stage, which
        # owns the embeddings, embeds the text tokens and keeps the received
        # image features at the image positions
        self._sends_token_ids, self._embeds_token_ids = sends_token_ids, embeds_token_ids
        self._hooked = False
        self._input = self._received = self._hidden = None
        self._downstream = {}

    def local_blocks(self):
        """The stage's own blocks: its range of the pipeline's block list, plus
        encoder_blocks()."""
        start, stop = self._local
        pipeline = self.model.get_submodule(self._blocks_name)
        return [pipeline[i] for i in range(start, stop)] + self.encoder_blocks()

    def encoder_blocks(self):
        """Blocks of every other block list the stage owns whole (a vision
        encoder's), not counting lists nested inside those (MoE experts)."""
        pipeline = self.model.get_submodule(self._blocks_name)
        blocks, taken = [], [self._blocks_name + "."]
        for name, module in self.model.named_modules():
            if (
                isinstance(module, nn.ModuleList)
                and len(module) >= 2
                and not name.startswith(tuple(taken))
                and module is not pipeline
                and len({type(b) for b in module}) == 1
                and not isinstance(module[0], (_PassThrough, _Unowned))
                and next(module.parameters(), None) is not None
            ):
                blocks.extend(module)
                taken.append(name + ".")
        return blocks

    def _install_hooks(self):
        # installed on first use, so the pickled stage carries no bound hooks
        if self._hooked:
            return
        blocks = self.model.get_submodule(self._blocks_name)
        start, stop = self._local
        self._n_blocks = len(blocks)
        for i in range(start, len(blocks)):
            blocks[i].register_forward_pre_hook(self._before_block(i), with_kwargs=True)
        if start < stop:  # a stage without blocks (vision only) has none to learn from
            blocks[start].register_forward_hook(self._learn_block_form)
        self._hooked = True

    def _learn_block_form(self, block, args, out):
        self._block_form[0] = len(out) if isinstance(out, tuple) else 0

    def _before_block(self, index: int):
        start, stop = self._local

        def hook(block, args, kwargs):
            args, kwargs = _with_block_tensors(args, kwargs, self._received.get(index, {}))
            if index == start and not self.is_first:
                # the model's setup may transform inputs_embeds (e.g. scale it)
                if args:
                    args = (self._input, *args[1:])
                else:
                    kwargs["hidden_states"] = self._input
            if index == stop:
                # what the loop hands the next block, so it includes the
                # model's own work between blocks (Qwen3-VL adds vision
                # features after block 0 and 1) but not code after the last
                # block (norm, head, logit soft-capping), which is not ours
                self._hidden = args[0] if args else kwargs["hidden_states"]
            if index >= stop:
                self._downstream[index] = _block_tensors(args, kwargs)
            if index == self._n_blocks - 1 and not self.is_last:
                # everything this stage sends is recorded; what follows is
                # other stages' work, and a stage without blocks would run it
                # on pass-through outputs of unknown form
                raise _Recorded
            return args, kwargs

        return hook

    def _embed_text(self, received, ids):
        """The input embeddings: the text tokens' own, the received image
        (and video) features at their positions."""
        config = self.model.config
        media = torch.zeros_like(ids, dtype=torch.bool)
        for name in ("image_token_id", "video_token_id"):
            token = getattr(config, name, None)
            if token is not None:
                media |= ids == token
        text = self.model.get_input_embeddings()(ids).to(received.dtype)
        return torch.where(media.unsqueeze(-1), received, text)

    def _setup_input(self, x):
        """What the model's setup code gets as inputs_embeds on a later stage:
        the hidden state itself when it is embedding-shaped, else zeros of
        that shape (GLM-5.3's hidden state has 4 streams). Blocks never see
        it: the first local block gets the hidden state from the hook."""
        embed = self.model.get_input_embeddings()
        dim = getattr(embed, "_embedding_dim", None) or getattr(embed, "embedding_dim", None)
        if dim is None or tuple(x.shape[2:]) == (dim,):
            return x
        return x.new_zeros(*x.shape[:2], dim)

    def forward(self, x=None, position_ids=None, **kwargs):
        """position_ids: a sequence shard's global positions (Ulysses SP)."""
        self._install_hooks()
        extras = {k: v for k, v in kwargs.items() if "/" in k or k.startswith("@")}
        if x is None:
            x = {k: v for k, v in kwargs.items() if k not in extras}
        if self._block_form[0] is None and _FORM_KEY in extras:
            self._block_form[0] = int(extras[_FORM_KEY][0, 0])  # once: it syncs
        if self._embeds_token_ids:
            x = self._embed_text(x, extras[_TOKEN_IDS_KEY])
        self._input, self._received, self._downstream = x, _unpack(extras), {}
        try:
            kw = {"use_cache": False}
            if position_ids is not None:
                kw["position_ids"] = position_ids
            try:
                if not self.is_first:
                    out = self.model(inputs_embeds=self._setup_input(x), **kw)
                elif isinstance(x, dict):
                    out = self.model(**x, **kw)
                else:
                    out = self.model(input_ids=x, **kw)
            except _Recorded:
                out = None
            if self.is_last:
                return out.logits if hasattr(out, "logits") else out[0]
            hidden = self._hidden
            out = _pack(self._downstream, hidden.shape[0], self._block_form[0])
            if self._sends_token_ids:
                out[_TOKEN_IDS_KEY] = x["input_ids"]
            return hidden, out
        finally:
            self._input = self._received = self._hidden = None
            self._downstream = {}


def build_hf_stage(
    model: nn.Module,
    block_start: int,
    block_stop: int,
    parameter_names: tuple[str, ...],
    stage_index: int | None = None,
) -> HFModelStage:
    """One stage of an HF model as an HFModelStage. Modules it does not own
    are never copied, so building from a full model costs only the stage's
    own share of memory. stage_index: the stage's position; without it the
    first stage is the one owning the input embeddings or a vision-only one."""
    blocks_name, blocks = find_block_list(model)
    depth = vision_injection_depth(model)
    if block_start == 0 and block_stop < depth:
        raise ValidationError(
            f"{type(model).__name__} adds vision features inside blocks 0-{depth - 1}, "
            f"which only the first stage can do: put the first cut at {depth} or later"
        )
    owned = set(parameter_names)
    # the first stage owns the input embeddings; the one after a vision-only
    # stage starts at block 0 too
    embedding = model.get_input_embeddings()
    owns_embedding = any(p is embedding.weight for n, p in model.named_parameters() if n in owned)
    vision_only = block_stop == 0
    is_first = stage_index == 0 if stage_index is not None else owns_embedding or vision_only
    block_form = [None]
    memo = {}
    for i, block in enumerate(blocks):
        if not block_start <= i < block_stop:
            memo[id(block)] = _PassThrough(block_form)
    encoder = _colocated_encoder(model, owned) if is_first else None
    if encoder is not None:  # it runs on every rank instead (vision.py)
        memo[id(encoder)] = PrecomputedVision(encoder)
    for name, module in model.named_modules():
        direct = [f"{name}.{p}" if name else p for p, _ in module.named_parameters(recurse=False)]
        if direct and not owned.intersection(direct):
            memo.setdefault(id(module), _Unowned(module))
    stage_model = copy.deepcopy(model, memo)
    kept = {n for n, _ in stage_model.named_parameters(remove_duplicate=False)}
    if kept != owned:
        raise ValidationError(
            f"stage over blocks [{block_start}, {block_stop}) cannot be cut out cleanly: "
            f"a module mixes parameters of different stages "
            f"({sorted(kept ^ owned)[:3]}...)"
        )
    stage = HFModelStage(
        stage_model,
        blocks_name,
        block_start,
        block_stop,
        is_first=is_first,
        is_last=block_stop == len(blocks),
        block_form=block_form,
        sends_token_ids=is_first and vision_only and not owns_embedding,
        embeds_token_ids=not is_first and owns_embedding,
    )
    # what intra-stage parallelism reads: the AutoTP plan, head counts
    # (Ulysses) and MoE settings (AutoEP), all on the text model's config
    config = getattr(model, "config", None)
    text_config = config.get_text_config() if hasattr(config, "get_text_config") else config
    plan = dict(
        getattr(text_config, "base_model_tp_plan", None)
        or _standard_tp_plan(blocks_name, blocks, _STANDARD_TP)
    )
    # some configs' plans omit the per-head norms (Qwen3-VL-MoE, GLM-4-MoE):
    # without their gradient sum the TP replicas drift apart
    for name, style in _standard_tp_plan(blocks_name, blocks, _PER_HEAD_NORMS).items():
        plan.setdefault(name, style)
    for name, vision_blocks in _vision_block_lists(model, blocks_name):
        plan.update(_standard_tp_plan(name, vision_blocks, _VISION_TP))
    attach_tp_plan(stage, plan)
    stage._rdsp_hf_config = text_config
    return stage


def _colocated_encoder(model: nn.Module, owned: set) -> nn.Module | None:
    """The vision encoder, if the model has one and the stage owns none of
    its parameters: colocated vision."""
    if getattr(getattr(model, "config", None), "vision_config", None) is None:
        return None
    try:
        path = find_vision_encoder(model)
    except ValidationError:
        return None
    if any(n.startswith(path + ".") for n in owned):
        return None
    return model.get_submodule(path)


# projection names shared by most decoder families, and how Megatron-style
# tensor parallelism splits them
_STANDARD_TP = {
    "self_attn.q_proj": "colwise",
    "self_attn.k_proj": "colwise",
    "self_attn.v_proj": "colwise",
    "self_attn.o_proj": "rowwise",
    "mlp.gate_proj": "colwise",
    "mlp.up_proj": "colwise",
    "mlp.down_proj": "rowwise",
    # per-head norms: replicated, but each rank sees only its heads
    "self_attn.q_norm": "replicated_with_grad_allreduce",
    "self_attn.k_norm": "replicated_with_grad_allreduce",
}


_PER_HEAD_NORMS = {k: v for k, v in _STANDARD_TP.items() if v == "replicated_with_grad_allreduce"}


# the same for vision encoder blocks (Qwen-VL family); qkv is one fused
# projection, split by thirds so every rank keeps whole heads of q, k and v
_VISION_TP = {
    "attn.qkv": "qkv_colwise",
    "attn.proj": "rowwise",
    "mlp.linear_fc1": "colwise",
    "mlp.linear_fc2": "rowwise",
}


def _standard_tp_plan(blocks_name: str, blocks, table: dict) -> dict:
    """TP plan entries for a block list from `table`: the projections that
    every block has. Used for decoders whose config has no plan (most, e.g.
    Qwen3-VL), and for vision encoders, which never have one."""
    names = [{n for n, _ in block.named_modules()} for block in blocks]
    prefix = blocks_name.rsplit(".", 1)[-1]
    return {
        f"{prefix}.*.{name}": style
        for name, style in table.items()
        if all(name in block_names for block_names in names)
    }


def _vision_block_lists(model: nn.Module, blocks_name: str) -> list:
    """(name, ModuleList) of a vision model's other block lists: lists of
    >=2 same-class blocks outside the decoder's."""
    if getattr(getattr(model, "config", None), "vision_config", None) is None:
        return []
    return [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.ModuleList)
        and len(module) >= 2
        and len({type(b) for b in module}) == 1
        and name != blocks_name
        and not name.startswith(blocks_name + ".")
    ]


def _checkpoint_files(weights_dir: str) -> dict:
    """Parameter name -> safetensors file, for an HF checkpoint directory."""
    index = os.path.join(weights_dir, "model.safetensors.index.json")
    if os.path.exists(index):
        with open(index) as f:
            return {k: os.path.join(weights_dir, v) for k, v in json.load(f)["weight_map"].items()}
    single = os.path.join(weights_dir, "model.safetensors")
    if not os.path.exists(single):
        raise ValidationError(f"no safetensors checkpoint in {weights_dir!r}")
    from safetensors import safe_open

    with safe_open(single, "pt") as f:
        return {k: single for k in f.keys()}


def load_meta_parameters(stage: nn.Module, weights_dir: str | None) -> None:
    """Replace every parameter still on the meta device by its value from the
    HF checkpoint in weights_dir; only this stage's tensors are read. stage:
    an HFModelStage, or a rank's colocated VisionTower."""
    meta = [(n, p) for n, p in stage.named_parameters() if p.is_meta]
    if not meta:
        return
    if weights_dir is None:
        raise ValidationError(
            f"{len(meta)} parameters are on the meta device (e.g. {meta[0][0]}); pass "
            f"weights=<HF checkpoint dir> to rdsp.initialize() to load them per stage"
        )
    if not isinstance(stage, (HFModelStage, VisionTower)):
        raise ValidationError("weights= needs a Hugging Face model (the HF stage builder)")
    if any(b.is_meta for b in stage.buffers()):
        raise ValidationError(
            "buffers are on the meta device too; build the model skeleton with "
            "accelerate.init_empty_weights(), which keeps buffers real"
        )
    files = _checkpoint_files(weights_dir)
    # a tied checkpoint stores only one of each tied pair (Qwen3-VL-2B has no
    # lm_head); the untied copy is read from the tensor it was tied to
    tied = getattr(getattr(stage, "model", None), "_tied_weights_keys", None)
    tied = tied if isinstance(tied, dict) else {}
    by_file = {}
    for name, param in meta:
        key = (
            stage.checkpoint_key(name)
            if isinstance(stage, VisionTower)
            else name.removeprefix("model.")
        )
        if key not in files:
            key = tied.get(key, key)
        if key not in files:
            raise ValidationError(f"{key!r} is not in the checkpoint at {weights_dir!r}")
        by_file.setdefault(files[key], []).append((name, key, param))
    from safetensors import safe_open

    for path, entries in by_file.items():
        with safe_open(path, "pt") as f:
            for name, key, param in entries:
                module_name, _, leaf = name.rpartition(".")
                value = f.get_tensor(key).to(param.dtype)
                stage.get_submodule(module_name)._parameters[leaf] = nn.Parameter(
                    value, requires_grad=param.requires_grad
                )
