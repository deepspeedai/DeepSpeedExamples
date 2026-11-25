# How the Hugging Face ecosystem splits models for pipeline parallelism, and what rdsp should take from it

Date: 2026-09-22. Scope: splitting only (where to cut, how a stage module is built, how tied weights and
rotary embeddings are handled). Schedules are out of scope.

Versions read (installed in `../.venv`, abbreviated `SP/` =
`.venv/lib/python3.12/site-packages/`): transformers 5.16.1, accelerate 1.14.0, torch 2.13.0,
deepspeed 0.19.6. Shallow clones read on 2026-09-22: torchtitan `7349a22`, nanotron `6447b0d`,
picotron `59714b1`, vllm `4edb551` (only `vllm/model_executor/models/transformers/` and `vllm/distributed/`).

Note: `rdsp/uv.lock` does **not** contain a transformers entry (grep finds none), and `rdsp/pyproject.toml`
does not list it. transformers 5.16.1 comes from the parent workspace venv. If rdsp is going to read
transformers-specific attributes, pin it (at least as an optional/test extra).

Labels used below: **[verified]** = read in the source named; **[estimate]** = my own arithmetic;
**[unverified]** = not checked against source or not run.

---

## 1. What rdsp does today (for reference)

`src/ray_deepspeed_pipeline/partition.py`:

- `find_block_list` picks the longest `nn.ModuleList` whose children all share a class (`model.layers` for Qwen3).
  Ambiguity is rejected.
- `partition_parameters` cuts the block list (explicit cuts or uniform by block count, remainder to the
  earliest stages), then assigns every parameter name (FQN) to a stage: blocks by range, anything declared
  before the block list to stage 0, anything after it to the last stage. Parameters interleaved with the
  block list are rejected. Parameters shared across stages (tied) are rejected.
- `build_stage_module` → `GenericSequentialStage` (pre modules, block slice, post modules, called as `x = m(x)`).
  Does not work for HF decoder layers, which need `position_embeddings`.
- `build_causal_lm_stage` → `CausalLMStage`, hard-coded to the Llama/Qwen attribute names
  (`embed_tokens`, `layers`, `norm`, `rotary_emb`, `lm_head`). Each stage gets a copy of the
  parameter-free `rotary_emb` and recomputes `(cos, sin)` locally. No attention mask is passed, so SDPA
  falls back to `is_causal=True` (`SP/transformers/integrations/sdpa_attention.py:124`) **[verified]**.
- `compiler.lower` turns the partition into a plan of `StageSpec(parameter_names, block_start, block_stop)`.
  The driver builds each stage module by deep copy and ships it to a Ray actor
  (`stage_worker.create_stage_clients`).

Two side notes found while reading:

- Stage-local parameter names differ from the original model's names (`emb.weight` vs
  `model.embed_tokens.weight`, `layers.0.*` vs `model.layers.14.*`). Anything that maps stage parameters back
  to the driver model (parity tests, checkpoint save/load) needs a name map.
- `docs/P0.md:52-54` says the Qwen3-0.6B tie is "intra-stage-safe only because embed and lm_head land on
  different stages". That reads backwards: the tie is cross-stage, which is exactly why
  `tests/integration/test_p6_first_row.py:62-70` unties it before partitioning. Worth rewording.

---

## 2. transformers: `base_model_pp_plan` / `_pp_plan`

### 2.1 Format [verified]

Two class attributes, merged at model init:

- On the **config**: `base_model_pp_plan: dict[str, tuple[list[str], list[str]]]`
  (`SP/transformers/configuration_utils.py:178-179, 256`). Qwen3 example
  (`SP/transformers/models/qwen3/configuration_qwen3.py:56-60`):

  ```python
  base_model_pp_plan = {
      "embed_tokens": (["input_ids"], ["inputs_embeds"]),
      "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
      "norm": (["hidden_states"], ["hidden_states"]),
  }
  ```
- On the **model class**: `_pp_plan`, e.g. `Qwen3ForCausalLM._pp_plan = {"lm_head": (["hidden_states"], ["logits"])}`
  (`SP/transformers/models/qwen3/modeling_qwen3.py:434`).

Keys are child-module names of the base model, in execution order; the value is (input names, output
names). `DistributedMixin.init_parallel_plans` (`SP/transformers/distributed/mixin.py:62-83`) copies the
class plan, adds the config's `base_model_pp_plan` when the model is the base model, and prefixes children's
plans (`model.embed_tokens`, `model.layers`, `model.norm`, then `lm_head`). `PreTrainedModel.supports_pp_plan`
(`SP/transformers/modeling_utils.py:4687-4697`) is just "is any of those non-empty".

Coverage: 103 of the 496 `configuration_*.py` files define `base_model_pp_plan` (grep count, includes some
multimodal sub-configs). Every dense and MoE decoder family relevant to us has it (llama, qwen2, qwen3,
qwen3_moe, mistral, mixtral, gemma 1–4, phi, phi3, olmo 1–3, granite, deepseek v2/v3, gpt_oss, ...).
Key names vary: GPT-NeoX uses `embed_in`, `emb_dropout`, `final_layer_norm`; Phi adds `embed_dropout`,
`final_layernorm` (`models/gpt_neox/configuration_gpt_neox.py:54-58`, `models/phi/configuration_phi.py:56-60`).

### 2.2 What it does **not** say [verified]

- It does not list `rotary_emb`. Qwen3's rotary module is a child of `Qwen3Model` (`modeling_qwen3.py:357`)
  and its output is passed to every layer as `position_embeddings` (`modeling_qwen3.py:410-421`), but the
  plan's `layers` entry only names `hidden_states, attention_mask`.
- It does not say how `attention_mask` is built (Qwen3 builds a dict keyed by layer type, full vs sliding
  window, `modeling_qwen3.py:392-407`; Gemma3 also builds one rotary output per layer type,
  `models/gemma3/modeling_gemma3.py:561`).
- It does not cover logic that lives in `forward()` rather than in a module: Granite's
  `inputs_embeds * embedding_multiplier` (`models/granite/modeling_granite.py:397`) and
  `logits / logits_scaling` (`:497`); Gemma2's final-logit soft-capping (`models/gemma2/modeling_gemma2.py:527-530`).

So the plan is a list of which children are "boundary" modules and in what order. It is not a dataflow
contract you can execute from.

### 2.3 Who consumes it

- **transformers itself: nobody splits with it.** The only reader outside model files is
  `supports_pp_plan`, and nothing inside transformers calls that property (grep for `supports_pp_plan`
  returns only its definition). **[verified]**
- transformers 5.x **does** now ship its own pipeline split, `SP/transformers/distributed/pipeline_parallel.py`
  (`apply_pipeline_parallelism`, enabled with `DistributedConfig(pp_size=N)`, `distributed/mixin.py:177-206`).
  It ignores `pp_plan` — line 184 is literally `# TODO(3outeille): involves pp_plan to do the split ?`.
  It hard-codes `base_model.embed_tokens`, `.layers`, `.norm` and `model.lm_head`, replaces non-owned
  ones with a `PipelineIdentityLayer` (returns its first argument), splits layers by count with the whole
  remainder on the **last** rank (`layer_range_for_rank`, lines 108-123, with a TODO to balance by layer
  type or parameter bytes), and runs a single forward: non-first stages receive hidden states and feed them
  back into the unmodified HF `forward` as `inputs_embeds` (lines 234-238, 247-282). No microbatching, no
  schedule, no backward handling; logits are broadcast to every rank so `generate()` works. It is an
  inference path. **[verified]** The checkpoint loader uses the same hard-coded names to skip keys owned by
  other stages (`SP/transformers/core_model_loading.py:1496-1504`).
  For tied embeddings it keeps `embed_tokens` on the last stage too so `lm_head` can tie locally (lines
  193-196, 207-209). That is fine for inference and wrong for training (two copies, no gradient sync).
- **vLLM's Transformers modeling backend is the real consumer** (the attribute was added for it: transformers
  PR #36091 by hmellor, https://github.com/huggingface/transformers/pull/36091; vLLM PR #12832,
  https://github.com/vllm-project/vllm/pull/12832). `TransformersBase.pipeline_parallel()`
  (`vllm/model_executor/models/transformers/base.py:362-426`) uses **only the keys**: it finds the single
  `ModuleList` among them, replaces modules before it with `PPMissingLayer` except on the first rank (and
  the last rank when tied), slices the layers, replaces modules after it except on the last rank, and then
  runs the model's own `forward` with `inputs_embeds` from the previous rank. Without a plan it falls back
  to "children of the decoder that have parameters, in declaration order" — the same heuristic rdsp uses.
  The input/output names in the tuples are not read. **[verified]**
- There is a tensor-parallel analogue (`base_model_tp_plan`, `_tp_plan`) that transformers **does** execute
  (`distributed/tensor_parallel.py`, `apply_tensor_parallelism`), plus `_ep_plan` and `_fsdp_plan`. The PP
  plan is the only one of the four that transformers does not act on. **[verified]**

---

## 3. accelerate: `prepare_pippy`

`SP/accelerate/inference.py` **[verified]**:

- `prepare_pippy(model, split_points="auto", example_args=..., num_chunks=None)` is documented as "pipeline
  parallel inference" (line 131). It builds `torch.distributed.pipelining.pipeline(model, mb_args, mb_kwargs,
  split_spec={name: SplitPoint.BEGINNING})` and a `ScheduleGPipe(stage, num_chunks)` (lines 75-96). No loss
  function is passed to the schedule, so it is forward-only.
- `split_points="auto"` runs `infer_auto_device_map` with `max_memory = 1.1 × (model bytes / num_chunks)`
  per GPU (lines 31-54) and takes the first module assigned to each device as a split point (lines 164-168).
  That is a **parameter-memory** balance, not a compute balance.
- `num_chunks` is both the number of GPUs and the number of microbatches (default: number of processes).
- `Accelerator.pipeline_parallel_rank` raises `NotImplementedError("Pipeline parallelism is currently not
  supported in Accelerate.")` (`SP/accelerate/accelerator.py:794-799`), and `ParallelismConfig` has no pp
  dimension (`SP/accelerate/parallelism_config.py:70-75`). Training PP in accelerate exists only through the
  Megatron-LM plugin (`SP/accelerate/utils/megatron_lm.py`), which uses Megatron's own models.

---

## 4. `torch.distributed.pipelining` (upstream PiPPy)

Two front ends, same runtime (`SP/torch/distributed/pipelining/__init__.py`):

**Tracer front end** — `pipeline(module, mb_args, mb_kwargs, split_spec=...)` (`_IR.py:1247-1297`).
`split_spec` maps module FQNs to `SplitPoint.BEGINNING` / `END`; `annotate_split_points` monkey-patches those
modules' `forward` to call `pipe_split()` before/after (`_IR.py:1223-1244`). Alternatively put `pipe_split()`
calls in your own forward, or pass a `split_policy` graph transform. The model is traced with
`torch.export.export` (`_IR.py:1050-1064`), so it must be exportable with the example inputs; graph breaks
fail. **[verified]** Whether current Qwen3 exports cleanly with its mask/cache code was **not tested
[unverified]**.
Tied/shared parameters: the default is `MultiUseParameterConfig.REPLICATE` (`_IR.py:1076-1078`). Each stage
gets a deep copy (`_IR.py:604-622`) and the pairs are recorded in `pipe.replicated_params`, but nothing in
torch reads that list (grep of `torch/distributed` finds only the definition), and the file's own TODO says
"investigate gradient sync for shared parameters" (`_IR.py:33-35`). So gradients of a replicated tied weight
are **not** synchronised for you. **[verified]**

**Manual front end** — `PipelineStage(submodule, stage_index, num_stages, device, ...)` (`stage.py:1639+`).
You build the per-stage module yourself; shapes are given or inferred from the first microbatch. This is what
TorchTitan uses.

### TorchTitan (manual, module deletion)

`torchtitan/distributed/pipeline_parallel.py` **[verified]**:

- `_split_module` (lines 470-529): deep-copy the whole model, then for each top-level child keep it, trim a
  `ModuleList`/`ModuleDict` to the listed indices/keys, or set it to `None`. The model's `forward` is written
  to tolerate missing pieces: `h = self.tok_embeddings(tokens) if self.tok_embeddings is not None else tokens`,
  same for `norm` and `lm_head` (`torchtitan/models/common/decoder.py:282-294`). Layers are a `ModuleDict`, so
  keeping `"17"` preserves the original FQN `layers.17.*`.
- Cuts: user can give `module_fqns_per_model_part` (a list of module-name lists per stage). Otherwise
  `_generate_llm_fqn_per_model_part` (lines 357-467) treats the embedding as `input_weight` layers and
  norm+head as `output_weight` layers (config `pipeline_parallel_first_stage_less_layers` /
  `..._last_stage_less_layers`, both default **1**, `torchtitan/config/configs.py:198-208`) and splits the
  resulting count evenly. `pipeline_parallel_layers_per_stage` sets virtual stages for looped schedules.
- Rotary: the RoPE cache is a non-persistent buffer inside each attention module
  (`models/common/attention.py:829, 867-869`; `models/common/rope.py:111`), so every layer carries it and
  there is no cross-stage rotary input.
- Tied embeddings: `raise NotImplementedError("Weight tying is not supported with Pipeline Parallel.")`
  (`models/common/decoder.py:158-161`). Same policy as rdsp.

TorchTitan also has an **HF-transformers backend** (`torchtitan/experiments/transformers_modeling_backend/`):
`pipeline.py` copies the split above with two changes noted at line 35 — removed modules become `Identity`
instead of `None` (so the unmodified HF forward runs), and `rotary_emb` is added to **every** stage's module
list (line 66, 144). It only ties embeddings when both ends are on the same stage (`model.py:1416-1421`).
This is the closest existing design to what a generic rdsp builder would be.

---

## 5. nanotron, picotron, DeepSpeed PipelineModule, Trainer/TRL

**nanotron** (`src/nanotron/models/base.py:187-237`, `models/llama.py`) **[verified]**: every pipeline unit is
a `PipelineBlock(module_builder, module_input_keys, module_output_keys)` — the same "inputs/outputs by name"
idea that later appeared as `pp_plan` (e.g. decoder layer `{"hidden_states","sequence_mask"} ->
{"hidden_states","sequence_mask"}`, `llama.py:855-867`). Blocks are assigned greedily by cumulative
**compute cost** from `get_block_compute_costs()` (`llama.py:938-950`: per decoder layer
`4·n_heads·d_head·H + 3·d_ff·H`, lm_head `V·H`; embedding cost 0). Rotary lives inside attention
(`llama.py:400`). Tied embedding/lm_head across stages are supported: parameters are marked tied and
`sync_tied_weights_gradients` all-reduces their gradients (`parallel/tied_parameters.py:31, 121`).

**picotron** (`picotron/pipeline_parallel/pipeline_parallel.py:8-63`) **[verified]**: teaching code. Uniform
layer count (remainder to early ranks), embedding on first rank, norm + projection on last, others
`nn.Identity`. cos/sin are computed inside each `DecoderLayer` (`picotron/model.py:225`). No weight tying.

**DeepSpeed `PipelineModule`** (`SP/deepspeed/runtime/pipe/module.py`) **[verified]**: the model must be
rewritten as a flat list of `LayerSpec`s whose outputs feed the next layer directly (tensor or tuple).
`partition_method` (default `'parameters'`, line 133):
- `'uniform'`: equal layer count (`runtime/utils.py:614`).
- `'parameters'`: trainable parameter count per spec, split with `partition_balanced` (lines 409-411).
- `'type:<regex>'`: weight 1 for layers whose class name matches, 0 otherwise, balanced (lines 412-417).
- `'profile'`: raises `NotImplementedError` (line 418).
`partition_balanced` (`runtime/utils.py:635-673`) is a DP that minimises (max stage − min stage) weight, not the
max stage weight. Tied layers use `TiedLayerSpec(key, ...)`; the engine all-reduces their gradients between the
stages that hold them (`module.py:457-462`, `engine.py:278, 1386`). Rotary/masks must be threaded through the
tuple that layers pass to each other or recomputed per layer — DeepSpeed does not help.

**HF Trainer / TRL** **[verified for Trainer, unverified for TRL]**: `transformers/trainer*.py` and
`transformers/integrations/` contain no reference to `PipelineModule`, `pp_size` or pipeline parallelism; the
DeepSpeed integration is ZeRO only. TRL is not installed here; I know of no PP support in it, but did not check.

---

## 6. Comparison table

| Tool | How cuts are specified | Tracing vs manual | Tied embed/lm_head across stages | Rotary / per-layer extras | Training | Maturity |
|---|---|---|---|---|---|---|
| **rdsp** (today) | Explicit block cuts or uniform block count; pre/post modules pinned to first/last stage | Manual, name-based; stage module rebuilt (`CausalLMStage`) | Rejected | Copy `rotary_emb` to every stage, recompute cos/sin | Yes (1F1B via Ray + DeepSpeed per stage) | Prototype; one model row |
| transformers `base_model_pp_plan` | Declares boundary modules + order; no cut positions | n/a (metadata only) | Not addressed | Not described (rotary omitted) | n/a | Present on ~100 configs; not executed by transformers |
| transformers native PP (`DistributedConfig(pp_size)`) | Uniform layer count, remainder on last rank; hard-coded names | Manual: Identity replacement, original HF forward | Keeps embed on last stage and ties locally (inference-correct only) | Unchanged HF forward recomputes per stage | No (single forward, no schedule) | New in 5.x, "naive", TODOs |
| vLLM Transformers backend | Uniform, remainder away from first/last; `VLLM_PP_LAYER_PARTITION` override (`vllm/distributed/utils.py:128-168`) | Manual: `PPMissingLayer` replacement driven by `pp_plan` keys | Embed also on last rank | Unchanged HF forward | No (inference) | Production |
| accelerate `prepare_pippy` | `split_points` FQNs or `"auto"` (param-memory balance) | Tracing (`torch.export`) | Replicated, grads not synced | Whatever export captures | No (forward-only GPipe) | Stable but inference-only |
| torch `pipeline()` + `SplitPoint` | FQN → BEGINNING/END, `pipe_split()`, or `split_policy` | Tracing (`torch.export`) | Replicated; sync is user's job | Captured in graph | Yes (all torch schedules) | Works for exportable models; HF models are fragile [unverified for Qwen3] |
| torch `PipelineStage` (manual) | You build the module | Manual | Your problem | Your problem | Yes | Stable, recommended path |
| TorchTitan | `module_fqns_per_model_part` or auto (embed/head count as N layers) | Manual: deep copy + delete / `None` | `NotImplementedError` | RoPE buffer inside attention | Yes | Production for its own models |
| TorchTitan HF backend | Same | Manual: `Identity` + `rotary_emb` on every stage | Only ties if both on same stage | Copies `rotary_emb` per stage | Yes | Experimental |
| nanotron | Automatic greedy by analytic compute cost | Manual: model written as `PipelineBlock`s | Supported, grads all-reduced | Inside attention | Yes | Production-ish, own model code only |
| picotron | Uniform layer count | Manual, own model | No tying | Inside each layer | Yes | Educational |
| DeepSpeed `PipelineModule` | `partition_method`: uniform / parameters (default) / type:regex | Manual rewrite to `LayerSpec` list | `TiedLayerSpec`, grads all-reduced | User threads it through | Yes | Mature, but requires model rewrite |

---

## 7. How each tool decides where to cut, and what it means for Qwen3-0.6B

Qwen3-0.6B (`config.json` in the local HF cache): H=1024, FFN 3072, 28 layers, 16 query heads, 8 KV heads,
head_dim 128, vocab 151,936, tied embeddings.

Per-unit numbers **[estimate]** (analytic, not measured):

| Unit | Params | Forward FLOPs / token |
|---|---|---|
| Decoder layer | 15.7 M | 35.7 M at seq 512, 48.2 M at seq 2048 (matmuls + attention scores) |
| Embedding | 155.6 M (= 9.9 layers) | ~0 (a lookup) |
| lm_head (untied) | 155.6 M (= 9.9 layers) | 311 M (= 8.7 layers at 512, 6.5 at 2048) |

The embedding and lm_head weigh the same in parameters but not in compute. So:

| Policy | 2 stages | slowest / mean stage compute | 4 stages | slowest / mean |
|---|---|---|---|---|
| Uniform block count (rdsp, picotron) | 14 / 14 | 1.24 (seq 512) | 7/7/7/7 | 1.71 |
| Parameter-balanced (DeepSpeed default, accelerate auto) | 14 / 14 | 1.24 | 2/12/12/2 | 1.31 (better than uniform, but stages 0 and 3 sit mostly idle) |
| TorchTitan default (embed=1, head=1 layer) | 14 / 14 | 1.24 | 7/8/7/6 | 1.60 |
| Compute-balanced (nanotron-style) | 18 / 10 | 1.02 | 9/9/9/1 (seq 512), 8/9/9/2 (2048) | 1.04–1.06 |

In steady state the pipeline runs at the pace of its slowest stage, so for the 2-stage row the compute
balance would cut the per-microbatch step from ~22.7 to ~18.7 layer-units, about 18% less **[estimate]**.
Numbers ignore the cross-entropy/softmax over the 152 k vocabulary, which adds more to the last stage.

Memory pulls the other way. With 1F1B, stage *i* of *N* holds activations for up to *N − i* microbatches,
so stage 0 carries the most activation memory; moving layers onto it (18/10) raises its peak. The last stage
also holds fp32 logits: 512 tokens × 151,936 × 4 B ≈ 311 MB per sequence **[estimate]**. A balancer should
optimise compute and then check memory, not the reverse.

---

## 8. Recommendations for rdsp

### 8.1 Should `partition.py` consume `base_model_pp_plan`? Yes, as a hint for names, not as the builder contract.

What the plan reliably gives: the names and execution order of boundary modules (`embed_tokens` / `embed_in`
/ `emb_dropout` before, `norm` / `final_layer_norm` / `lm_head` after) and which child is the block list.
Use it to replace the "longest same-class `ModuleList`" heuristic when it exists, and keep the heuristic as
the fallback (that is exactly vLLM's behaviour). This removes the reliance on declaration order in
`partition_parameters` and gives clear errors for multi-list models (vLLM also rejects more than one
`ModuleList`).

What it does not give: rotary, masks, and logic in `forward()`. So a builder that lifts modules out and
re-wires them (today's `CausalLMStage`) cannot be made generic from the plan alone. For Qwen3-0.6B it works
because there is one rotary output and no sliding-window layers; for Gemma3 it would silently call the
rotary module with the wrong signature, and for any model with mixed `layer_types` it would use the wrong
mask per layer.

### 8.2 Recommended generic builder: "hollowed HF model" instead of re-wiring

Follow vLLM / transformers-native / TorchTitan-HF: keep the real HF model object, replace the modules this
stage does not own with a placeholder, and run the model's own `forward`. Properties:

- Rotary, masks per layer type, sliding windows, logit soft-capping and scaling all come from HF's own code.
  rdsp's per-stage rotary recompute happens for free (HF recomputes from `inputs_embeds`).
- Parameter names stay the original FQNs (`model.layers.17.self_attn.q_proj.weight`), so parity tests and
  checkpoints need no name map.
- Use Identity placeholders for non-owned layers rather than slicing the `ModuleList`: HF indexes
  `config.layer_types[i]` and each layer's `layer_idx` by position (`modeling_qwen3.py:412-415`), so a sliced
  list would pick the wrong mask type on mixed models. The cost of calling 20 no-op placeholders is negligible.

Known hazards, which argue for keeping a per-`model_type` allow-list (default-deny, like `support_matrix.py`):

- Anything `forward()` applies to `inputs_embeds` runs on **every** stage when hidden states are fed back in
  as `inputs_embeds`: Granite multiplies by `embedding_multiplier` (`models/granite/modeling_granite.py:397`),
  GPT-NeoX applies `emb_dropout` (`models/gpt_neox/modeling_gpt_neox.py:355`). transformers' native PP and
  vLLM appear to have this problem for Granite as well (from reading the code; not tested **[unverified]**).
- Non-last stages must return the base model's hidden states (not logits); the last stage calls the
  `*ForCausalLM` forward. Don't pass `labels` through HF (rdsp's `loss_fn` owns the loss).
- `use_cache=False` must be forced during training.

Code sketch (not run; names are illustrative):

```python
import copy
import torch
import torch.nn as nn


class _Skip(nn.Module):
    """Placeholder for a module owned by another stage: returns its first input."""
    def forward(self, *args, **kwargs):
        return args[0] if args else next(iter(kwargs.values()))


def pp_layout(model: nn.Module) -> tuple[list[str], str, list[str]]:
    """(pre FQNs, block-list FQN, post FQNs) in execution order.
    Prefers the HF pp_plan; falls back to find_block_list + declaration order."""
    plan = getattr(model, "pp_plan", None) or {}          # merged, prefixed: model.embed_tokens, ..., lm_head
    if plan:
        names = list(plan)
        lists = [n for n in names if isinstance(model.get_submodule(n), nn.ModuleList)]
        if len(lists) != 1:
            raise ValidationError(f"pp_plan must have exactly one ModuleList, got {lists}")
        i = names.index(lists[0])
        return names[:i], lists[0], names[i + 1:]
    blocks_name, _ = find_block_list(model)
    ...  # current declaration-order logic, returned as module FQNs


class HFCausalLMStage(nn.Module):
    """One pipeline stage that runs the unmodified HF forward on a hollowed model."""
    def __init__(self, hf_model, is_first: bool, is_last: bool):
        super().__init__()
        self.hf = hf_model
        self.is_first, self.is_last = is_first, is_last

    def forward(self, x):
        kw = {"use_cache": False}
        kw.update({"input_ids": x} if self.is_first else {"inputs_embeds": x})
        if self.is_last:
            return self.hf(**kw).logits
        base = getattr(self.hf, self.hf.base_model_prefix)
        return base(**kw).last_hidden_state                # norm is _Skip on non-last stages


def build_hf_stage(model, block_start, block_stop, parameter_names, *, is_first, is_last):
    pre, blocks_name, post = pp_layout(model)
    blocks = model.get_submodule(blocks_name)
    not_owned = [model.get_submodule(n) for n in (pre if not is_first else [])]
    not_owned += [model.get_submodule(n) for n in (post if not is_last else [])]
    not_owned += [blocks[i] for i in range(len(blocks)) if not block_start <= i < block_stop]
    # deepcopy with a pre-seeded memo: non-owned modules are never copied, so the driver
    # does not pay for a full second model per stage. [unverified trick; test it]
    memo = {id(m): _Skip() for m in not_owned}
    stage_model = copy.deepcopy(model, memo)
    got = {n for n, _ in stage_model.named_parameters(remove_duplicate=False)}
    if got != set(parameter_names):                        # must match partition_parameters exactly
        raise ValidationError(f"stage params differ from plan: {sorted(got ^ set(parameter_names))[:5]}")
    return HFCausalLMStage(stage_model, is_first, is_last)
```

The final check ties the builder to the name-based plan the compiler already produces; with original FQNs
preserved it becomes an exact set comparison. The `create_stage_clients` builder signature would need
`is_first`/`is_last` (or derive them from `block_start == 0` / `block_stop == n_blocks`).

Keep `CausalLMStage` until the new builder passes the existing P6 parity test on Qwen3-0.6B, then retire it.

### 8.3 Add a compute-balanced partition policy: yes

Add a `BalancedBlocks(cost="flops")` policy (keep `ExplicitCuts` and uniform):

1. Units = [pre modules as one unit] + each block + [post modules + loss as one unit].
2. Cost per unit from shapes, per token: `2 × (weight elements of every nn.Linear in the unit)` plus
   `4 × seq_len × n_heads × head_dim` per attention layer (from config). Embedding cost 0 (it is a lookup).
   This is nanotron's approach (`llama.py:938-950`), computed from the module tree instead of hand-written,
   so it works on any HF model.
3. Solve min–max contiguous partition (minimise the largest stage). Binary search on the answer or a small DP
   over 28 blocks is instant. Don't copy DeepSpeed's objective (it minimises max − min spread).
4. Check memory per stage afterwards (params + optimizer state + 1F1B in-flight activations `(N − i)` ×
   per-microbatch activations + last-stage logits) and reject or shift the cut if a stage exceeds its GPU.
   Print the chosen cuts and per-stage cost so users can copy them into `ExplicitCuts`.
5. Later, optional: a `cost="profile"` mode that times one forward+backward per block inside a stage actor.
   Nobody in this survey ships one (DeepSpeed's `'profile'` raises `NotImplementedError`).

Do **not** make parameter-count balancing the default. For Qwen3 it gives 14/14 at 2 stages (the same as
uniform, because the embedding and lm_head weigh the same) and 2/12/12/2 at 4 stages, which beats uniform (1.31 vs 1.71) but still leaves the first and last stages mostly idle, because it balances the embedding's memory, not its (near-zero) compute.

### 8.4 Tied embeddings

rdsp's rejection matches TorchTitan. If support is wanted later, the proven pattern is DeepSpeed's
`TiedLayerSpec` / nanotron's `sync_tied_weights_gradients`: each end keeps its own copy, gradients are summed
between the first and last stage before the optimizer step, and weights stay equal because both apply the
same update. In rdsp that means a gradient exchange between stage 0 and stage N−1 actors (a two-rank NCCL
group or coordinator-driven transfer) before DeepSpeed's `step()`. Don't adopt the transformers/vLLM trick of
keeping `embed_tokens` on the last stage — it is inference-only.

### 8.5 Housekeeping

- Pin transformers in `rdsp/pyproject.toml` (optional extra) if partition code starts reading `pp_plan`;
  the attribute and `DistributedMixin` are 5.x-era and moved recently (`distributed/mixin.py` is copyright
  2026).
- Fix the `docs/P0.md` wording about the tie (section 1).

---

## Sources

Local (transformers 5.16.1, accelerate 1.14.0, torch 2.13.0, deepspeed 0.19.6 under
`../.venv/lib/python3.12/site-packages/`): file paths as cited inline.

Repositories (shallow clones, commits above):
- TorchTitan: https://github.com/pytorch/torchtitan — `torchtitan/distributed/pipeline_parallel.py`,
  `torchtitan/experiments/transformers_modeling_backend/pipeline.py`, `torchtitan/models/common/decoder.py`
- nanotron: https://github.com/huggingface/nanotron — `src/nanotron/models/base.py`, `src/nanotron/models/llama.py`,
  `src/nanotron/parallel/tied_parameters.py`
- picotron: https://github.com/huggingface/picotron — `picotron/pipeline_parallel/pipeline_parallel.py`, `picotron/model.py`
- vLLM: https://github.com/vllm-project/vllm — `vllm/model_executor/models/transformers/base.py`, `vllm/distributed/utils.py`
- transformers PR adding the PP plan: https://github.com/huggingface/transformers/pull/36091
- vLLM PR using it: https://github.com/vllm-project/vllm/pull/12832
