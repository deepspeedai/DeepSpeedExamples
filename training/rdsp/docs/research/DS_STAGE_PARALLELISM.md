# Parallelism inside one pipeline stage with DeepSpeed: a source-checked cookbook

Date: 2026-09-22. DeepSpeed checkout: `../DeepSpeed` at rev
`53a2ac44` (read only, not modified). Transformers 5.16.1 and torch 2.13.0 from the workspace `.venv`
(`SP/` below = `.venv/lib/python3.12/site-packages/`). All DeepSpeed paths are relative to the checkout.

Scope: how to turn on four kinds of parallelism for one rdsp stage, i.e. one stage-local
`torch.distributed` world that we create ourselves before calling `deepspeed.initialize(...,
dist_init_required=False)`. The four kinds:

- **Tensor parallelism (TP)**: each weight matrix is split across GPUs. DeepSpeed calls its version
  "AutoTP".
- **Sequence parallelism (SP)**: each GPU holds a slice of the sequence (token positions) for the same
  rows. DeepSpeed's version for Hugging Face models is "Ulysses".
- **Expert parallelism (EP)**: for mixture-of-experts (MoE) layers, each GPU holds a subset of the experts
  and tokens are sent to the GPU that owns their expert. DeepSpeed calls it "AutoEP". "Parallel Folding"
  is DeepSpeed's name for running AutoEP and AutoTP in the same stage: the attention part is split by TP
  and the expert part by EP, over the same GPUs.
- **Data parallelism (DP) with ZeRO stage 0/1/2**: different rows on each GPU, gradients averaged. ZeRO-1
  splits optimizer state across GPUs and ZeRO-2 also splits gradients.

Labels used below:
- **[verified]**: read in the source cited.
- **[ran]**: checked by running a small experiment on CPU with the gloo backend, 2 processes, fp32
  (setup in Appendix A). Nothing here ran on GPU/NCCL.
- **[inferred]**: my reading of the source; not run.
- **[unverified]**: not checked.

---

## 0. Findings that change how rdsp works today

1. **The adapter's gradient accumulation is wrong under ZeRO-1 and ZeRO-2 [ran].**
   `deepspeed_adapter.py` sets `gradient_accumulation_steps=1`, calls `engine.backward` N times, then
   `engine.step()` once. With gas=1, every backward counts as an accumulation boundary. At each boundary,
   ZeRO-1/2 *overwrites* its reduced gradient buffer (`deepspeed/runtime/zero/stage_1_and_2.py:926-939`).
   The step therefore uses only the **last** microbatch's gradient. Measured parameter error against the
   true full-batch update: 4.0e-2 under ZeRO-1 and ZeRO-2. ZeRO-0 is correct but runs N allreduces.
   Fix: call `engine.set_gradient_accumulation_boundary(k == N-1)` before each backward (Section 4).
   With that fix the error is ≤ 7.5e-9 for ZeRO-0/1/2, including 1F1B interleaving and our
   `(out*grad).sum()` backward.
2. **The adapter's `reset()` does not discard a half-finished step under ZeRO-2 [ran].** Under ZeRO-2,
   gradients that were already reduced sit in `optimizer.all_grad_tensors`, and neither
   `engine.zero_grad()` nor `optimizer.zero_grad()` clears it (`stage_1_and_2.py:1931-1946`). Measured
   error after an abandoned step: 5.0e-2. Clearing that private dict fixes it (Section 4.4).
3. **DeepSpeed's default `gradient_clipping` is 1.0** (`deepspeed/runtime/constants.py:254`). Each stage
   engine clips by its own stage-local gradient norm. That differs from clipping a single model by its
   global norm. Set `"gradient_clipping": 0.0` for parity tests. If real clipping is needed, rdsp has to
   compute the norm across stages itself.
4. **AutoTP's default path fails on `CausalLMStage` [ran].** The error is `AssertionError: Not able to
   determine model policy automatically`. It works when the config supplies explicit layer rules
   (`preset_model: "qwen2"` or `partition_config`).
5. **AutoTP gives Qwen3's `q_norm`/`k_norm` wrong gradients [ran].** Those weights are shared across
   attention heads, and after TP each rank sees only its own heads. Each rank gets a partial gradient and
   DeepSpeed never sums it across the TP group, so the TP ranks drift apart (error about 1.0 relative).
   HF marks these layers `replicated_with_grad_allreduce` (`SP/transformers/models/qwen3/
   configuration_qwen3.py:49-50`). DeepSpeed's plan converter supports only `colwise` and `rowwise`
   (`deepspeed/module_inject/tp_plan_converter.py:12`). A 6-line gradient hook fixes it [ran]
   (Section 1.6).
6. **The Ulysses tutorial's loss gives SP-size-times-too-large gradients [ran].** With SP=2 and gradient
   clipping off, the tutorial's differentiable `all_gather` loss produced exactly 2.000× the true
   gradient. Having each SP rank use `local token-loss sum / global token count` gives the exact gradient
   (Section 2.6). [inferred] DeepSpeed's own test (`tests/unit/ulysses_alst/test_ulysses_sp_hf.py`)
   probably passes because the default clipping of 1.0 hides the factor.
7. **Ulysses checkpoints do not reload on SP rank ≥ 1 [ran].** The model file is written only as
   `mp_rank_00_model_states.pt`, but the load looks for `mp_rank_{sp_rank}` and fails with `IndexError`
   (`deepspeed/runtime/state_dict_factory.py:91`). Wrapping the SP group object so that it reports model
   rank 0 fixes it [ran] (Section 2.8).

---

## 1. AutoTP (tensor parallelism) training

### 1.1 Config and how it is triggered [verified]

```json
"tensor_parallel": {
  "autotp_size": 2,
  "partition_config": { "use_default_specs": false, "layer_specs": [ ...see 1.2... ] }
}
```
Alternatively use `"preset_model": "qwen2"` instead of `partition_config`.

- `deepspeed/__init__.py:210-212`: `autotp_size > 0` turns on AutoTP training mode. Schema:
  `deepspeed/runtime/tensor_parallel/config.py:38-139` (`TPTrainingConfig`: `autotp_size`,
  `tp_overlap_comm`, `tp` (alias of `tensor_parallel`), `partition_config`, `preset_model`,
  `keep_module_on_host`). The `dtype` field (default fp16) is used only by the heuristic path we cannot
  use.
- The engine sets up TP only if `autotp_size > 1` (`deepspeed/runtime/engine.py:311-312`), after AutoEP
  (`engine.py:310`).
- ZeRO-3 is rejected (`engine.py:630-631`).
- No `mpu` is needed. The engine replaces `self.mpu` with the `deepspeed.utils.groups` module and builds
  the TP mesh itself (`engine.py:633-634`). Any user-supplied `mpu` is silently overwritten here, which
  matters for combining with SP (Section 5).

### 1.2 Which layers get split, and whether a non-HF root works [verified + ran]

Priority order (`engine.py:700-758`):
1. `partition_config` / `preset_model`: regex rules on parameter names (`engine.py:706-719`).
2. The HF `tp_plan`, read from `model.config.base_model_tp_plan` or `model._tp_plan`
   (`tensor_parallel/config.py:149-163`, `engine.py:731-753`).
3. The heuristic parser `AutoTP.tp_parser` (`engine.py:756-758`).

For `CausalLMStage`:
- Path 2 does not apply: the stage has no `config` attribute and no `_tp_plan`. Even if we attached the
  HF config, the Qwen3 plan contains `replicated_with_grad_allreduce`. The converter then returns `None`
  (`tp_plan_converter.py:25-30`) and DeepSpeed falls through to path 3.
- Path 3 fails. `AutoTP.supported()` runs a regex over `str(model)` looking for `": (.*?)Model"`
  (`deepspeed/module_inject/auto_tp.py:238-249`). The repr of `CausalLMStage` contains no "Model", so
  the assertion fails. **[ran]: `AssertionError('Not able to determine model policy automatically...')`.**
- Path 1 works. `_replace_module` walks the module tree and builds dotted names starting at our root
  (`auto_tp.py:572-633`), e.g. `layers.0.self_attn.q_proj`. It appends `.weight` and matches with
  `re.match` (`autotp_config.py:199-210`, `auto_tp.py:411-442`). The model type comes from
  `module.config.model_type` or a class-name regex (`auto_tp.py:480-498`). For our root it is `None`,
  which only matters for rules that set `model_types`; the built-in presets do not.

Rules for Qwen3 (identical to the `qwen2` preset, `autotp_config.py:488-508`). Use them as a
`partition_config` so they can be combined with AutoEP (Section 5):
```python
QWEN3_TP_SPECS = {"use_default_specs": False, "layer_specs": [
    {"patterns": [r".*\.self_attn\.o_proj\.weight$"],     "partition_type": "row"},
    {"patterns": [r".*\.self_attn\.[qkv]_proj\.weight$"], "partition_type": "column"},
    {"patterns": [r".*\.mlp\.down_proj\.weight$"],        "partition_type": "row"},
    {"patterns": [r".*\.mlp\.(gate|up)_proj\.weight$"],   "partition_type": "column"},
]}
```

What happens to each layer [ran, tp=2]:
- `q/k/v_proj`, `gate/up_proj` become `LinearLayer`: split by output features (column split).
- `o_proj`, `down_proj` become `LinearAllreduce`: split by input features (row split), then summed across
  the TP group.
- Everything else stays **replicated** (a full copy on every TP rank): all RMSNorms including
  `q_norm`/`k_norm`, `emb` (Embedding), `head` (Linear), and `rotary`.
  - A Linear that matches no rule is left unchanged (`auto_tp.py:425-431`).
  - An Embedding is sliced only if some rule matches its name (`auto_tp.py:593-603`).
- `update_mp_params` shrinks head-count attributes (`auto_tp.py:518-532`). Qwen3Attention has none of
  those; its forward uses `view(..., -1, head_dim)` (`SP/transformers/models/qwen3/modeling_qwen3.py:250`),
  so it adapts to the local head count automatically.

### 1.3 Rank layout [verified]

`groups._init_tp_mesh_device` builds a device mesh of shape `(dp, tp)` with dimension names
`("data_parallel", "tensor_parallel")` (`deepspeed/utils/groups.py:136-171`).

- **TP groups are consecutive ranks**: {0,1},{2,3},… **DP groups are strided**: {0,2},{1,3},…
- The fallback path makes the same groups explicitly (`groups.py:88-133`).
- The groups live in module globals and are created only once per process (`groups.py:143`).
- `get_tensor_model_parallel_src_rank()` is `(rank // tp) * tp` (`groups.py:233-238`).

### 1.4 What each TP rank sends and receives

- **Input**: identical on every rank of a TP group; different across DP groups. On the first forward the
  engine installs a one-time pre-hook. It broadcasts `args`/`kwargs` from the TP source rank and asserts
  that they are equal (`engine.py:637-683`). [ran]
- **Output**: replicated. `RowParallel.forward` all-reduces the row-split output
  (`layers.py:150-162`), and the `head` is not split. [ran: both TP ranks' outputs match the unsplit
  reference to 3.6e-7.]
- **Gradient w.r.t. the stage input**: complete and identical on every TP rank, provided the incoming
  output gradient is identical on every TP rank (rdsp sends the same gradient to all of them).
  - The column split's autograd function `ColumnParallel` is identity in forward and **all-reduces the
    input gradient in backward** (`deepspeed/module_inject/layers.py:201-228`). The overlapped variant
    `AsyncColumnParallel` does the same (`layers.py:189-198`).
  - `RowParallel.backward` passes the gradient through unchanged (`layers.py:164-169`).
  - [ran: input-grad error vs unsplit reference 7.2e-7 on both ranks.]
  - Caveat [inferred]: `ColumnParallel.backward` runs `all_reduce(grad_output.contiguous())` and then
    returns `grad_output` (`layers.py:227-228`). If `grad_output` were not contiguous, the reduction would
    act on a copy and the returned gradient would be unreduced. It was contiguous in every run.
- **Gradients of split weights**: each rank's local slice is exact [ran].
- **Gradients of replicated weights**: exact and identical on every rank, **except** Qwen3 `q_norm` and
  `k_norm` (Section 1.6).

### 1.5 DP world size and `train_batch_size` [verified + ran]

- The engine's DP size is `world/tp`. `groups.mpu` is the `groups` module (`engine.py:1671`), so
  `_get_data_parallel_world_size` uses `get_data_parallel_world_size()` of the TP mesh
  (`groups.py:771-782`). [ran: `dp_world_size=1` for world 2, tp 2.]
- **But `DeepSpeedConfig` validates batch sizes using the full world**, because there is no `mpu` and no
  mesh (`deepspeed/runtime/config.py:703-708`, `907-922`). So an explicit `train_batch_size` must equal
  `micro * gas * WORLD`, not `micro * gas * dp`. [ran: derived `train_batch_size=2` for micro 1, world 2,
  tp 2.]
- rdsp does not use DeepSpeed's dataloader, so this number is bookkeeping only. **Recommendation:** do
  not set `train_batch_size`. Pass `train_micro_batch_size_per_gpu` and `gradient_accumulation_steps` and
  let DeepSpeed derive it (`config.py:949-952`). The "world" in that check differs by feature (TP/EP:
  full world, SP: world/sp), so the adapter's current `micro * world` formula is wrong for SP.

### 1.6 The Qwen3 `q_norm`/`k_norm` fix [ran]

`q_norm` and `k_norm` are applied per head on `[.., heads, head_dim]` (`modeling_qwen3.py:252-253`).
After the column split each rank holds `heads/tp` heads, so its weight gradient covers only those heads.
Nothing in DeepSpeed sums replicated-parameter gradients across TP:
- `is_model_parallel_parameter` is used only for grad-norm bookkeeping (`stage_1_and_2.py:1531`,
  `runtime/utils.py:397`).
- The one TP-replicated reduction that exists is limited to AutoEP folding (`engine.py:2778-2795`).

Fix: sum these gradients over the TP group in a tensor hook. A tensor hook runs before PyTorch adds the
new gradient into `.grad`, so it is in place before DeepSpeed's ZeRO hooks see it:
```python
def install_qk_norm_tp_grad_sum(engine):
    from deepspeed.utils import groups
    tp_group = groups.get_tensor_model_parallel_group()
    def _sum(g):
        g = g.clone(); torch.distributed.all_reduce(g, group=tp_group); return g
    for n, p in engine.module.named_parameters():
        if n.endswith(("self_attn.q_norm.weight", "self_attn.k_norm.weight")):
            p.register_hook(_sum)
```
[ran, ZeRO-1 and ZeRO-2, tp=2: `q_norm`/`k_norm` error dropped from ~1.0 relative to 4e-7.]

### 1.7 Divisibility and head/embedding sharding

- Training-mode splitting uses `torch.chunk` with **no divisibility check** (`layers.py:639`, `layers.py:726`).
  - Require `num_attention_heads % tp == 0`, `num_key_value_heads % tp == 0`, and
    `intermediate_size % tp == 0`.
  - Otherwise shards are uneven or cut through a head. Expect an error at `view(..., -1, head_dim)`, or
    silently wrong math if the sizes happen to line up. Qwen3-0.6B (16 q heads, 8 kv heads, 3072 MLP):
    tp ∈ {1, 2, 4, 8}.
- Leave `emb` and `head` replicated (the rules above do). If they were split:
  - Embedding slicing splits the **hidden** dimension (`auto_tp.py:500-516`) and nothing gathers it
    afterwards.
  - A row-split `lm_head` uses `LmHeadLinearAllreduce`. It slices the input and calls
    `inference_all_reduce`, which is not autograd-aware (`layers.py:888-913`). The input gradient would
    then be partial on each rank [inferred].
  - Our module's attribute is named `head`, not `lm_head`, so the special case never triggers.
  - No vocabulary-divisibility requirement applies because nothing splits the vocabulary.

### 1.8 Checkpoints with AutoTP [verified + ran]

- **One file per TP rank.** The checkpoint name uses `mpu.get_model_parallel_rank()` = TP rank
  (`engine.py:4018-4041`). ZeRO files are `zero_pp_rank_{dp}_mp_rank_{tp}_optim_states.pt`
  (`engine.py:4002-4013`).
- **DP rank 0 of each TP lane writes.** The writer check uses the DP rank (`engine.py:1459-1474`).
- [ran, tp=2 ZeRO-1: files `mp_rank_00/01_model_states.pt` and `zero_pp_rank_0_mp_rank_00/01_optim_states.pt`;
  reload with the same topology restored parameters exactly.]
- Reloading with a different `tp` requires DeepSpeed's "universal checkpoint" conversion. The metadata is
  saved (`engine.py:5050-5052`). [unverified]

### 1.9 Failure modes to test (AutoTP)
- A root module without rules. Expect the `AssertionError` from 1.2.
- `autotp_size` does not divide the world (`groups.py` `_ensure_divisibility`).
- `autotp_size` that does not divide the head counts.
- ZeRO-3 (assert at `engine.py:630`).
- Inputs that differ within a TP group. Expect the first-forward consistency assert.
- Divergence of `q_norm`/`k_norm` across TP ranks without the hook, and none with it.
- An explicit `train_batch_size = micro * dp`. Expect the batch assert.
- A second engine in the same process with a different `tp`. Groups are cached (`groups.py:143`), so
  expect a silent wrong layout [inferred].

---

## 2. Ulysses sequence parallelism for HF layers

### 2.1 Call signature and return value [verified]

`UlyssesSPAttentionHF.register_with_transformers(model_name_or_path, core_attn_implementation,
sequence_parallel_size, micro_batch_size, seq_length=None, seq_length_is_variable=True,
disable_in_eval=False, max_length=None)`
(`deepspeed/runtime/sequence_parallel/ulysses_sp.py:394-561`).

**Returns** the module `deepspeed.runtime.sequence_parallel.parallel_state_sp`, which acts as a
Megatron-style `mpu`. It returns `None` if `sequence_parallel_size == 1`.

What it does:
- Creates the SP groups (`ulysses_sp.py:442`). This asserts that `deepspeed.comm` is initialized
  (`parallel_state_sp.py:20`). **[ran] calling `torch.distributed.init_process_group` alone is not
  enough.** First call `deepspeed.comm.init_distributed(dist_backend="nccl", dist_init_required=False)`.
- **It can be called only once per process**; a second call asserts (`parallel_state_sp.py:42`).
- `model_name_or_path` can be anything with a `.config` attribute (duck-typed, `ulysses_sp.py:445-447`),
  so `types.SimpleNamespace(config=hf_config)` works for our layer slice [ran]. It reads
  `num_attention_heads`, `num_key_value_heads`, `head_dim` and `num_hidden_layers`
  (`ulysses_sp.py:494-512`). The last one is used only for a debug mode.

### 2.2 How attention is switched [verified]

Nothing on the model changes. The call **replaces the process-wide entry**
`ALL_ATTENTION_FUNCTIONS[core_attn_implementation]` with a Ulysses wrapper (`ulysses_sp.py:559`).
Consequences:
- `config._attn_implementation` must already equal `core_attn_implementation`, or the call raises
  (`ulysses_sp.py:452-458`). Every `Qwen3Attention` looks it up on each forward
  (`modeling_qwen3.py:262-264`).
- `eager` is rejected (`ulysses_sp.py:461-468`). Use `sdpa` or a flash-attention variant.
- Every module in that process that uses this attention implementation goes through Ulysses, including
  eval, unless `disable_in_eval=True`, which bypasses it when `module.training` is False
  (`ulysses_sp.py:262`).
- The wrapper drops any tensor `attention_mask` (`ulysses_sp.py:539`). Causal masking then comes from
  SDPA's `is_causal` (`SP/transformers/integrations/sdpa_attention.py:124`).

### 2.3 What to pass to `deepspeed.initialize` [verified + ran]

- `mpu=<returned module>`.
- Do **not** set both `sequence_parallel_size` and `data_parallel_size` in the config. When both are
  present, `initialize` also builds a `(dp, sp)` mesh (`deepspeed/__init__.py:203-206`), and
  `groups._get_*` prefers that mesh over the `mpu` (`groups.py:701-715`, `799-805`).
- `sequence_parallel_size` alone is harmless once `mpu` is passed (`config.py:696-700`). It is not needed.
- With this `mpu`, DeepSpeed sees:
  - config world = `world/sp` (`config.py:697-698`), so derived `train_batch_size = micro * gas * world/sp`;
  - `engine.dp_world_size = None`, because `_get_data_parallel_world_size` returns `None` for an SP mpu
    (`groups.py:778-779`);
  - `seq_dp_world_size = world`;
  - `sequence_parallel_size = sp`.
  - [ran: `{'dp': None, 'seq_dp': 2, 'sp': 2}`]

### 2.4 Rank layout [verified]
- **SP groups are consecutive**: {0..sp-1}, {sp..2sp-1}, … (`parallel_state_sp.py:43-47`).
- The "sequence-data-parallel" group, over which ZeRO splits state and reduces gradients, is **the whole
  world** (`parallel_state_sp.py:52-58`; the engine uses it at `engine.py:1715`).
- DP index = `rank // sp`.

### 2.5 Inputs, positions, outputs and gradients per rank [verified + ran]

- **Rows**: every SP rank of a group gets the **same rows**, and exactly `micro_batch_size` of them (the
  value passed at registration). The shape is asserted (`ulysses_sp.py:297-303`, shapes set at
  `ulysses_sp.py:158-164`), so our microbatch row count must equal it.
- **Sequence**: rank r gets the contiguous chunk `[r*L/sp, (r+1)*L/sp)`.
  - `L % sp == 0`, and chunks must be equal-sized; the all-to-all reshapes assume this
    (`ulysses_sp.py:186-188`).
  - This matches the chunking in `UlyssesSPDataLoaderAdapter` (`ulysses_sp.py:697-715`).
- **Heads**: `num_attention_heads % sp == 0`. KV heads must divide `sp` or be divisible by it; if
  `sp > kv_heads`, the KV heads are replicated (`ulysses_sp.py:129-148`).
- **position_ids are required**, as global offsets. They are passed as a keyword argument all the way
  into the attention call, then gathered across SP ranks (`ulysses_sp.py:285-294`). `Qwen3DecoderLayer`
  forwards `position_ids` to attention (`modeling_qwen3.py:305-314`). **`CausalLMStage` must change**:
  ```python
  def forward(self, x, position_ids=None):
      if self.emb is not None: x = self.emb(x)
      if position_ids is None:
          position_ids = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
      cos_sin = self.rotary(x, position_ids)          # global offsets -> correct rotary angles
      for layer in self.layers:
          x = layer(x, position_ids=position_ids, position_embeddings=cos_sin)
      ...
  ```
  Call it as `engine(x_chunk, position_ids=torch.arange(r*c, (r+1)*c)[None])`.
- **Output**: the chunk for the same positions.
- **Gradient w.r.t. the input chunk**: exact `dL/d(chunk)`, provided the backward seed on each rank is the
  true gradient for its own chunk.
- [ran, layer-slice stage without a PreTrainedModel root, `(out*G_chunk).sum()` backward, ZeRO-0/1/2:
  parameter-update error vs full-sequence reference 7.2e-7; input-gradient error 1.8e-7.]
- DeepSpeed's gradient reduction **sums over SP and averages over DP**: it divides by
  `sdp_world / sp = dp` (`stage_1_and_2.py:1211`, `1356`). That is correct exactly when each SP rank's
  backward seed is the true partial derivative (no extra factor).

### 2.6 Loss on the terminal stage [ran]

Use local sum over global count, with no differentiable gather:
```python
def sp_terminal_loss(logits_chunk, shift_labels_chunk, sp_group, n_mb):
    tok = F.cross_entropy(logits_chunk.flatten(0, 1).float(), shift_labels_chunk.flatten(),
                          ignore_index=-100, reduction="sum")
    n = (shift_labels_chunk != -100).sum()
    torch.distributed.all_reduce(n, group=sp_group)   # no autograd needed: a count
    return tok / n.clamp_min(1) / n_mb                # backward seed on this rank is exact
```
To report the loss, all-reduce `tok.detach()` over `sp_group` and divide by `n`.

- Labels must be **shifted before chunking**: `shift_labels = pad(labels, (0,1), -100)[..., 1:]`, then
  chunk. Otherwise each chunk boundary loses a label (tutorial and `ulysses_sp.py:703-705`).
- [ran] The tutorial's weighted `torch.distributed.nn.functional.all_gather` loss (docs tutorial,
  `ulysses_sp.py:1532-1545`) gives **exactly sp× the gradient**. Every SP rank backpropagates the same
  total loss, and `all_gather`'s backward sums over ranks
  (`SP/torch/distributed/nn/functional.py`, `_AllGather.backward`). Measured ratio of update norm to true
  gradient norm: 2.0000 for sp=2. It is invisible under Adam, which ignores a uniform gradient scale, but
  wrong for SGD and for clipping.
- `deepspeed/sequence/cross_entropy.py` (`vocab_sequence_parallel_cross_entropy`) expects the Megatron
  layout `[S/P, B, V]`. Its backward takes only the local slice (no sp× factor), but it indexes
  `grad_2d[arange, target]` with raw targets (`cross_entropy.py:48-55`). A `-100` padding label would be
  used as an index [inferred], so it is unsafe with shifted labels. Not recommended.

### 2.7 `UlyssesSPDataLoaderAdapter` [verified]

The adapter wraps a DataLoader:
- Each step it all-gathers one batch from every SP rank, pre-shifts the labels, and splits every
  tensor's sequence dimension (`ulysses_sp.py:644-715`).
- Iteration k over the SP group then processes rank k's sample, sharded across all SP ranks, so SP
  iterations cover SP samples (`ulysses_sp.py:566-600` docstring).
- It requires `position_ids` in the batch (`ulysses_sp.py:660-667`).

rdsp's coordinator feeds data itself, so we should **not** use the adapter. We should copy its chunking
(same rows, contiguous chunks, pre-shifted labels, global `position_ids`).

### 2.8 ZeRO and checkpoints with SP

- **ZeRO-0/1/2 all give exact results [ran]** (numbers in 2.5). ZeRO splits its state over the whole
  stage world (the sequence-data-parallel group).
- ZeRO-3 with Ulysses is what the tutorial shows [unverified here].
- **Checkpoint bug [ran]**:
  - Only sequence-data-parallel rank 0 writes the model file, named `mp_rank_00_model_states.pt`
    (`engine.py:1467-1474`).
  - On load each rank looks for `mp_rank_{mpu.get_model_parallel_rank()}`
    (`engine.py:4248-4249`). The SP `mpu` returns the SP rank there (`parallel_state_sp.py:94`).
  - SP rank 1 gets `IndexError: list index out of range`.
  - Copying the file to `mp_rank_01` fails a count check instead (`state_dict_factory.py:174`).

  **Working fix [ran]**: pass a wrapper that reports model rank 0 (the model is a full copy on every SP
  rank):
  ```python
  sp_mpu = UlyssesSPAttentionHF.register_with_transformers(...)
  mpu = types.SimpleNamespace(**{k: getattr(sp_mpu, k) for k in dir(sp_mpu) if not k.startswith("__")})
  mpu.get_model_parallel_rank = lambda: 0
  mpu.get_model_parallel_world_size = lambda: 1
  ```
  With this wrapper, save and reload restored parameters exactly on both SP ranks. Gradient clipping
  still used the correct global norm (clip=1.0 gave an update norm of exactly 1.0, with and without the
  wrapper, ZeRO-1 and ZeRO-2).

### 2.9 Failure modes to test (Ulysses)
- `register_with_transformers` before `deepspeed.comm.init_distributed`. Expect an `AssertionError`.
- A second registration in the same process.
- `_attn_implementation` mismatch, or `eager`.
- A missing `position_ids` keyword. Expect the assertion message.
- Row count different from `micro_batch_size`. Expect the shape assert.
- `L % sp != 0` or uneven chunks.
- `heads % sp != 0`.
- SP rank ≥ 1 reloading a checkpoint without the wrapper.
- Tutorial loss vs local-sum loss under SGD. Expect 2× vs 1×.
- Both `data_parallel_size` and `sequence_parallel_size` in the config. Expect mesh and mpu to disagree
  [inferred].

---

## 3. AutoEP (expert parallelism) and Parallel Folding

### 3.1 Config for a `CausalLMStage` holding Qwen3-MoE layers [verified + ran]

```json
"expert_parallel": {
  "enabled": true,
  "autoep_size": 2,
  "preset_model": "qwen3_moe",
  "moe_layer_pattern": "layers\\.\\d+\\.mlp",
  "top_k": 2,
  "route_norm": true,
  "use_grouped_mm": true
}
```
The config keys are parsed at `deepspeed/module_inject/auto_ep_config.py:41-87`.

- **Preset matching.** The preset uses `moe_layer_pattern = r"model\.layers\.\d+\.mlp"`
  (`auto_ep_presets/qwen3_moe.py:12`), matched with `re.fullmatch` against `named_modules()` names
  (`auto_ep.py:295`). Our names are `layers.N.mlp`, so **the preset alone detects nothing**. Override
  the pattern: the override is applied at `auto_ep_presets/registry.py:126-127`, `158-161`.
- **Child structure.** A matched module must have an `experts` child with 3-D `gate_up_proj`, plus a
  `gate` child (`auto_ep.py:307-330`). Dense `Qwen3MoeMLP` layers have no `experts` child and are skipped
  (`auto_ep.py:309-315`).
- **Without `model.config`** (our root has none, `auto_ep.py:275`):
  - `num_experts` comes from the router weight shape (`auto_ep.py:341-350`).
  - **`top_k` must be set explicitly**, or init fails with "Could not determine top_k"
    (`auto_ep.py:356-361`).
  - **`route_norm` must equal the model's `norm_topk_prob`**. Without a config the preset default is
    `True` (`auto_ep_presets/base.py:227-239`, `qwen3_moe.py:24`), but HF's default is `False`
    (`SP/transformers/models/qwen3_moe/configuration_qwen3_moe.py:106`).
- `use_grouped_mm: true` needs `torch._grouped_mm` (`deepspeed/moe/ep_experts.py:172-176`). On CPU
  use `false` (for-loop path).
- **Optimizer**: ZeRO-1/2 require MoE parameter groups, otherwise init fails with "None of the param groups
  are marked as MoE" (`stage_1_and_2.py:749-760`). [ran]
  - An optimizer defined in the config gets them automatically (`engine.py:1899-1901`).
  - A client optimizer (or a callable) must split them itself:
    `torch.optim.SGD(split_params_into_different_moe_groups_for_optimizer({"params": list(ps)}), ...)`
    (`deepspeed/moe/utils.py:72`).
  - Do not pass `model_parameters`: the engine collects them after AutoEP has created the new expert
    parameters (`engine.py:377-382`).

### 3.2 Routing: every token is processed (no capacity limit) [verified + ran]

`AutoEPMoELayer.forward` computes top-k routing and sends each token's copies to the owning ranks with an
all-to-all of variable size per rank (`deepspeed/module_inject/auto_ep_layer.py:596-720`,
`_AllToAllV` at `205-248`).
- There is **no capacity factor, no token dropping and no padding to a capacity**. Even the folded path
  builds `drop_mask=zeros` (`auto_ep_layer.py:641`).
- No capacity key exists in the config (`auto_ep_config.py:41-87`). (The older `deepspeed.moe.layer.MoE`
  has `capacity_factor`; AutoEP does not use it.)
- `load_balance_coeff` must be null (`auto_ep_config.py:64-72`).
- AutoEP computes no auxiliary loss. HF computes the Qwen3-MoE auxiliary loss only at model level when
  `output_router_logits` is on, so the non-EP baseline must not include it either.
- [ran, ep=2, different rows per rank vs a single-process baseline averaged over DP: max relative error
  2.5e-5 (fp32 summation order) over all parameters and input gradients, for ZeRO-0, 1 and 2.]
- Remaining parity risks:
  - [unverified] Ties in top-k under bf16.
  - [unverified] The grouped-matmul path's numerics.
  - [verified] The router's parameter name changes: `mlp.gate.weight` becomes `mlp.router.gate.weight`,
    and `experts.gate_up_proj`/`down_proj` become `experts.w1` (gate), `w3` (up) and `w2` (down)
    (`deepspeed/moe/ep_repack.py:39-75`).

### 3.3 Rank layout [verified]

Without TP or SP: `groups._create_expert_and_data_parallel`, legacy path (`groups.py:361-402`):
- **EP groups are consecutive**: {0..ep-1}, {ep..2ep-1}, … (`groups.py:396-397`).
- **Expert-data-parallel (EDP) groups are strided**: {i, i+ep, …} (`groups.py:372`).
- **Dense (non-expert) DP group = the whole stage.**
- [ran, world 2, ep 2: EP [0,1], EDP [0], dp 2.]

With SP (`mp_mode="sp"`): EP groups consecutive over the stage ranks (`groups.py:447-455`)
[inferred, not run].

### 3.4 What each EP rank receives, and gradient averaging [verified + ran]

- **Each EP rank gets different rows, like DP.** For rdsp an EP stage is a DP stage of size `world`, with
  expert weights split inside it.
- **Expert parameters** are tagged `allreduce=False`, `group_name="ep_size_N"`
  (`auto_ep_layer.py:500-505`). Their gradients are summed over the EDP group and divided by the
  **full DP world**:
  - ZeRO path: `stage_1_and_2.py:1356` plus the expert DP process group, `1316-1317`.
  - ZeRO-0 path: `engine.py:3577-3598`, comment "utilize dp_world_size".
  - This matches the non-EP baseline in which each rank's loss is the mean over its own rows.
- **Router, shared-expert and dense parameters** use the normal DP allreduce (`auto_ep_layer.py:511-522`).

### 3.5 Parallel Folding (AutoEP + AutoTP in one stage) [verified; not run]

Meaning: attention is split by TP×DP and experts by EP×EDP, over the same ranks. The folding spec is
built in `deepspeed/module_inject/auto_ep_folding.py:81-127`; the rank tables at `130-158`:
- TP groups consecutive.
- Dense DP groups strided by TP lane.
- EP groups made of consecutive stage ranks, which may cross TP lanes.
- EDP groups by position within the EP group.

Enabled simply by setting both `tensor_parallel.autotp_size > 1` and `expert_parallel` (`engine.py:555-592`).

Folded forward: each TP group gets identical tokens. The routed assignments are divided among the TP
peers, dispatched over EP, and gathered back over TP (`auto_ep_layer.py:625-657`, `706-711`).

Requirements (`auto_ep_folding.py:184-262`):
- `autoep_size > 1`.
- `pp_size == 1`, which refers to DeepSpeed's own pipeline; our stage-local world has none.
- No SP.
- `expert_tensor_parallel_size == 1`.
- ZeRO ≤ 2 and no offload.
- No `use_data_before_expert_parallel`.
- **If both `tensor_parallel.preset_model` and `expert_parallel.preset_model` are set, they must be
  equal.** There is no `qwen3_moe` AutoTP preset (`autotp_config.py:540-554`), so for Qwen3-MoE use
  `tensor_parallel.partition_config` (Section 1.2) together with `expert_parallel.preset_model:
  "qwen3_moe"`.

Gradient rules under folding (`auto_ep_folding.py:341-462`), applied at the accumulation boundary
(`engine.py:2761-2762`):
- Replicated and dense parameters: averaged over TP.
- Router: averaged.
- Experts: divided by tp.

**[inferred] Qwen3 `q_norm`/`k_norm` would then get `partial/tp` instead of the full gradient.** Install
the 1.6 hook, which sums first so the average is correct. Not run.

The docs page `docs/code-docs/source/autoep.rst` still says "AutoEP currently cannot be combined with
AutoTP"; the code at this revision supports folding (tests `tests/unit/v1/moe/test_autoep_autotp_*.py`).

### 3.6 Checkpoints with AutoEP [verified; not run]

- `has_moe_layers` switches `save_checkpoint` to `_save_moe_checkpoint` (`engine.py:4599-4606`,
  `4725+`).
- Non-expert state goes to `mp_rank_XX_model_states.pt`.
- Each global expert goes to its own file `layer_{i}_expert_{e}_mp_rank_XX_model_states.pt`
  (`engine.py:4052-4061`), written by the rank with EDP rank 0 for its experts.
- The checkpoint stores `ds_autoep_layers` metadata.
- A normal load needs the same `autoep_size`. Changing it goes through the universal checkpoint
  (docs `autoep.rst`, "Constraints").

### 3.7 Smallest random Qwen3-MoE [ran]

```python
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
cfg = Qwen3MoeConfig(vocab_size=64, hidden_size=32, intermediate_size=64, moe_intermediate_size=16,
                     num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                     num_experts=4, num_experts_per_tok=2, norm_topk_prob=True,
                     decoder_sparse_step=1, mlp_only_layers=[0], tie_word_embeddings=False)
cfg._attn_implementation = "sdpa"
model = Qwen3MoeForCausalLM(cfg)   # build through the model: Qwen3MoeExperts uses torch.empty
```
- **`mlp_only_layers` makes layers dense.** A layer is MoE iff
  `idx not in mlp_only_layers and num_experts > 0 and (idx+1) % decoder_sparse_step == 0`
  (`SP/transformers/models/qwen3_moe/modeling_qwen3_moe.py:309-314`). Dense layers use
  `Qwen3MoeMLP(intermediate_size)`.
- **Build through `Qwen3MoeForCausalLM`**: `Qwen3MoeExperts` allocates with `torch.empty`
  (`modeling_qwen3_moe.py:218-219`), and the router weight starts at zeros (`:256`), so only the model's
  weight initialization gives real values.
- The HF block is dropless too: it loops over the experts that received tokens (`:228-246`).

### 3.8 Failure modes to test (AutoEP)
- Preset without the `moe_layer_pattern` override. Expect "no MoE layers detected" (`auto_ep.py:579-597`).
- `top_k` missing.
- `route_norm` ≠ `norm_topk_prob`. Expect silent parity loss.
- `autoep_size` that does not divide `num_experts`, or exceeds it (`auto_ep_config.py:273-289`).
- `world % ep != 0`.
- ZeRO-1/2 with an unsplit client optimizer.
- `use_grouped_mm=true` without `torch._grouped_mm`.
- A folding preset mismatch. Folding with ep=1.
- Checkpoint reload with a different `autoep_size`.
- A router tie under bf16.

---

## 4. Accumulating N microbatches under ZeRO 0/1/2, then reducing and stepping once

### 4.1 What `gas=1` does today [verified + ran]

- The boundary check is `(micro_steps + 1) % gas == 0` unless overridden (`engine.py:3113-3129`), so with
  gas=1 **every backward is a boundary**. `micro_steps` increases only in `step()` (`engine.py:3354`).
- At a boundary:
  - ZeRO-0 all-reduces `.grad` (`engine.py:2768-2774`). This is linear, so accumulating after each
    reduction still gives the right sum: correct, with N allreduces.
  - ZeRO-1 reduces, then its epilogue **assigns** `averaged_gradients[i] = …` and clears `.grad`
    (`stage_1_and_2.py:900-950`).
  - ZeRO-2 reduces on every backward regardless and keeps a running sum in `all_grad_tensors`, but at a
    boundary it assigns `averaged_gradients[i]` and resets `all_grad_tensors[i] = None`
    (`stage_1_and_2.py:918-939`).
  - So under ZeRO-1/2 each boundary **replaces** the previous microbatch's gradient.
- [ran, DP=2, N=3, fp32 SGD: ZeRO-0 error 7e-9; ZeRO-1 and ZeRO-2 error 4.0e-2, i.e. last microbatch
  only.]

### 4.2 The public API to use [verified + ran]

`engine.set_gradient_accumulation_boundary(is_boundary)` (`engine.py:3131-3155`). Its docstring describes
exactly this use (flag `False` for the first N-1 backwards, `True` for the last, then one `step()`).
DeepSpeed's own tests use this pattern with gas=1 (`tests/unit/v1/zero/test_zero2_offload_multi_backward.py:64-80`).

With the flag:
- The engine forwards it to the optimizer before each backward (`engine.py:2826-2827`).
- Non-boundary backwards:
  - ZeRO-0 and ZeRO-1 only accumulate locally (`engine.py:2766-2776`; ZeRO-1's per-parameter hook
    reduces only at a boundary, `stage_1_and_2.py:1681-1683`).
  - ZeRO-2 reduces each microbatch and adds it into `all_grad_tensors` (`stage_1_and_2.py:918-925`).
- The boundary backward produces the complete sum.
- `step()` applies it because `is_gradient_accumulation_boundary()` returns the override
  (`engine.py:3267`).
- [ran: ZeRO-0/1/2 errors ≤ 7.5e-9 fp32. It also held with 1F1B interleaving (fwd0, fwd1, bwd0, fwd2,
  bwd1, bwd2), non-leaf detached inputs and `(out*grad).sum()` backward, fp32 ≤ 7.5e-9. bf16 ≤ 9.8e-4,
  which is at bf16 rounding level.]

Details that matter for rdsp:
- **Set the flag before every backward**, including `True` on the last. It persists until changed.
- Keep `gradient_accumulation_steps=1` and do the 1/N scaling ourselves. With gas>1, DeepSpeed divides
  every output gradient by gas via hooks on the forward outputs (`engine.py:2698-2703`, `2857-2863`), and
  `step()` would have to be called after every backward to advance `micro_steps`. That is why gas=N plus
  one `step()` does **not** work.
- `engine.forward` calls `optimizer.clear_backward_seen_flag()` on every forward (`engine.py:2679-2681`).
  It clears `grad_accum` only if an epilogue ran; under ZeRO-1 that happens only at a boundary
  (`stage_1_and_2.py:952-968`). So interleaved forwards are safe [ran].
- Alternatives:
  - `engine.no_sync()` works for ZeRO-0/1 only; it asserts under ZeRO-2 (`engine.py:2894-2911`).
  - `engine.coalesce_grad_reduction()` is ZeRO-1/2/3 only; it rejects ZeRO-0 and BF16 wrappers and is a
    context manager spanning all backwards (`engine.py:2914-2970`). Awkward across Ray calls.
  - The boundary flag works for all three stages.

### 4.3 Adapter change

```python
def backward(self, mb, grad=None):
    inp, out = self._acts.pop(mb)
    self.engine.set_gradient_accumulation_boundary(self._backwards == self.n_mb - 1)
    if self.is_last:
        self.engine.backward(self._losses.pop(mb) / self.n_mb)
    else:
        self.engine.backward((out * grad.to(self.device)).sum())
    self._backwards += 1
    ...
```
`apply()` stays `engine.step()`, which is only valid after the boundary backward. `ready()` already
guarantees that.

### 4.4 `reset()` under ZeRO-2 [ran]

After an abandoned step, `engine.zero_grad()` plus `optimizer.zero_grad()` leaves the ZeRO-2 running
sum in place. The next step included it (error 5.0e-2). Also clear:
```python
for k in list(getattr(self.engine.optimizer, "all_grad_tensors", {}) or {}):
    self.engine.optimizer.all_grad_tensors[k] = None       # private attribute; ZeRO-2 running sum
```
[ran: error 0 after the fix.] Private attribute: pin it with a test.

Why this is enough:
- `averaged_gradients` is reassigned at the next boundary, so it needs no clearing.
- ZeRO-0/1 were already correct.
- `BF16_Optimizer.zero_grad` clears its fp32 gradients (`deepspeed/runtime/bf16_optimizer.py:491-493`).

---

## 5. Combining features

| combination | status | source |
|---|---|---|
| TP + DP, ZeRO 0/1/2 | supported | mesh `(dp,tp)`, `groups.py:136-171`; [ran] ZeRO-1/2 |
| TP + ZeRO-3 | rejected | `engine.py:630-631` |
| TP + Ulysses SP | **do not combine** [inferred] | AutoTP overwrites `self.mpu` with `groups` (`engine.py:633`), which drops the SP mpu; folding forbids it explicitly (`auto_ep_folding.py:210-212`) |
| SP + DP, ZeRO 0/1/2 | supported [ran] | SP groups consecutive, ZeRO over whole stage |
| EP + DP, ZeRO 0/1/2 | supported [ran] | `groups.py:361-402` |
| EP + TP (folding) | supported by code, not run | `auto_ep_folding.py:184-262`; presets must match or one unset; ep>1; ZeRO≤2; no offload |
| EP + mpu-provided TP | rejected | `engine.py:546-552` |
| EP + SP | allowed by validation, **not run**; ZeRO-0 expert divisor uses the full world (`engine.py:3580`) while the ZeRO path uses `world/sp` (`stage_1_and_2.py:1356`), so check parity before use [inferred] | `engine.py:544`, `groups.py:447` |
| EP + ZeRO-3 | constrained (no TP, no SP, …) | `engine.py:1745-1790` |
| any + `gradient_clipping` default 1.0 | per-stage clipping ≠ global clipping | `constants.py:254` |

---

## 6. Engine factory

```python
import copy, types, torch

QWEN3_TP_SPECS = {"use_default_specs": False, "layer_specs": [
    {"patterns": [r".*\.self_attn\.o_proj\.weight$"],     "partition_type": "row"},
    {"patterns": [r".*\.self_attn\.[qkv]_proj\.weight$"], "partition_type": "column"},
    {"patterns": [r".*\.mlp\.down_proj\.weight$"],        "partition_type": "row"},
    {"patterns": [r".*\.mlp\.(gate|up)_proj\.weight$"],   "partition_type": "column"},
]}

def make_engine(stage_module, conf, *, dp, tp=1, sp=1, ep=1,
                hf_config=None, rows_per_rank=1, backend="nccl"):
    """One stage-local engine. torch.distributed is already initialized (world = this stage).
    Layouts: tp -> TP groups consecutive, DP strided.   sp -> SP groups consecutive.
             ep -> EP groups consecutive, EDP strided, dense DP = whole stage (dp == world).
             tp>1 and ep>1 -> Parallel Folding (dp*tp == world, ep | world)."""
    import deepspeed
    import deepspeed.comm as dscomm
    world = torch.distributed.get_world_size()
    if tp > 1 and sp > 1:
        raise ValueError("AutoTP and Ulysses SP cannot share a stage")
    if ep > 1 and tp == 1 and sp == 1 and dp != world:
        raise ValueError("an EP stage is data-parallel over the whole stage: dp must equal world")
    if not (ep > 1 and tp == 1 and sp == 1) and dp * tp * sp != world:
        raise ValueError(f"dp*tp*sp={dp*tp*sp} != stage world {world}")
    if ep > 1 and world % ep:
        raise ValueError(f"ep={ep} must divide stage world {world}")

    conf = copy.deepcopy(conf)
    conf["gradient_accumulation_steps"] = 1        # rdsp accumulates; see Section 4
    conf.pop("train_batch_size", None)             # DeepSpeed's 'world' differs per feature; let it derive
    conf.setdefault("gradient_clipping", 0.0)      # per-stage clipping is not global clipping
    mpu = None

    if tp > 1:
        conf["tensor_parallel"] = {"autotp_size": tp, "partition_config": QWEN3_TP_SPECS}

    if sp > 1:
        from deepspeed.runtime.sequence_parallel.ulysses_sp import UlyssesSPAttentionHF
        dscomm.init_distributed(dist_backend=backend, dist_init_required=False)  # required first
        sp_mpu = UlyssesSPAttentionHF.register_with_transformers(
            types.SimpleNamespace(config=hf_config),
            core_attn_implementation=hf_config._attn_implementation,   # e.g. "sdpa"
            sequence_parallel_size=sp, micro_batch_size=rows_per_rank,
            seq_length_is_variable=True)
        mpu = types.SimpleNamespace(**{k: getattr(sp_mpu, k) for k in dir(sp_mpu) if not k.startswith("__")})
        mpu.get_model_parallel_rank = lambda: 0        # checkpoint fix, Section 2.8
        mpu.get_model_parallel_world_size = lambda: 1

    if ep > 1:
        conf["expert_parallel"] = {
            "enabled": True, "autoep_size": ep, "preset_model": "qwen3_moe",
            "moe_layer_pattern": r"layers\.\d+\.mlp",
            "top_k": hf_config.num_experts_per_tok,
            "route_norm": bool(hf_config.norm_topk_prob),
            "use_grouped_mm": torch.cuda.is_available(),
        }
        # Put the optimizer in conf["optimizer"] (DeepSpeed then builds MoE param groups itself),
        # or pass a callable that uses split_params_into_different_moe_groups_for_optimizer.

    engine, _, _, _ = deepspeed.initialize(model=stage_module, config=conf, mpu=mpu,
                                           dist_init_required=False)   # no model_parameters
    if tp > 1:
        install_qk_norm_tp_grad_sum(engine)            # Section 1.6
    return engine
```

Per-feature contract for the adapter and coordinator:

| feature | input to rank r | output | input grad | terminal loss seed |
|---|---|---|---|---|
| DP (ZeRO 0/1/2) | own rows | own rows | exact for own rows | mean over own rows / N |
| TP | identical within TP group | replicated | replicated, complete | same on all TP ranks |
| SP | same rows, sequence chunk r, `position_ids=` global offsets, row count == `rows_per_rank` | chunk r | exact dL/d(chunk r) | local token-loss sum / global token count / N |
| EP (no TP) | own rows (like DP over the whole stage) | own rows | exact for own rows | mean over own rows / N |
| Folding | identical within TP group, different across dense-DP | replicated in TP group | [inferred] replicated | as TP |

---

## 7. Not verified

- Nothing ran on GPU/NCCL, flash attention, or `torch._grouped_mm`. All runs used CPU/gloo, 2 processes,
  fp32 (bf16 only for Section 4).
- Not run: AutoTP with ZeRO-0; AutoTP with dp>1 (the hook fix was run with tp=2, dp=1); bf16 parity for
  TP/SP/EP.
- Not run: Parallel Folding, EP+SP, the effect of folding on `q_norm`/`k_norm`, AutoEP checkpoint
  save/load, universal-checkpoint resharding (TP/EP/SP), and ZeRO-3 with any of these features.
- Inferred only: that TP and SP cannot share a stage (from `engine.py:633`); the reason DeepSpeed's
  Ulysses test passes despite the sp× factor; and that `ColumnParallel` would be wrong for
  non-contiguous gradients.

---

## Appendix A: running DeepSpeed multi-rank tests on CPU (used above)

The runs used the checkout at rev `53a2ac44` (put it on `PYTHONPATH`), gloo, and `torch.multiprocessing`
spawn with 2 ranks.

- `DS_ACCELERATOR=cpu PYTHONPATH=<checkout> .venv/bin/python script.py`
- In each rank, before `deepspeed.initialize`:
  `import deepspeed.comm.torch; sys.modules["deepspeed.comm.torch"].build_shm_op = lambda: None`.
  Without it the CPU accelerator tries to JIT-build a shared-memory op and fails without `ninja` on
  PATH. Note that `import deepspeed.comm.torch as x` binds the *torch package*, so use `sys.modules`.
- ZeRO with a plain `torch.optim.SGD` needs `"zero_allow_untested_optimizer": true`. Use SGD with
  clipping off: Adam's step ignores a uniform gradient scale, so it would hide the sp×/partial-gradient
  errors above.
- Checkpoint reload from a source checkout fails the ZeRO-1 version check (`__version__` is `0.0.0`).
  Patch `deepspeed.runtime.engine.version` and `deepspeed.runtime.zero.stage_1_and_2.version` in tests.
  An installed wheel does not have this problem.

Results:

| experiment | result |
|---|---|
| Q4, gas=1 without the flag | ZeRO-0 7e-9; ZeRO-1/2 4.0e-2 |
| Q4, with the flag | ≤ 7.5e-9 |
| 1F1B + flag + VJP backward | fp32 ≤ 7.5e-9; bf16 ≤ 9.8e-4 |
| ZeRO-2 abandoned step | 5.0e-2 → 0 with the `all_grad_tensors` clear |
| AutoTP, default path | AssertionError |
| AutoTP, preset/partition_config | outputs, input grads and split-weight grads ≤ 2e-6 |
| AutoTP `q_norm`/`k_norm` | ~1.0 relative error → 4e-7 with the hook |
| SP, tutorial loss | 2.000× the true gradient |
| SP, local-sum loss | 1.000× (2.8e-5) |
| SP, layer slice, ZeRO-0/1/2 | params 7.2e-7, input grads 1.8e-7 |
| AutoEP, ep=2, layer slice, ZeRO-0/1/2 | 2.5e-5 |
| SP checkpoint without the wrapper | IndexError on SP rank 1 |
| SP checkpoint with the wrapper | exact reload |
| TP checkpoint | exact reload |
