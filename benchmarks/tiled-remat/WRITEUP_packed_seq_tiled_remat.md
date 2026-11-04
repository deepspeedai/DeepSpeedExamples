# Packed-sequence tiled rematerialization

*Cut mixer rematerialization at physical `mbs=1` when a long sequence is several independent shorter samples.*

Implemented and measured. Batch-axis `TiledRematerialize` is still a no-op at `mbs=1`. Packed tiling is a separate path. On 2026-10-03 it ran on 8×H200, ZeRO-2 GPU Adam, FlashAttention, Liger fused cross-entropy, `fla`, and TiledMLP shards=4 (`autorun/logs/job-20261003T220933Z.log`).

## Contribution

Three claims, and no more.

1. **After TiledMLP, packed length still grows the checkpoint peak, and `ds_async` removes most of that growth.** Equal 4k samples, one physical row. Qwen3.5-4B checkpointing goes 20.3 → 61.7 GiB from 8k to 128k. `ds_async` goes 20.3 → 42.9 GiB. Qwen3.5-9B at 128k is 93.8 GiB checkpointing and 62.8 GiB `ds_async`. Qwen3.5-27B is flat at 112.8 GiB through 16k. At 64k, packed checkpointing OOMs and `ds_async` fits at 117.0 GiB.
2. **Packed whole-sample tiling stacks with offload. It is a memory cut, not a throughput win.** Budget 8192. Qwen3.5-4B at 128k: checkpointing 61.7 GiB and 9917 tok/s/gpu, `ds_async` 42.9 GiB and 10161, tiling 53.7 GiB and 8857, tiling plus `ds_async` 36.2 GiB and 8401. Qwen3.5-9B at 128k: 93.8 / 62.8 / 81.8 / 53.8 GiB. Qwen3.5-27B at 64k: checkpointing and tiling alone OOM; `ds_async` 117.0 GiB; tiling plus `ds_async` 112.9 GiB. The 128k `ds_async` cells for 27B did not return a peak (`rc=1`).
3. **The cut stops at a sample boundary.** A 32k row of 1k+1k+30k stays at 27.3 GiB checkpointing and 27.2 GiB with the 8192 budget, because the 30k sample is one tile. Batch-axis tiling now composes with TiledMLP: `ds_grad_is_ready` stays false until both the inner MLP shard and the outer batch tile are last. The Oct 3 Qwen3.5 microbatch cells predate that gate. The Oct 4 Qwen3 cells are the measured composition.

Blog 1 can show the packed 4k ladder and the skewed-pack negative. The paper's result is claim 2: the combination is what fits 27B at 64k, where checkpointing does not. Do not claim a tokens/s lead, and do not claim the batch-axis arm on this baseline.

## Why this is the remaining gap

After FlashAttention / GDN, Liger fused-CE, TiledMLP, and gradient checkpointing, gc peak is still **floor + store + FFN workspace + mixer workspace**. TiledMLP owns FFN. Offload owns store. Mixer remat is what is left at packed `mbs=1`.

On Qwen3.5-4B @ 32k, with **TiledMLP off**, wrap workspace minus an FFN architecture estimate (`~3·B·S·I + B·S·H` bf16) is **~14.3 vs ~1.8 GiB per extra microbatch**. That 14.3 is a **remainder** (GDN/FA plus residuals/norms in the wrap), not a profiler split of mixer tensors. With TiledMLP on, FFN is already token-tiled; packed-`mbs=1` OOM is then mixer remainder growing with packed length. Do not scale the 14.3/1.8 mbs-slope as if it were a measured mixer-only `mbs=1` sequence slope.

Batch-axis remat is a no-op at physical `mbs=1`. Logger-off GBS=16: `mbs=1` does **not** match larger mbs on tok/s (**~5–12%**, `mbs=2×gas=8` vs bigger mbs). Z3 comm is negligible vs GAS at **fixed** mbs; that does not make small mbs free. Packed `mbs=1` is a **capacity / packing** operating point, not a throughput win over tiled `mbs>1`.

## The independence that can exist

Packing concatenates independent samples into one token stream, usually `[1, sum(S_i), H]`, with boundaries in pack metadata (`cu_seqlens` or HF `cu_seq_lens_q/k`: `[0, S_0, S_0+S_1, …]`).

A tile is a **contiguous group of whole packed samples**, not a cut through one sample. That is legal for FlashAttention varlen only after slicing tokens **and** rebasing every associated argument (cumulative lengths, `max_seqlen_*`, sequence-major masks, `position_ids`). It is legal for GDN only if recurrent **and** short-conv state already reset at every document boundary — **require and verify per model**, do not assert. A packed GDN conv that slides across the row will change numerics when tiles force a reset; that may be a packed-baseline bugfix, not “unchanged semantics.”

## Idea

Keep the remat contract, change the shard axis:

- first forward can stay full packed length (same varlen kernels as today) or use `tile_forward`;
- backward rematerializes **one whole-sample tile** at a time;
- ZeRO `compute_params` stay gathered across the tile loop;
- `ds_grad_is_ready` is True only on the **last mixer tile** (invariant, not a vs-GAS win).

Steps:

1. Read pack metadata (allowlist below) and group **whole samples** into tiles under a token budget or shard count.
2. Slice `hidden_states` on the sequence axis for that tile (`[:, start:end, :]` or `[sum(S), H]`).
3. Rebase that tile's cumulative lengths so they start at 0 and end at the tile token count; recompute `max_seqlen_*`.
4. Run the **mixer only**. Concatenate outputs in sample order.

Peak **mixer** workspace then tracks the **largest tile**, not `sum(S_i)`. Peak step is still `max(fwd workspace, tiled bwd, keep-last saved input)`. Default full-S first forward is the same packed shape as gc (unlike unpack, which can raise peak via pad). If mixer **forward** OOMs, `tile_forward` is required. `tile_forward` does **not** shrink the concatenated output or the keep-last packed row; offload still owns store.

## In scope / out of scope

**In scope.** One physical row of many shorter independent samples, with pack metadata on every boundary. Mixer modules only.

**Out of scope.** Splitting one contiguous sample (would reset GDN and change causal attention). Whole-decoder wrapping (nests TiledMLP; inner last-**token**-tile can set `ds_grad_is_ready` True on a non-last outer tile). KV / Cache objects.

If packing is a single sample, this is a no-op (negative control: must not silently change that run).

## What shrinks and what does not

| Term | Packed-sequence tiled remat |
|---|---|
| Mixer remat workspace | Yes — ~ largest tile, not full packed `S` |
| FFN workspace | No — TiledMLP / `sequence_tiled_compute` (sibling, not nested) |
| Residuals / norms on the packed row | No |
| Checkpoint store / keep-last input | No — full packed length unless offload |
| Floor | No |
| Semantics | Unchanged **iff** FA varlen + GDN state + GDN short conv already honor every boundary |

## Why not unpack to `mbs = N` (and why keep it as a baseline)

External unpack: split at boundaries, pad `[N, L_max, H]`, use **today's** dim-0 remat. Zero new tiling code. It can win when `N>1`, `L_max ≪ T`, and `tile_forward=True`. Costs: pad tax on the saved residual, ranks must share a global `N` or shard agreement hangs, GDN still needs boundary resets, varlen kernels are abandoned. Packed-sample tiling has to **beat this baseline** on peak for skewed packs; “unpadding wastes padding” is not enough to dismiss it.

## Implementation notes

Pack metadata is first-class, not a denylist substring. Allowlist (reject anything else): `cu_seqlens`, `cu_seq_lens_q`, `cu_seq_lens_k`, `max_seqlen_q/k`, `max_length_q/k`, `seq_idx`, `position_ids`, sequence-major `attention_mask`. Rebase / slice each; do not pass the global table into a token slice.

**Cross-rank tile count.** Do not reuse `_agree_rematerialize_shards` keyed on `(shards, B=1, world)`. Ranks do not share a packing histogram. A cached all_reduce on different keys hangs; MAX shards can exceed a rank's sample count and create empty tiles. Open choice: (a) collate so every rank has the same sample count / global token budget decided once per microstep, (b) uncached agree on a common tile count bounded by **min samples across ranks**, each rank partitioning its own samples into that many nonempty groups, (c) confirm whether per-tile collectives are only the last-tile grad reduce (if order is per-param, not per-tile, different local tile counts might be legal). Empty tiles stay forbidden.

**TiledMLP.** Sibling wrap on FFN only. If a future decoder wrap appears, last-**outer**-tile owns `ds_grad_is_ready`; add a nested reduction test before claiming compose.

A sample longer than the token budget makes the budget soft (that sample is the tile). Dummy work to pad tile counts is a last resort.

## What would count as a win

Versus **packed `mbs=1` gc** (TiledMLP on, no mixer tiling) — primary:

- **Memory.** Mixer slice of peak falls with the token budget. Long packed context that OOMs today should fit if no packed sample exceeds the budget **and** keep-last/fwd peak is not already the cap.
- **Correctness.** Packed tiled vs packed untiled: outputs, input grads, parameter grads to usual remat tolerance. Unpacked `mbs=N` is **secondary** (different kernels, pad, possible GDN-conv numerics).
- **Throughput.** Capacity claim, not a tok/s win. Launch tax is expected; treat **>5% slower** than packed `mbs=1` gc as a failure to investigate, not an automatic pass.

A single indivisible sample that already OOMs remains the negative control.

## GDN state reset at packed boundaries (investigation)

*Answers the "require and verify per model" requirement above for Qwen3.5 GDN (`linear_attn`). Sources read: Transformers 5.16.1 wheel (the version the job scripts install), `fla-core` 0.5.2, `causal-conv1d` 1.7.0, and the local Transformers 4.57.6. Paths are relative to each package root.*

**Verdict.** Whole-sample tiling preserves numerics only if the *untiled* packed forward already resets both GDN states at every sample boundary. On Transformers 5.16.1 that is true only when `fla` is importable (recurrent state) **and** `causal_conv1d` is importable **and** the caller passes `seq_idx` (short conv). In every other case the packed baseline itself leaks state from one sample into the next. Tiling would then reset state at tile boundaries but not inside a tile, which matches neither the baseline nor the per-sample truth. That is a packed-baseline bug, not a tiling bug.

| State | Kernel resolved | Resets at boundaries? | Why |
|---|---|---|---|
| Recurrent | `fla` `chunk_gated_delta_rule` | Yes | receives `cu_seqlens`; varlen kernel runs per sequence from zero state |
| Recurrent | torch fallback `torch_chunk_gated_delta_rule` | **No** | `cu_seqlens` is filtered out before the call; one scan over the whole row |
| Short conv | `causal_conv1d.causal_conv1d_fn` + `seq_idx` | Yes | conv taps are masked across a `seq_idx` change |
| Short conv | `causal_conv1d` without `seq_idx` | **No** | Transformers never derives `seq_idx` from `cu_seq_lens_q` |
| Short conv | torch fallback (`F.conv1d`) | **No** | slides across the whole row unconditionally |

### How the call reaches the kernels

- `Qwen3_5DecoderLayer.forward` does `hidden_states = self.linear_attn(hidden_states=..., cache_params=past_key_values, attention_mask=..., **kwargs)` (`models/qwen3_5/modeling_qwen3_5.py:771`). That is a **bare tensor**, while attention is unpacked as `hidden_states, _ = self.self_attn(...)` (`:779`). `Qwen3_5GatedDeltaNet.forward` ends `return output` (`:546`). Top-level `cu_seq_lens_q/k`, `max_length_q/k` and `seq_idx` (a documented `TransformersKwargs` field, `utils/generic.py:853-867`) all ride `**kwargs` down to the mixer.
- The conv is called as `causal_conv1d_fn(mixed_qkv, weight, bias, activation=..., **kwargs)` (`modeling_qwen3_5.py:473`). The scan is called as `torch_chunk_gated_delta_rule(..., cu_seqlens=kwargs.pop("cu_seq_lens_q", None), **kwargs)` (`:522-531`).
- Both are decorated with `use_kernel_func_from_hub_with_fallback`. That decorator imports the original package (`fla`, `causal_conv1d`) when it can, and otherwise uses the torch function. It then does `applicable_params = tuple(inspect.signature(implementation).parameters)` (`integrations/hub_kernels.py:850`) and `kwargs = {k: v for k, v in kwargs.items() if k in applicable_params}` (`:860`). The filter works on **names**, so an implementation's `**kwargs` catch-all only admits a key literally called `kwargs`.

### Recurrent state

- **fla path.** `fla/ops/gated_delta_rule/chunk.py:397-414`: `chunk_gated_delta_rule(..., cu_seqlens=None, ...)` names `cu_seqlens`, so it survives the filter. It requires batch 1 when `cu_seqlens` is set (`:546-551`). The state kernel indexes every sequence `i_n` by its own `bos, eos = cu_seqlens[i_n], cu_seqlens[i_n+1]` (`fla/ops/common/chunk_delta_h.py:79-80`) and loads `h0` only when an initial state is supplied (`:114-115`, `:130`). Qwen3.5 passes `initial_state=None` in training, so every sample starts from zero. A tile call with rebased `cu_seqlens` is the same computation on a sub-range.
- **Torch path.** `torch_chunk_gated_delta_rule(query, key, value, g, beta, chunk_size=64, initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=False, **kwargs)` (`modeling_qwen3_5.py:249-259`) does not name `cu_seqlens`, so the filter drops it silently. A single state is zero-initialised once (`:302`) and carried through every chunk of the whole row (`:310-318`).

### Short-conv state

- **causal_conv1d path.** `causal_conv1d_fn(x, weight, bias=None, seq_idx=None, initial_states=None, ...)` (`causal_conv1d/causal_conv1d_interface.py:96-105`) resets only through `seq_idx`. It survives the filter only if the caller put `seq_idx` in kwargs. Nothing in Transformers builds it from `cu_seq_lens_q`; the only mention of `seq_idx` in the package is the `TransformersKwargs` docstring/field.
- **Torch path.** The fallback `causal_conv1d_fn(hidden_states, weight, bias=None, activation=None, **kwargs)` (`modeling_qwen3_5.py:219-239`) is a `F.conv1d(padding=K-1)` over the whole row, so the first `K-1` tokens of each sample see the tail of the previous sample.
- **Padding mask is not a boundary signal.** `create_recurrent_attention_mask` returns `None` for an all-ones mask (`masking_utils.py:1496-1502`), so packing without padding passes no mask to GDN at all.

### Empirical confirmation (CPU, no fla / causal_conv1d)

The real 5.16.1 `Qwen3_5GatedDeltaNet` was run on a two-sample packed row `[5 | 7]` with `cu_seq_lens_q`, and compared against each sample run alone. The output for sample A was identical (max |diff| 0.0). Sample B differed by **0.64** max |diff|, i.e. state leaked from A into B. With the two kernels swapped for boundary-resetting reference implementations, registered through the real `use_kernel_func_from_hub_with_fallback` decorator, the packed run matched per-sample runs to 2e-7. A 3-layer `Qwen3_5TextModel` then gave tiled-vs-untiled outputs identical (0.0) and input/parameter grads within 6e-11. (Ad hoc probe; it needs the 5.16.1 wheel on `PYTHONPATH`, so it is not a unit test.)

### Other versions

- **Qwen3-Next on 5.16.1 and Qwen3.5-MoE** use the same decorated pattern (`models/qwen3_next/modeling_qwen3_next.py:345,374,669-682`; `models/qwen3_5_moe/modeling_qwen3_5_moe.py:220,249,516-529`), so the same analysis applies.
- **Qwen3-Next on the local 4.57.6** never resets: it hard-codes `seq_idx=None` in the conv call (`modeling_qwen3_next.py:712`) and calls `self.chunk_gated_delta_rule(...)` without `cu_seqlens` (`:739-747`), even with `fla` and `causal_conv1d` installed. That version has no `qwen3_5` module.

### What the harness does about it (`autorun/bench_qwen3_offload.py`)

- `_gdn_boundary_reset_gaps(mixer_cls)` reads the module-level `torch_chunk_gated_delta_rule` / `causal_conv1d_fn` of the mixer's own module and pulls `applicable_params` out of the HF wrapper's closure (`_hf_kernel_applicable_params`). It reports a gap when `cu_seqlens` / `seq_idx` would be dropped. It also fails closed when no such wrapper exists, for example on 4.57.6 Qwen3-Next.
- `_install_packed_seq_tiled_rematerialize` **raises at install time** for any `linear_attn` class that has gaps, rather than tiling a baseline that is already wrong. At call time it requires `seq_idx` for GDN mixers, checks it against `cu_seq_lens_q`, and regenerates it per tile from the rebased `cu_seqlens`, for every mixer type. It also rejects a live cache object instead of silently replacing it with `None`, and returns whatever container the unwrapped mixer returned: a bare tensor for GDN, `(out, None)` for attention.
- For GDN models the packed input path now also passes `seq_idx` at `model(...)`, so the *untiled* baseline resets the conv too. If the kernels cannot reset, it prints a `WARNING packed_seq: ... leaks state` line. The untiled baseline still runs, because peak memory and throughput are still measurable, but its numerics are not trustworthy.
- Tests in `tests/unit/ulysses_alst/test_tiled_compute.py`: `PackedGDNMix` mimics this contract, with module-level kernels behind the same name-based kwarg filter.
  - `test_packed_gdn_mix_baseline_resets_at_boundaries` checks the oracle.
  - `test_packed_seq_installer_mixer_return_types` checks the return shapes.
  - `test_packed_seq_installer_refuses_non_resetting_gdn` checks the guards.
  - `TestPackedSeqTiledRematerializeGDN` checks ZeRO-2/3 parity and the last-tile `ds_grad_is_ready` invariant.

### Residual risk

- The guard reasons about which implementation the HF wrapper resolved. With `kernels` installed and the model kernelized (`use_kernels=True` / `kernelize`), hub kernels (`kernels-community/fla`, `kernels-community/mamba-ssm`) can replace these functions through `use_kernelized_func`. That path was not inspected (`kernels` is not installed here), so whether it keeps `cu_seqlens` / `seq_idx` is unverified. The SM121-only `Atlas-Inference/gdn` whole-layer kernel is likewise uninspected.
- GPU check (`autorun/parity_packed_gdn.py`, job `20261002T200252Z`, H200, fla-core 0.5.2, causal-conv1d 1.7.0): a 3-layer bf16 `Qwen3_5TextModel` of only `linear_attention` layers, pack lengths 20+28+12, token budget 24 (3 tiles). Untiled packed vs each sample alone was 0. Tiled vs untiled outputs were 0; the worst parameter grad differed by 4.8e-7 (`conv1d.weight`). Full-attention layers were not in that model.
- Under ZeRO-3 the bench leafs name-matched `GatedDeltaNet` classes and every mixer class `_install_packed_seq_tiled_rematerialize` patched, for the same per-tile `ds_id` reason as `_install_tiled_rematerialize`. A full-size model has not been measured.
