# AutoTP equivalence check

Tensor parallelism partitions a layer's arithmetic across ranks. It is not
supposed to change what the layer computes: an AutoTP=3 run and an AutoTP=1 run
of the same model, on the same data, should train the same model.

Existing AutoTP smoke tests check that a sharded run exits cleanly. That is a
weak signal, because a shard cut on the wrong boundary also exits cleanly — it
just trains a different model. This example compares the runs' loss curves
instead, which distinguishes a correct partition from a plausible-looking wrong
one.

## The three configurations

| | Split of Qwen3-0.6B's 16 attention / 8 KV heads | Role |
|---|---|---|
| AutoTP=1 | not split | baseline |
| AutoTP=3 | **uneven** — 6/6/4 heads per rank | the case under test |
| AutoTP=4 | **even** — 4/4/4/4 heads per rank | control |

AutoTP=3 is the interesting one. Even splits divide every dimension exactly and
so hide off-by-one bugs; an uneven split is where a shard boundary is most likely
to be computed wrongly. It also forces the head count, rather than the raw
element count, to drive the split — partitioning `q_proj`'s 2048 output features
as 683/683/682 would land mid-head and corrupt the attention reshape.

AutoTP=4 exists to calibrate the answer. Because it divides evenly, whatever gap
it shows against the baseline is pure floating-point reassociation. If AutoTP=3
drifts no more than AutoTP=4 does, the uneven split is contributing no error of
its own — a much stronger statement than "AutoTP=3 stayed under some tolerance I
picked".

## Running it

```
cd training/autotp_equivalence

bash run_gpu.sh 500 0,1,2,3   # accelerators, NCCL
bash run_cpu.sh 500           # CPU, gloo
```

Each script runs `train.py` at all three widths and then diffs each sharded run
against the AutoTP=1 baseline with `compare_loss.py`. Set `MASTER_PORT` if you
want a CPU and a GPU run to proceed at the same time.

## What is pinned, and why

Anything that could make the runs diverge for a reason other than sharding is
fixed:

| | |
|---|---|
| **Data** | Batches come from a CPU generator seeded identically on every rank, so all ranks — and all runs — see the same tokens in the same order. AutoTP replicates the input across the tensor-parallel group, so ranks that disagreed on the batch would invalidate the comparison. |
| **Precision** | fp32. bf16 rounding noise is orders of magnitude larger than the reassociation error being measured, and would mask a real bug. |
| **Dropout** | Asserted to be zero at startup, so no RNG is consumed inside the forward. |
| **Parallelism** | `world_size` must equal `autotp_size`. Spare ranks would silently become a data-parallel dimension, averaging gradients over more samples and changing what is being compared. |
| **Threads** | The CPU runs cap `OMP_NUM_THREADS`, identically at every width. Without it each rank sizes its thread pool for the whole machine and several ranks oversubscribe it — 27s per step instead of 1.5s. |

The training data is random tokens, so the loss itself is meaningless. What
matters is only that differently-sharded runs agree on it.

## Reading the result

The runs are not expected to be bit-identical. Collective reductions sum partial
results in a different order than a single rank does, so the curves differ by
floating-point reassociation from the first step where a reduction feeds back
into the weights. What distinguishes that from a bug is its shape:

* **Reassociation error jitters.** It grows slowly and non-monotonically as the
  runs' weights drift within the same basin, and the control run shows the same
  amount of it.
* **A sharding bug compounds.** The runs are optimizing different models, so the
  gap grows steadily, does not come back, and the uneven split shows it while the
  even control does not.

`compare_loss.py` therefore checks every step rather than the final loss, prints
the whole trajectory (sampled, with the worst step always shown), and reports the
mean and worst relative gap.

The first step is checked separately and far more tightly (`--forward-rtol`,
default 1e-6). Both runs start from the same checkpoint and no optimizer step has
happened yet, so a gap there is a wrong forward, not accumulated drift — training
dynamics cannot be blamed for it.

## Measured results

500 steps, Qwen3-0.6B, fp32:

| Backend | Sharding | seq_len | step 0 | mean rel | worst rel |
|---|---|---|---|---|---|
| NCCL, GPUs | AutoTP=3 (uneven) | 128 | `0.00e+00` | `7.69e-05` | `2.52e-03` (step 467) |
| NCCL, GPUs | AutoTP=4 (even, control) | 128 | `0.00e+00` | `7.58e-05` | `2.72e-03` (step 467) |
| gloo, CPU | AutoTP=3 (uneven) | 64 | `6.82e-08` | `2.75e-05` | `1.15e-03` (step 379) |
| gloo, CPU | AutoTP=4 (even, control) | 64 | `6.82e-08` | `2.79e-05` | `2.39e-03` (step 379) |

The uneven split and the even control land on top of each other. Their means
agree to within a few percent, and on both backends the worst step is *the same
step* — 467 on GPU, 379 on CPU — which says the spike belongs to the training
trajectory at that point rather than to how the heads were divided. On CPU, and
at the worst step on GPU, the even control is in fact the *further* of the two
from the baseline.

So the answer is not merely "AutoTP=3 stayed under the tolerance". It is that
splitting 16 heads unevenly across 3 ranks costs nothing in accuracy beyond what
an evenly-divisible tensor-parallel run already costs.

Note that the GPU runs match exactly at step 0 while the CPU runs do not. gloo
reduces in a different order than a single rank does, so the sharded forward is
not bit-identical there — which is why the forward check uses a small tolerance
rather than demanding equality.

## Tests

```
cd training/autotp_equivalence
python -m pytest tests/ -v
```

The tests cover `compare_loss.py`: that it accepts reassociation-scale noise,
rejects a compounding gap, catches a wrong forward at the first step without
mistaking later drift for one, reports the worst step rather than the last, and
never hides the worst step when sampling a long run.
