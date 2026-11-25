# rdsp

Pipeline-parallel training in which one CPU process hands each GPU its work and every GPU trains its share of the model with DeepSpeed. This glossary fixes the words we use for rdsp and for comparing it with Megatron-Bridge and MegatronMIMO.

## Language

### Processes

**Driver**:
The single CPU process that plans each training step and gives every GPU its work. It holds no model weights and no activations.
_Avoid_: controller, master, launcher

**Stage worker**:
The Ray actor that owns one GPU and trains that GPU's part of one stage.
_Avoid_: worker, actor, rank (a rank is a number in a group, not the process)

### Pipeline layout

**Stage**:
A consecutive slice of the model, together with the GPUs that train it. The first stage can also hold the vision encoder, and the last stage holds the loss.
_Avoid_: partition, pipeline rank, module (MegatronMIMO uses "module" for a whole sub-model)

**Stage world**:
The `torch.distributed` group of one stage's GPUs. Tensor and data parallelism happen inside it, and it does not know that other stages exist.
_Avoid_: process group, mesh

**Cross-stage groups**:
The two NCCL groups, `fwd` and `bwd`, that span every GPU and carry only boundary tensors.
_Avoid_: P2P world, pipeline group

**Boundary tensor**:
The activation sent forward to the next stage, or its gradient sent back, for one microbatch.
_Avoid_: message, payload

**Cut**:
The number of decoder layers each stage holds, written in stage order, e.g. 13 | 19.
_Avoid_: split, partition, boundary

**Balanced cut**:
rdsp's default cut. It gives each stage an equal share of the estimated compute, and that estimate includes the vision encoder.
_Avoid_: auto cut, smart cut

**Measured cut**:
A balanced cut redone from what a few profiled steps measured: each stage's time per step, and its memory as a limit on how many layers it may gain.
_Avoid_: calibrated cut, tuned cut

**Even cut**:
A cut with the same number of decoder layers on every stage. Megatron uses it by default, and it does not count the vision encoder.
_Avoid_: uniform split, default split

### Vision placement

**Shared first stage**:
The vision encoder runs on the first stage's GPUs, ahead of that stage's decoder layers.
_Avoid_: colocated, encoder on stage 0

**Vision-only stage**:
The vision encoder is a stage of its own and holds no decoder layers.
_Avoid_: non-colocated, separate encoder

**Replicated encoder**:
Every GPU holds a copy of the vision encoder outside the pipeline. Each copy encodes a share of the images and sends the features to the first stage.
_Avoid_: colocated vision, colocated

### Parallelism inside a stage

**DP copy**:
One of a stage's data-parallel replicas. It holds the stage's full slice of the model (split across its TP GPUs) and trains on different rows from the other copies.
_Avoid_: replica, DP rank

### Training step

**Row**:
One training sample (an image, its prompt and its answer) as a single token sequence.
_Avoid_: sample, example, sequence

**Step**:
One optimizer update, covering every row of the batch.
_Avoid_: iteration

**Microbatch**:
The group of rows that moves through the pipeline as one unit.
_Avoid_: chunk, minibatch

**Microbatch count**:
How many microbatches make up one step. It sets the size of the bubble.
_Avoid_: accumulation steps, number of microbatches

**Microbatch size**:
The number of rows in one microbatch, counted across all DP copies of a stage.
_Avoid_: batch size, micro batch size per GPU

**Job list**:
The ordered forwards and backwards that one stage runs in one step, delivered to it in a single call.
_Avoid_: op list, command list, schedule

**1F1B**:
The job-list order in which a stage runs its warm-up forwards and then alternates one forward with one backward.
_Avoid_: interleaved schedule, PipeDream

**Bubble**:
The share of a step during which a stage sits idle, waiting for a neighbouring stage.
_Avoid_: pipeline stall, idle time

**Apply**:
The optimizer update at the end of a step. It starts on any stage only after every stage has finished all its backwards.
_Avoid_: optimizer step, update

### Model

**GDN layer**:
A Gated DeltaNet decoder layer: Qwen3.5's linear-attention layer type. Qwen3.5-4B has 24 of them next to 8 full-attention layers.
_Avoid_: linear layer, DeltaNet, SSM layer

**Remainder master**:
Exact fp32 master weights kept as the bf16 weights plus the 16 bits the rounding dropped, 2 bytes per parameter instead of 4.
_Avoid_: bf16 master (that one is not exact), compressed master

**Untied output layer**:
The last stage's own copy of the output-layer weight, which the original model shares with the input embedding on stage 0.
_Avoid_: lm head copy, duplicated embedding

### Failure

**Rebuild**:
Recovery that replaces every stage worker and reloads the last checkpoint, while the driver and the Ray cluster keep running.
_Avoid_: restart, failover, elastic recovery

### Data and loss

**Replayed data**:
The rows that Megatron-Bridge exported, given to rdsp in the same order and the same microbatch groups, so that both systems train on identical input.
_Avoid_: shared dataset, same data

**Real token**:
A position that holds an actual token of the row: an image, prompt or answer token.
_Avoid_: valid token, non-pad token

**Answer token**:
A real token from the row's answer. It is the only kind of token with a label, so it is the only kind that counts in the loss.
_Avoid_: caption token, supervised token, label token

**Padding**:
Positions added after a row's last real token so that the rows in a microbatch share one length. They are computed, but they never count in the loss.
_Avoid_: pad tokens, filler

**Positions per real token**:
The number of positions computed, divided by the number of real tokens. It measures how much compute goes to padding.
_Avoid_: padding ratio, efficiency

**Token-mean loss**:
A step's loss taken as the plain average over every answer token in the step, whichever microbatch or GPU holds the token.
_Avoid_: per-token loss, global mean

**Microbatch-mean loss**:
A step's training loss taken as the average of each rank-microbatch's own answer-token average: each data-parallel rank's share of a microbatch is averaged on its own. Tokens in rank-microbatches with fewer answer tokens count for more. The reported loss is still the token mean.
_Avoid_: per-microbatch loss, mean of means

### Comparison

**Recipe comparison**:
A benchmark with the same layout, data, batch, microbatch count, optimizer and loss on both systems. Each system keeps its own padding, cut and kernels.
_Avoid_: same settings, apples-to-apples, fair comparison

**Equal-work comparison**:
A recipe comparison in which both systems also compute the same token positions.
_Avoid_: same settings, apples-to-apples
