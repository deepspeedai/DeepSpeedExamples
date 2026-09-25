# Activation / checkpoint hidden-state offload

Compares `use_pin_memory` True vs False on `CheckpointHiddenStatesOffload`
with **async / side streams held on**. This is not the async-vs-blocking
table from DeepSpeed #8282.

See the [parent README](../README.md).
