# RDT (Ray Direct Transport) for rdsp: research notes

Status: research only. No rdsp source was changed. Everything below was checked against Ray **2.58.0**, which `rdsp/uv.lock` pins (line ~1016, `name = "ray"`, `version = "2.58.0"`). The installed wheel is at `rdsp/.venv/lib/python3.13/site-packages/ray` (`_version.py`: commit `cec0a091b12675bbf101cf8a350c8706107baee1`). In this document, "source" means that wheel, with paths relative to `site-packages/ray/`.

Every claim is marked with how it was checked:

- **[src]**: read in the 2.58.0 source.
- **[doc]**: stated in the Ray docs or a blog post.
- **[ran]**: reproduced locally with a small CPU/Gloo script. Scripts are in the session scratchpad, not the repo. They ran under the parent venv `Deepspeed_Ray_PP/.venv`: Ray 2.58.0, torch 2.13.0, macOS. The `rdsp/.venv` lacks numpy, so Ray would not start there.
- **[unverified]**: not checked. Treat it as a hypothesis.

---

## 0. TL;DR

- **RDT stands for Ray Direct Transport.** Nothing in Ray is called "Ray Data Transfer". Earlier Ray releases called the feature "GPU objects". The code lives in `ray.experimental.rdt` [src]. The stability label is **alpha** [doc][src].
- **What it does:** when an actor method returns torch tensors, Ray keeps them on the producing actor's device. Only a small handle travels through Ray. When that handle is passed as an argument to another actor's task, Ray moves the tensor bytes directly between the two actors over NCCL, Gloo, NIXL or CUDA IPC. Nothing goes through the plasma object store.
- **Today rdsp does GPU→CPU→object store→CPU→GPU on every boundary.** `DeepSpeedStageAdapter.forward/backward` return `.detach().cpu()`. The receiving stage calls `.to(self.device)`. The refs travel between actors as keyword arguments (`upstream=`, `grad=`).
- **Three blockers apply to rdsp as it is written today** [ran]:
  1. **An RDT ref passed as a keyword argument is never transferred.** The receiving task hangs until the 60 s RDT timeout and then fails. rdsp passes `upstream`/`grad` as kwargs, so they must become positional arguments.
  2. **The Gloo transport cannot coexist with rdsp's stage-local `torch.distributed` process group.** `create_collective_group(..., backend="gloo")` fails with `ValueError: the new group's world size should be less or equal to the world size set by init_process_group`. So there is no RDT path on CPU. The CPU tests must stay on the object store.
  3. **The driver cannot `ray.get` an NCCL/Gloo RDT object** without `_use_object_store=True`. rdsp's coordinator only resolves terminal losses and control results, so this is fine as long as the terminal forward does *not* use a tensor transport.
- **NCCL transport on GPU looks viable.** It uses Ray's own cupy-based NCCL communicator, separate from DeepSpeed's torch NCCL group [src]. It needs `cupy` installed, and all actors of all stages must be in **one** collective group [src][doc].
- **The Anyscale multimodal-training repo does not actually use RDT for its cross-model transfers** at HEAD `5f456e6` (2026-01-13). It sets `enable_tensor_transport=True` and creates an NCCL collective group. But no method has `tensor_transport=...`, so the tensors still go through the object store, or through hand-rolled CUDA IPC when actors share a GPU. Details in §3.

---

## 1. What RDT is, and its API in Ray 2.58

### 1.1 Name and status

- The docs page is titled "Ray Direct Transport (RDT)" [doc]: https://docs.ray.io/en/latest/ray-core/direct-transport/direct-transport.html. The 2.58 docs moved it to this sub-path; the older URL `.../ray-core/direct-transport.html` 404s for 2.58.
- API reference [doc]: https://docs.ray.io/en/latest/ray-core/api/direct-transport.html
- Anyscale blog, "Ray Direct Transport: RDMA Support in Ray Core (Part 1)" [doc]: https://www.anyscale.com/blog/ray-direct-transport-rdma-support-in-ray-core. It dates the release to Ray 2.51.1. Search snippets of the 2.50 docs describe it as "formerly GPU objects" and "as of Ray v2.50". **[unverified]** which exact release first shipped it.
- Stability: every public RDT function is `@PublicAPI(stability="alpha")` (`experimental/rdt/util.py`, `experimental/rdt/rdt_manager.py`, `experimental/collective/collective.py`) [src]. The docs warn: "Future releases may introduce breaking API changes" [doc].
- Other things called "RDT": a web search found nothing else in the Ray ecosystem. Unrelated uses of the acronym exist, such as the "Robotics Diffusion Transformer" (RDT-1B) model. **[unverified]** beyond a search.

### 1.2 Transports available in 2.58

`experimental/rdt/util.py`, `DEFAULT_TRANSPORTS = ["NIXL", "GLOO", "NCCL", "CUDA_IPC"]` and `_ensure_default_transports_registered()` [src]:

| Transport | Devices | One- or two-sided | Needs a collective group? | Notes |
|---|---|---|---|---|
| `NCCL` | cuda | two-sided (Ray runs a matching `__ray_send__` on the sender and `__ray_recv__` on the receiver) | **yes** | Uses `ray.util.collective`'s `NCCLGroup`, which **requires cupy** (`util/collective/collective_group/nccl_collective_group.py`: `try: import cupy`; otherwise `_NCCL_AVAILABLE=False`) [src]. |
| `GLOO` | cpu | two-sided | **yes** | Uses `torch.distributed` gloo and **initializes the default process group** if none exists (`torch_gloo_collective_group.py: TorchGLOOGroup.__init__`) [src]. Incompatible with rdsp (§2.6). |
| `NIXL` | cuda, cpu | one-sided (the receiver pulls) | no | `pip install nixl`. Uses UCX, or LIBFABRIC on AWS EFA. The docs suggest `UCX_TLS=all` off EFA [doc]. Supports driver `ray.get`, `ray.put(_tensor_transport="nixl")`, and receive into preallocated buffers or onto another device (`set_target_for_ref`, `set_target_device_for_ref`) [src]. |
| `CUDA_IPC` | cuda | one-sided | no | **Same node and same physical GPU only.** The receiver must have been given the sender's GPU by Ray (`cuda_ipc_transport.py: recv_multiple_tensors` raises otherwise) [src]. Only useful for co-located actors that share a GPU. Not useful for rdsp's one-GPU-per-stage layout. |

Custom transports can be added with `ray.experimental.register_tensor_transport(name, devices, TensorTransportManager subclass, data_type)`. This must be called before the actors are created [src][doc]: https://docs.ray.io/en/latest/ray-core/direct-transport/custom-tensor-transport.html

### 1.3 API surface (exact names in 2.58)

**Actor class option.** `@ray.remote(enable_tensor_transport=True)` [src: `_common/ray_option_utils.py:245`; `actor.py:_process_option_dict` ~L1513]. It is switched on automatically when any method is decorated with `tensor_transport=`. Side effect: it adds concurrency groups `_ray_system` and `_ray_system_error` of size 1 each. The source comment says this "forces Ray to execute all tasks on background threads instead of the main thread" [src]. A probe confirmed that the actor's tasks ran on thread `Dummy-1`, not `MainThread` [ran].

**Per-method, at definition time.** `@ray.method(tensor_transport="nccl" | "gloo" | "nixl" | "cuda_ipc")`. The value is case-insensitive and normalized to upper case [src: `actor.py:method` ~L627–740]. The docstring reads: "Ray will store a *reference* instead of a copy of any `torch.Tensors` found inside values returned by this task."

**Per-call, at submit time.** `actor.method.options(tensor_transport="nccl").remote(...)` [src: `actor.py` ~L977–984 and `ActorMethod._remote` ~L1091–1184]. It is validated at submit time:
- If `num_returns != 1`: `ValueError: ... only support 1 return value` [src].
- If the actor class lacks `enable_tensor_transport`: `ValueError` [src].
- If NCCL/Gloo is requested but the actor is not in a group: `ValueError: ... please create a communicator with ray.experimental.collective.create_collective_group` [src].

This per-call form suits rdsp best, because one `execute` method returns tensors for some commands and scalars or bools for others (§4).

**Collective groups** (`ray.experimental.collective`) [src: `experimental/collective/collective.py`]:
```python
from ray.experimental.collective import (
    create_collective_group, destroy_collective_group,
    destroy_all_collective_groups, get_collective_groups)

group = create_collective_group(actors, backend="nccl", name=None)  # ranks = list order
destroy_collective_group(group)  # or by name
```
- `backend` is `"nccl"`, `"gloo"`, or `"torch_gloo"` (an alias for `GLOO`) [src: `util/collective/types.py: Backend.__new__`].
- It blocks until every actor has run `ray.util.collective.init_collective_group` via `__ray_call__` [src].
- **"An actor can only participate in one collective group per backend at a time"** [src docstring + `RuntimeError` check] [doc].
- When a transfer is triggered, exactly one group must contain both sender and receiver. If there are two, the result is `ValueError: ... RDT objects only support one communicator` (`collective_tensor_transport.py: get_communicator_metadata`) [src].

**Other public helpers** (`ray.experimental`, all alpha) [src: `experimental/rdt/__init__.py`, `util.py`, `rdt_manager.py`]:
- `wait_tensor_freed(tensor, timeout=None)`: blocks until every ref to a tensor the actor returned is gone, so the actor can safely write to it again.
- `set_target_for_ref(ref, [buffers])` and `set_target_device_for_ref(ref, "cpu"|"cuda:0")`: receive into given buffers or onto a chosen device. One-sided transports only (NIXL).
- NIXL only: `register_nixl_memory`, `deregister_nixl_memory`, `register_nixl_memory_pool(size, device)`, `set_nixl_cuda_stream(stream)`.
- `ray.put(value, _tensor_transport="nixl")` works for one-sided transports only. For NCCL/Gloo it raises `ray.put is not supported for two-sided RDT transport` [src: `_private/worker.py` ~L845–857].
- `ray.get(ref, _use_object_store=True)` fetches an RDT object through the object store instead [src: `_private/worker.py` ~L2882, L2918].

### 1.4 How a caller hands a GPU-object ref to another actor

1. Driver: `r = A.f.options(tensor_transport="nccl").remote(x)`. The driver records `RDTMeta(src_actor=A, backend, ...)` for `r` (`RDTManager.add_rdt_ref`) [src].
2. On A, the return value is serialized with every `torch.Tensor` pulled out. The tensors stay in A's in-process `RDTStore` as the "primary copy". Only the CPU skeleton plus the shapes and dtypes go through the normal object path [src: `_private/serialization.py: serialize_rdt_objects`, `rdt_store.py`].
3. Driver: `B.g.remote(r)`. Inside `ActorMethod._remote.invocation`, the driver calls `rdt_manager.queue_or_trigger_out_of_band_tensor_transfer(dst_actor, args)` (`actor.py` L1153–1154) [src]. For NCCL/Gloo this submits `__ray_send__` to A and `__ray_recv__` to B. Both run on the actors' `_ray_system` concurrency group, which has one thread (`rdt_manager.py: trigger_out_of_band_tensor_transfer`) [src]. If A has not produced the object yet, the transfer is queued until it has.
4. On B, before `g` runs, argument deserialization waits, with a timeout, for the tensors to appear in B's local RDTStore. It then pops them and rebuilds the original value [src: `_private/worker.py: deserialize_objects` → `RDTManager.get_rdt_objects`].

---

## 2. Constraints and semantics that matter for rdsp

### 2.1 Only top-level positional arguments are transferred (confirmed by experiment)

In step 3 above, the trigger only inspects `args` (the positional tuple). It skips `kwargs` and refs nested inside containers [src: `actor.py` L1153–1154; `rdt_manager.py: queue_or_trigger_out_of_band_tensor_transfer` iterates `task_args` and skips non-`ObjectRef`s]. Results with Gloo on 2 actors [ran]:

| How the RDT ref was passed | Result |
|---|---|
| `b.consume.remote(ref)` (positional) | works: `('Tensor', 28.0)` in 0.6 s |
| `b.consume.remote(kw=ref)` (keyword) | **`RayTaskError(TimeoutError)` after the RDT fetch timeout** (8.6 s with `RAY_rdt_fetch_fail_timeout_milliseconds=8000`; the default is 60 s: `_private/ray_constants.py` L613) |
| `b.consume.remote([ref])` (inside a list) | **`NotImplementedError: Tensor transport metadata is not available ... at the time of borrowing ... see issue #59644`** |

The docs only say refs "can only be passed as direct arguments to other actor tasks" [doc]. They do not mention the keyword-argument trap.

**Consequence for rdsp:** `StageGroupClient.submit` currently passes `upstream=` and `grad=` as keyword arguments (`stage_worker.py` `submit`). This **must** change to a positional argument before RDT can work.

### 2.2 Data types and nested structures

- Only `torch.Tensor` objects go out-of-band. Everything else in the return value travels as normal pickled data [doc][src].
- **Nested return values work**: `{"a": tensor, "b": (tensor, 3)}` arrived intact [ran]. The docs show the same with a dict [doc].
- One return value per call (`num_returns=1`) [src].
- All tensors in one RDT object must share a device type. For CUDA IPC they must also be on the same GPU index [src: `collective_tensor_transport.py: extract_tensor_transport_metadata`, `cuda_ipc_transport.py`].
- A method with `tensor_transport` set that returns `None` or a Python float works. There is simply nothing to transfer [ran]. The driver still needs `_use_object_store=True` to `ray.get` it under NCCL/Gloo.
- **bf16 over NCCL:** the cupy-based NCCL wrapper maps `torch.bfloat16` only `if hasattr(nccl, "NCCL_BFLOAT16")`, i.e. only with a new enough cupy (`util/collective/collective_group/nccl_util.py` ~L71–73) [src]. **[unverified]** which cupy version is needed. Test bf16 explicitly on Modal.

### 2.3 Driver access

- `ray.get(ref)` on an NCCL/Gloo RDT object from the driver fails: `ValueError: ray.get is not allowed on RDT objects using the two-sided transport GLOO. Either use a one-sided RDT transport or pass _use_object_store=True` [ran][src].
- `ray.get(ref, _use_object_store=True)` works [ran]. NIXL supports a plain `ray.get` [doc][src].
- Only the process that created the collective group can submit tasks that pass RDT objects. That process "cannot serialize and pass RDT ObjectRefs to other Ray tasks or actors" [doc]. In rdsp the driver both creates the group and submits every command, so this holds.

### 2.4 Lifetime and garbage collection

- The sender keeps the primary copy, a *reference* to the tensor rather than a copy, until **the owner's ObjectRef goes out of scope**. The owner is the driver in rdsp. At that point the driver sends `__ray_free__` to the source actor [src: `rdt_manager.py: queue_or_free_object_primary_copy`, `free_object_primary_copy`].
- On the receiver, a received (non-primary) copy is **popped** from the RDTStore when the task argument is deserialized [src: `get_rdt_objects` → `wait_and_pop_object`]. After that it lives only as long as the receiving task holds it.
- **Mutability:** "RDT objects are mutable, meaning that Ray only holds a reference to the tensor and will not copy it until a transfer is requested" [doc]. If the producer changes the tensor in place before the transfer runs, the receiver sees the changed data. `wait_tensor_freed` exists to guard against this.
  - rdsp returns `out.detach()` and `inp.grad.detach()` and never writes to them in place, so this should be safe.
  - **[unverified]** whether DeepSpeed ever zeroes an input-leaf `.grad` in place before the transfer runs. The leaf is per-microbatch and dropped, so this looks unlikely.
- **Memory effect in rdsp:** the coordinator keeps every `fwd`/`bwd` handle in dicts until `_run` returns. With RDT, each stage's boundary gradients (`inp.grad`) therefore stay pinned on the GPU until the end of the step. With the object store they are freed right after `.cpu()`. Forward outputs share storage with `out`, which the adapter keeps until backward anyway, so they cost nothing extra. Fix: drop handles from `fwd`/`bwd` once their consumer has been submitted (§4.2).

### 2.5 Failures, timeouts, ordering, concurrency

- **Ordering:** all send and receive work for an actor runs on its `_ray_system` concurrency group, which has one thread. It runs in the order the owner (driver) submitted it [src: comment in `trigger_out_of_band_tensor_transfer`: "to ensure that all communication operations are executed in a global order"]. rdsp has a single driver submitting in worklist order. An 8-microbatch, 3-stage forward+backward chain, with everything submitted before any `get`, completed correctly on Gloo [ran]. **[unverified]** on NCCL with real GPUs.
- **Fan-out:** one ref passed to two receivers worked. That means two sends from one producer [ran]. This matters because rdsp currently feeds rank 0's output to *every* rank of the next stage (`StageGroupClient` returns `refs[0]`).
- **Timeouts:** `RAY_rdt_fetch_fail_timeout_milliseconds`, default 60000 [src]. A monitor thread on the driver watches every send and receive ref (`RDTManager._monitor_failures`) [src].
- **Actor, node or transport failure:**
  - NCCL/Gloo cannot abort a transfer (`can_abort_transport() = False`). On failure Ray **kills both the source and destination actors and destroys the collective group** (`RDTManager._abort_transport`) [src][doc].
  - An application-level exception in the producer just propagates to the tasks that depend on it [doc].
  - For rdsp, a transport failure mid-step becomes a dead-actor error, which rdsp reports as `StepFailed`. Recovery needs rdsp's rebuild path, and that path must also create a fresh collective group.
- **Async actors, concurrency, `await`:** "`await` on an RDT ref is temporarily not supported". For collective transports there is "no support for out-of-order actors (async actors or actors with `max_concurrency > 1`)" [doc]. rdsp's actors are synchronous with default concurrency, so this is fine.
- **Threading side effect:** because `enable_tensor_transport` moves user tasks off the main thread [src][ran], anything thread-local that `__init__` sets could differ in `execute`. For example, `torch.cuda.set_device(0)` is per-thread. With one visible GPU per actor, device 0 is the default anyway. **[unverified]** whether DeepSpeed relies on anything main-thread-only.

### 2.6 Coexisting with DeepSpeed's own `torch.distributed` group

rdsp's actors call `torch.distributed.init_process_group(backend, rank, world_size=spec.num_gpus, ...)` in `StageWorkerActor.__init__` to form a **stage-local** group.

- **Gloo RDT is incompatible** [src][ran]. `TorchGLOOGroup.__init__` only initializes the default group `if not dist.is_initialized()`. It then calls `dist.new_group(ranks=range(world_size))` on top of the existing default group. Here that is the stage-local group of size 1, so it fails with `ValueError: the new group's world size should be less or equal to the world size set by init_process_group`. The reverse order would also break: RDT would claim the default group first, and rdsp's `init_process_group` would then fail. So: **no RDT on the CPU/Gloo path.** Keep the object store there.
- **NCCL RDT should coexist** [src]. `ray.util.collective`'s `NCCLGroup` creates its own NCCL communicators through cupy, sets them up via a Ray-side rendezvous actor, and runs them on its own CUDA streams (`nccl_collective_group.py`). It never touches `torch.distributed`. Before each send or receive it makes its stream wait on the current stream, and it calls `tensor.record_stream` so the tensor is not freed while in use (`_point2point`, ~L663–675) [src].
  - **[unverified on GPU]**: running it alongside DeepSpeed's torch NCCL group in the same process, extra GPU memory for another set of NCCL communicators (created lazily per peer pair), and stream-ordering behaviour now that user tasks run on a non-main thread.
- **NIXL** does not involve `torch.distributed` at all [src]. It needs NIXL and UCX installed on the Modal image. **[unverified]**.

---

## 3. Does ray-project/multimodal-training use RDT between stages?

Repo: https://github.com/ray-project/multimodal-training, cloned at HEAD `5f456e6` (2026-01-13), which pins `ray[...]==2.51.0` in `requirements.txt`. Blog: https://www.anyscale.com/blog/30-faster-multimodal-ai-training-with-ray-and-disaggregated-hybrid. The blog says nothing about the transport [doc].

What the code does [src of that repo]:
- Both trainer actor classes are declared `@ray.remote(enable_tensor_transport=True, num_gpus=1, num_cpus=6)` (`python/ray/vision.py:660`, `python/ray/text.py:1253`).
- When the two models run on separate GPUs, `python/train_ray.py:260` calls `create_collective_group(vision_actors + text_actors, backend="nccl")`. This is one group spanning both models, as §1.3 requires.
- **But no method uses `tensor_transport=` or `.options(tensor_transport=...)`.** A grep for `tensor_transport` finds only the two `enable_tensor_transport=True` lines. `ActorGroup._execute_single_actor` is a plain `getattr(actor, name).remote(*args, **kwargs)` (`python/ray/actor_group.py:140–154`).
  - The vision `forward_step` returns a dict containing the CUDA tensor. The text `forward_step(vision_embeddings_ref, ...)` receives it through the normal object store.
  - The code path labelled "Ray Direct Transport" in `python/ray/tensor_transfer.py:196–200` just puts the tensor in the returned dict.
- When actors share a GPU, the repo uses hand-rolled **CUDA IPC** (`tensor_transfer.py: create_ipc_handle` / `reconstruct_tensor_from_ipc`, using `torch.multiprocessing.reductions.reduce_tensor` and an interprocess CUDA event). This is the same technique Ray later packaged as the `CUDA_IPC` transport.
- Its README says "Ray ObjectRefs handle cross-component data transfer automatically" (`README.md:126`).

**Conclusion:** at this commit the repo prepares for RDT (actor flag and a single NCCL group) but moves its inter-model activations and gradients through the object store or CUDA IPC. It does not use RDT. It is useful as a template for "one NCCL group across all actor groups" and for "each stage is its own torch.distributed world". It gives no evidence that RDT works well for this pattern. **[unverified]** whether a branch or later commit enables it.

---

## 4. Recommendation for rdsp

### 4.1 Decision

- Add an optional **activation transport** setting with values `"object_store"` (default, today's behaviour), `"nccl"`, and later `"nixl"`.
- `"nccl"` is allowed only when `use_gpu` is true. On CPU always use the object store (§2.6).
- Keep all tensor-transport knowledge inside `stage_worker.py`. The coordinator stays transport-agnostic and only benefits from releasing handles earlier.
- Start with NCCL. It needs no extra network stack beyond cupy, and it matches the single-node, multi-GPU Modal layout. Try NIXL afterwards if you want driver-side debugging with `ray.get` or cross-node RDMA.

Where the setting lives is a design decision:
- `config.py` says `PipelineConfig` is "Frozen at P3a", so adding a field means reopening that freeze.
- The alternative is a keyword argument on `create_stage_clients(..., transport=...)` threaded from `_default_coordinator_factory`, with an environment variable override for benchmarks.
- It must **not** go into the DeepSpeed config. `api.py` already reserves `"transport"` in `_RESERVED_CONFIG_KEYS`.

Dependency: add `cupy-cuda12x` (matching the CUDA version on Modal) to the `gpu` extra in `pyproject.toml`. **[unverified]** which CUDA major version the Modal image uses.

### 4.2 Code sketch (not applied)

**`deepspeed_adapter.py`**: stop moving boundary tensors to CPU when a device transport is active. The adapter stays Ray-free.
```python
class DeepSpeedStageAdapter:
    def __init__(self, ..., keep_boundary_on_device: bool = False):
        ...
        self._keep = keep_boundary_on_device

    def _out(self, t):
        t = t.detach()
        return t if self._keep else t.cpu()

    def forward(self, mb, x, labels=None):
        ...                                   # unchanged up to the return
        return self._out(out)                  # was: out.detach().cpu()

    def eval_forward(self, mb, x, labels=None):
        ...
        return self._out(out)

    def backward(self, mb, grad=None):
        ...
        return None if self.is_first else self._out(inp.grad)
```
`x.to(self.device)` and `grad.to(self.device)` are already no-ops when the tensor arrives on the right GPU.

**`stage_worker.py`**: the boundary tensor becomes the one **positional** argument.
```python
class StageWorkerActor:
    def __init__(self, ..., transport: str = "object_store"):
        ...
        self.adapter = DeepSpeedStageAdapter(
            ..., keep_boundary_on_device=(transport != "object_store"))

    # payload = upstream activation (forward/eval) or downstream grad (backward).
    # MUST stay positional: RDT only transfers refs that are top-level
    # positional args (see docs/research/RDT.md §2.1).
    def execute(self, command_id, kind, mb, payload=None, *,
                inputs=None, labels=None, control=None):
        self.executed.append(command_id)
        if kind == "forward":
            return self.adapter.forward(mb, inputs if payload is None else payload, labels=labels)
        if kind == "eval":
            return self.adapter.eval_forward(mb, inputs if payload is None else payload, labels=labels)
        if kind == "backward":
            return self.adapter.backward(mb, grad=payload)
        ...  # control kinds unchanged


class StageGroupClient:
    def __init__(self, actors, stage=0, *, n_stages=1, transport="object_store"):
        self.actors, self.stage = actors, stage
        self.n_stages, self.transport = n_stages, transport

    def _returns_boundary_tensor(self, kind):
        if self.transport == "object_store":
            return False
        if kind in ("forward", "eval"):
            return self.stage < self.n_stages - 1   # terminal returns a float loss
        if kind == "backward":
            return self.stage > 0                    # first stage returns None
        return False

    def submit(self, command, *, inputs=None, labels=None,
               upstream=None, grad=None, control=None):
        payload = _unwrap(upstream if upstream is not None else grad)
        rdt = self._returns_boundary_tensor(command.kind)
        refs = []
        for actor in self.actors:
            m = actor.execute
            if rdt:
                m = m.options(tensor_transport=self.transport)
            refs.append(m.remote(command.command_id, command.kind,
                                 command.microbatch, payload,
                                 inputs=inputs, labels=labels, control=control))
        ...  # handle construction unchanged


def create_stage_clients(model, plan, loss_fn, *, engine_factory=None,
                         use_gpu=None, stage_builder=None,
                         transport: str = "object_store"):
    import ray
    ...
    if transport != "object_store" and not use_gpu:
        raise ValidationError("device transports need GPUs; Gloo RDT conflicts "
                              "with the stage-local torch.distributed group")
    remote_opts = {"enable_tensor_transport": True} if transport != "object_store" else {}
    Actor = ray.remote(**remote_opts)(StageWorkerActor) if remote_opts else ray.remote(StageWorkerActor)
    ...  # build actors exactly as today, passing transport=transport
    ray.get(ready)
    if transport == "nccl":
        from ray.experimental.collective import create_collective_group
        all_actors = [a for c in clients for a in c.actors]
        # ONE group over every rank of every stage: an actor may be in only
        # one group per backend, and each sender/receiver pair must share
        # exactly one group.
        group = create_collective_group(all_actors, backend="nccl",
                                        name=f"rdsp-{uuid.uuid4().hex[:8]}")
        for c in clients:
            c.collective_group = group   # so shutdown() can destroy it
    return clients
```
`StageGroupClient.shutdown()` should call `destroy_collective_group(self.collective_group)` inside a try/except before `ray.kill`, once per group rather than once per stage. Otherwise every rebuild leaves stale group records in the driver.

**`coordinator.py`** (optional, transport-agnostic): release boundary handles as soon as their consumer has been submitted, so the sender's primary copy is freed. The terminal forward handles must be kept, because they hold the losses.
```python
# after submitting forward (s, k) with s > 0:
if s - 1 != terminal:
    fwd.pop((s - 1, k), None)
# after submitting backward (s, k) with s < terminal:
bwd.pop((s + 1, k), None)
```
This needs one adjustment: the dependency checks `(s - 1, k) not in fwd` currently also serve as "already submitted" markers. Keep a separate `submitted_fwd` set for those checks.

**Tests.**
- CPU: the CPU suite runs unchanged on `"object_store"`. Add a CPU unit test asserting that `submit` passes the boundary ref **positionally**. It can use a fake actor that records `args`/`kwargs`.
- GPU (Modal): parity test. Same seed, N steps, `object_store` vs `nccl`. Losses and final parameters should match exactly, since the transport must not change the numbers.

### 4.3 Benchmark plan: object store vs RDT

Measure at two levels, and keep numerics identical between the two runs.

**A. Micro-benchmark: one boundary transfer.** Two actors on two GPUs of the same Modal node.

- **Payload sizes:** use real boundary sizes for Qwen3-0.6B. The activation is `micro_batch × seq_len × hidden × 2 bytes` (bf16). The hidden size is 1024 per the HF config **[unverified here]**. That gives about 4 MiB at 1×2048 and 64 MiB at 8×4096. Sweep 64 KiB to 256 MiB.
- **Paths compared:**
  1. Object store, as today: `.cpu()` on the sender, then plasma put, get, and `.to(cuda)` on the receiver.
  2. RDT NCCL.
  3. RDT NIXL, if it installs.
  4. A no-tensor task, to measure the fixed Ray task overhead.
- **Metrics:**
  - Latency, measured from the moment the producer task returns to the moment the consumer has the tensor on its GPU. Use `torch.cuda.synchronize()` plus `time.perf_counter_ns()` stamps inside the actors, and report p50/p95 over ≥200 repetitions after warm-up.
  - Achieved bandwidth (GB/s).
  - The first-transfer cost, reported separately. NCCL communicators are created lazily per peer pair.
- **Breakdown for the object-store path:** time spent in `.cpu()` versus in Ray versus in `.to(device)`. This shows how much RDT can possibly save.

**B. End-to-end: `train_batch` on Qwen3-0.6B.**
- Setup: 2 and 4 stages, 1F1B, several microbatch counts (4, 8, 16) and sequence lengths.
- Metrics:
  - Step time (median and spread over ≥3 runs × ≥20 steps, excluding warm-up) and tokens per second.
  - Per-stage busy time versus idle time (pipeline bubble), from timestamps in `execute`.
  - Peak GPU memory per stage (`torch.cuda.max_memory_allocated`). RDT keeps boundary tensors on the GPU longer (§2.4). Measure with and without the early handle release.
  - Object store usage (`ray memory` / plasma bytes) and driver CPU time.
  - Correctness: loss curves and final parameters must match between transports.
- Report the **fraction of step time spent in boundary transfers** under each transport. If the object-store path is already a small share of step time at these sizes, RDT's gain is capped by that share (Amdahl's law), however fast RDT itself is.

**C. Robustness checks to run once.**
- Kill a stage actor mid-step. Expect `StepFailed` within the RDT timeout rather than a hang, and a clean `load_checkpoint` rebuild that includes a new collective group.
- bf16 over NCCL with the chosen cupy version.
- A DP>1 stage. Today's rank-0 fan-out means one sender feeds K receivers.

---

## 5. Open questions and unverified items

1. Whether RDT NCCL runs cleanly in the same process as DeepSpeed's torch NCCL group. The source suggests yes. Not run on a GPU.
2. The cupy version needed for bf16 over NCCL (§2.2), and the CUDA major version of the Modal image.
3. Whether NIXL has the same keyword-argument limitation. The trigger code is shared, so probably yes. Not run: NIXL was not installed.
4. Side effects of moving actor tasks off the main thread (§2.5) on DeepSpeed and CUDA device state.
5. The exact Ray release in which RDT first shipped (2.50 vs 2.51.1 in different sources).
6. Whether a later commit of multimodal-training enables `tensor_transport`.
7. rdsp design point, outside RDT: every rank of stage s+1 receives **rank 0's** output of stage s (`StageGroupClient` returns `refs[0]`). With RDT this becomes a K-way fan-out from one GPU. Whether data-parallel ranks should instead pair up (rank i sends to rank i) is a separate decision.
