# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""The Ray actor behind one rank of one pipeline stage.

Each actor owns one GPU, one DeepSpeed engine (through DeepSpeedStageAdapter)
and its links to neighbouring stages. A training step arrives as one call,
run_step(), carrying the rank's whole 1F1B op list; activations and gradients
then move rank to rank over the p2p groups, never through the driver.
"""

import collections
import contextlib
import json
import os
import socket
import time
import traceback

import torch

from ray_deepspeed_pipeline import checkpoint as ckpt
from ray_deepspeed_pipeline.boundary import (
    Grid,
    assemble,
    assemble_boundary,
    gradient_scale,
    p2p_destinations,
    p2p_sources,
    rank_cell,
    rank_coords,
)
from ray_deepspeed_pipeline.deepspeed_adapter import DeepSpeedStageAdapter, observed_mesh
from ray_deepspeed_pipeline.errors import ValidationError
from ray_deepspeed_pipeline.hf_stage import load_meta_parameters
from ray_deepspeed_pipeline.losses import token_weight
from ray_deepspeed_pipeline.varlen_attention import register_if_used
from ray_deepspeed_pipeline.vision import vision_schedule

_VISION_FILE = "colocated_vision.pt"


_GRAD_KEYS = 1 << 20  # colocated vision: feature gradients' message numbers


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


class _PhaseTimer:
    """Wall time per phase of a step when RDSP_PROFILE=1 (or set_profile),
    printed as one JSON line per rank and kept in `records`. Pending compute
    is waited for before each reading, so a phase's time is its own;
    profiling makes steps slower."""

    def __init__(self, device):
        self.on = os.environ.get("RDSP_PROFILE") == "1"
        self.device = device
        self.totals = collections.defaultdict(float)
        self.records = []

    def _sync(self):
        # the compute stream only: a device-wide sync would also wait for this
        # rank's async sends, which finish only when the neighbour receives,
        # and two neighbours waiting on each other's sends deadlock
        if self.device.type == "cuda":
            torch.cuda.current_stream(self.device).synchronize()

    @contextlib.contextmanager
    def phase(self, name: str):
        if not self.on:
            yield
            return
        self._sync()
        start = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            self.totals[name] += time.perf_counter() - start

    def report(self, **where) -> None:
        if self.on:
            ms = {k: round(v * 1e3, 1) for k, v in self.totals.items()}
            if torch.cuda.is_available():  # GB now, and peak since the last report
                ms["mem_gb"] = round(torch.cuda.memory_allocated() / 2**30, 2)
                ms["peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
                torch.cuda.reset_peak_memory_stats()
            print(json.dumps({"rdsp_profile": where, **ms}), flush=True)
            self.records.append({**where, **ms})
        self.totals.clear()


class StageWorkerActor:
    """One rank of one stage-local world. Construction is cheap; bootstrap()
    joins the stage world once every rank exists."""

    def __init__(
        self,
        stage_module,
        ds_config_json: str,
        *,
        stage: int,
        rank: int,
        grid: Grid,
        backend: str,
        n_microbatches: int,
        is_first: bool,
        is_last: bool,
        loss_fn=None,
        engine_factory=None,
        weights: str | None = None,
        vision=None,
    ):
        self.stage, self.rank, self.grid = stage, rank, grid
        self.p2p = None
        self._p2p_store = None
        self.cell = rank_cell(grid, rank)
        self._init = dict(
            stage_module=stage_module,
            ds_config_json=ds_config_json,
            backend=backend,
            n_microbatches=n_microbatches,
            is_first=is_first,
            is_last=is_last,
            loss_fn=loss_fn,
            engine_factory=engine_factory,
            weights=weights,
            vision=vision,
        )
        self.adapter = None
        self.vision = None  # this rank's colocated vision encoder (vision.py)
        # the last two steps' command ids in execution order, for tests and diagnostics
        self.executed = collections.deque(maxlen=2 * (2 * n_microbatches + 2))
        self.generation = None

    def address(self) -> tuple[str, int, str]:
        """(host, free port, node id) for this stage's rendezvous; called on rank 0."""
        import ray

        return (
            ray.util.get_node_ip_address(),
            _free_port(),
            ray.get_runtime_context().get_node_id(),
        )

    def bootstrap(self, master_addr: str, master_port: int) -> dict | None:
        """Join the stage world and build the engine; returns observed_mesh()."""
        init = self._init
        os.environ["RANK"] = str(self.rank)
        os.environ["LOCAL_RANK"] = "0"  # Ray sets CUDA_VISIBLE_DEVICES per actor
        os.environ["WORLD_SIZE"] = str(self.grid.world)
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
        module = init["stage_module"]
        register_if_used(module)
        load_meta_parameters(module, init["weights"])
        if init["backend"] == "nccl":
            torch.cuda.set_device(0)
            module = module.to("cuda")
        torch.distributed.init_process_group(
            init["backend"],
            rank=self.rank,
            world_size=self.grid.world,
            init_method=f"tcp://{master_addr}:{master_port}",
        )
        self.adapter = DeepSpeedStageAdapter(
            module,
            init["ds_config_json"],
            init["n_microbatches"],
            init["is_first"],
            init["is_last"],
            loss_fn=init["loss_fn"],
            engine_factory=init["engine_factory"],
            input_grads=("pixel_values",) if init["vision"] is not None else (),
        )
        if init["vision"] is not None:
            self._build_vision(init["vision"], init["weights"], init["ds_config_json"])
        self._init = None  # drop the driver-built module copy
        self._timer = _PhaseTimer(self.adapter.device)
        return observed_mesh(self.adapter.engine)

    def _build_vision(self, vision, weights, ds_config_json):
        from ray_deepspeed_pipeline.partition import compile_blocks, recompute_blocks
        from ray_deepspeed_pipeline.vision import ColocatedVisionEngine

        tower, spec = vision  # the tower and its plan.VisionSpec
        register_if_used(tower)
        self._vision_per_microbatch = spec.encode_per_microbatch
        load_meta_parameters(tower, weights)
        if spec.recompute:
            recompute_blocks(tower)
        if spec.compile:
            compile_blocks(tower)
        self.vision = ColocatedVisionEngine(tower, json.loads(ds_config_json), self.adapter.device)

    def ping(self) -> bool:
        return True

    def set_profile(self, on: bool) -> None:
        """Time the phases of the steps from now on (as RDSP_PROFILE=1 does
        from the start), or stop; clears the kept records."""
        self._timer.on = on
        self._timer.totals.clear()
        self._timer.records.clear()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def profile_records(self) -> dict:
        """The phase timer's records since set_profile, and this GPU's memory."""
        total = (
            torch.cuda.get_device_properties(self.adapter.device).total_memory / 2**30
            if self.adapter.device.type == "cuda"
            else None
        )
        return {"records": list(self._timer.records), "total_gib": total}

    def has_loss_fn(self) -> bool:
        return self.adapter.loss_fn is not None

    def executed_commands(self):
        return list(self.executed)

    def named_parameters_numpy(self):
        return self.adapter.named_parameters_numpy()

    def _enter_generation(self, generation, *, begin: bool = True) -> None:
        # a new generation discards whatever an abandoned one left behind
        if generation == self.generation:
            return
        self.generation = generation
        if begin:
            self.adapter.begin_generation()
            if self.vision is not None:
                self.vision.reset()

    def _forward(self, kind: str, mb: int, x, labels, extras, loss_weight=1.0):
        offset = self.cell.sp_index * x.shape[1] if self.grid.sp > 1 else None
        run = self.adapter.forward if kind == "forward" else self.adapter.eval_forward
        return run(
            mb, x, labels=labels, position_offset=offset, extras=extras, loss_weight=loss_weight
        )

    def _receive_forward(self, sources, mb: int):
        """This rank's cell of the upstream hidden state and extras."""
        pieces = [(cell, self.p2p.recv_boundary("fwd", peer, mb)) for peer, cell in sources]
        return assemble_boundary(pieces, self.cell)

    def execute(self, command_id: str, kind: str, generation=None, control=None):
        """Run one control command: apply, save or load (verify or load)."""
        self.executed.append(command_id)
        self._enter_generation(generation, begin=kind != "load")
        if kind == "apply":
            with self._timer.phase("apply"):
                applied = self.adapter.apply()
                if self.vision is not None:
                    self.vision.apply()
            self._timer.report(stage=self.stage, rank=self.rank, kind="apply")
            return applied
        if kind == "save":
            return self._save(control["root"], control["tag"])
        if kind == "load" and "verify" in control:
            return self._verify(control["root"], control["tag"], control["verify"])
        if kind == "load":
            return self._load(
                control["root"],
                control["tag"],
                control.get("load_optimizer_states", True),
                control.get("load_lr_scheduler_states", True),
            )
        raise ValidationError(f"unknown command kind {kind!r}")

    # -- the step ----------------------------------------------------------------

    def init_p2p(
        self, host: str, port: int, epoch: int, grids: list, timeout_s: float = 300.0
    ) -> bool:
        """Join fresh cross-stage groups for `epoch` and warm up every
        neighbour link, so no connection setup happens inside a step."""
        from ray_deepspeed_pipeline.p2p import PipelineP2P, make_store

        grids = [Grid(*grid) for grid in grids]
        offsets = [0]
        for grid in grids:
            offsets.append(offsets[-1] + grid.world)
        self._grids, self._offsets = grids, offsets
        global_rank = offsets[self.stage] + self.rank
        if self._p2p_store is None:
            self._p2p_store = make_store(host, port, offsets[-1], global_rank == 0, timeout_s)
        if self.p2p is not None:
            # release unmatched sends of the failed step before NCCL's watchdog does
            self.p2p.abort()
        self.p2p = PipelineP2P(
            self._p2p_store,
            global_rank,
            offsets[-1],
            epoch,
            timeout_s,
            self.adapter.device,
            collective=self.vision is not None,
        )
        self._warm_up_links()
        return True

    def _peers(self):
        """(prev, next) neighbour links in global ranks; None at a pipeline end.
        prev = (activation senders as (rank, cell), gradient receivers);
        next = (activation receivers, gradient senders as (rank, cell))."""
        stage, grids, offsets = self.stage, self._grids, self._offsets
        prev = next_ = None
        if stage > 0:
            up, base = grids[stage - 1], offsets[stage - 1]
            prev = (
                [(base + r, cell) for r, cell in p2p_sources(up, self.grid, self.rank)],
                [base + r for r in p2p_destinations(self.grid, up, self.rank)],
            )
        if stage < len(grids) - 1:
            down, base = grids[stage + 1], offsets[stage + 1]
            next_ = (
                [base + r for r in p2p_destinations(self.grid, down, self.rank)],
                [(base + r, cell) for r, cell in p2p_sources(down, self.grid, self.rank)],
            )
        return prev, next_

    def _warm_up_links(self):
        prev, next_ = self._peers()
        dummy = torch.zeros(1, device=self.adapter.device)
        self.p2p.begin_step()
        if next_:
            for peer in next_[0]:
                self.p2p.send("fwd", dummy, peer, 0)
        if prev:
            for peer, _ in prev[0]:
                self.p2p.recv("fwd", peer, 0)
            for peer in prev[1]:
                self.p2p.send("bwd", dummy, peer, 0)
        if next_:
            for peer, _ in next_[1]:
                self.p2p.recv("bwd", peer, 0)
        self.p2p.end_step()

    def run_step(self, *args, **kwargs):
        """_run_step, under the torch profiler on the training step named by
        RDSP_TORCH_PROFILE (0-based), whose kernel table goes to
        /tmp/rdsp_torch_profile_stage<s>_rank<r>.txt."""
        at = os.environ.get("RDSP_TORCH_PROFILE")
        self._steps_run = getattr(self, "_steps_run", 0) + 1
        if at is None or self._steps_run - 1 != int(at):
            return self._run_step(*args, **kwargs)
        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            out = self._run_step(*args, **kwargs)
            torch.cuda.synchronize()
        path = f"/tmp/rdsp_torch_profile_stage{self.stage}_rank{self.rank}.txt"
        with open(path, "w") as f:
            f.write(
                prof.key_averages().table(
                    sort_by="cuda_time_total", row_limit=60, max_name_column_width=90
                )
            )
        return out

    def _run_step(
        self,
        generation: int,
        ops: list,
        train: bool,
        inputs=None,
        labels=None,
        vision=None,
        token_total=None,
        varying=False,
    ):
        """Run this rank's whole step. ops: [(kind, mb, command_id)] in 1F1B
        order. inputs / labels: this rank's cell of every microbatch (first /
        last stage only). vision: this rank's vision.route_images() entry
        under colocated vision. token_total: the step's token count under a
        TokenMeanLoss (last stage). varying: the microbatches differ in
        length, so every boundary travels with its shape."""
        self._enter_generation(generation)
        prev, next_ = self._peers()
        adapter, p2p, timer = self.adapter, self.p2p, self._timer
        # DeepSpeed averages gradients over DP and sums over SP, so the gradient
        # entering this stage must be dp_s x dL/d(output): scale by dp_s/dp_{s+1}
        scale = gradient_scale(self._grids[self.stage + 1].dp, self.grid.dp) if next_ else 1.0
        losses = {}
        loss_weight = (
            token_weight(token_total, self.grid.dp, adapter.n_mb)
            if token_total is not None
            else 1.0
        )
        p2p.begin_step(varying=varying)
        try:
            schedule, tail = {}, []
            if vision is not None:
                self._local_features, self._image_rows, self._mine = {}, [], {}
                if train and self._vision_per_microbatch:
                    schedule, tail = vision_schedule(
                        ops, set(vision["own_mb"].values()), len(self._grids)
                    )
                else:  # everything at once, backpropagated at the end
                    with timer.phase("vision_forward"):
                        self._encode_images(vision, train)
                    tail = sorted(set(vision["own_mb"].values()))
            ready = True
            for i, (kind, mb, command_id) in enumerate(ops):
                if i in schedule:
                    backpropagate, encode = schedule[i]
                    with timer.phase("vision_backward"):
                        self._vision_backward(vision, backpropagate)
                    with timer.phase("vision_forward"):
                        for m in encode:
                            self._encode_images(vision, train, m)
                self.executed.append(command_id)
                if kind == "ready":
                    ready = adapter.ready()
                    continue
                if kind in ("forward", "eval"):
                    if prev:
                        with timer.phase("wait_fwd"):
                            x, extras = self._receive_forward(prev[0], mb)
                    else:
                        x, extras = inputs[mb], None
                        if vision is not None:  # the features stand in for the pixels
                            with timer.phase("wait_features"):
                                features = self._features_for(vision, mb)
                            if features is not None:
                                x = dict(x, pixel_values=features)
                    mb_labels = labels[mb] if labels is not None else None
                    with timer.phase("forward"):
                        out = self._forward(kind, mb, x, mb_labels, extras, loss_weight)
                    if next_:
                        hidden, out_extras = out if isinstance(out, tuple) else (out, {})
                        for peer in next_[0]:
                            p2p.send_boundary("fwd", hidden, out_extras, peer, mb)
                    else:
                        losses[mb] = out
                else:  # backward
                    grad = None
                    if next_:
                        with timer.phase("wait_bwd"):
                            grad = assemble(
                                [(cell, p2p.recv("bwd", peer, mb)) for peer, cell in next_[1]],
                                self.cell,
                            )
                        if scale != 1.0:
                            grad = grad * scale
                    with timer.phase("backward"):
                        input_grad = adapter.backward(mb, grad=grad)
                    if prev:
                        for peer in prev[1]:
                            p2p.send("bwd", input_grad, peer, mb)
                    elif input_grad is not None and vision is not None and train:
                        # on its way at once: the owners backpropagate when
                        # their own pipeline work is done
                        self._send_feature_grads(mb, input_grad["pixel_values"])
            if vision is not None and train:
                with timer.phase("vision_backward"):
                    self._vision_backward(vision, tail)
                    # the first stage's gradients carry its data-parallel degree
                    self.vision.reduce_gradients(self.p2p.all, divide_by=self._grids[0].dp)
            with timer.phase("wait_sends"):
                p2p.end_step()
        except Exception:
            # printed at once: the driver may be waiting on another stage that
            # is itself waiting on this one, and would only report it later
            traceback.print_exc()
            raise
        finally:
            p2p.begin_step()  # drop per-step shape records either way
            timer.report(stage=self.stage, rank=self.rank, kind="step")
        return {
            "losses": [losses[k] for k in sorted(losses)] if adapter.is_last else None,
            "ready": ready,
        }

    # -- colocated vision ----------------------------------------------------------

    def _global_rank(self) -> int:
        return self._offsets[self.stage] + self.rank

    def _encode_images(self, vision: dict, train: bool, mb: int | None = None) -> None:
        """Encode this rank's images (of microbatch mb, or all) in one batch
        and send each image's features to the first-stage ranks of its row
        (over gloo: a send may wait for its receiver as long as needed
        without holding the GPU)."""
        me, p2p = self._global_rank(), self.p2p
        own = [e for e in vision["own"] if mb is None or vision["own_mb"][e[0]] == mb]
        features = self.vision.forward([(i, px, grid) for i, px, grid, _, _ in own], train)
        for image, _, _, dests, _ in own:
            for dest in dests:
                if dest == me:
                    self._local_features[image] = features[image]
                else:
                    p2p.send_cpu(features[image], dest, image)

    def _features_for(self, vision: dict, mb: int):
        """First stage: the features of microbatch mb's images, in row order,
        received just before its forward; None if it has no images."""
        me, parts = self._global_rank(), []
        for m, image, owner in vision["need"]:
            if m != mb:
                continue
            got = (
                self._local_features.pop(image) if owner == me else self.p2p.recv_cpu(owner, image)
            )
            parts.append(got)
            self._image_rows.append((mb, image, owner, got.shape[0]))
        return torch.cat(parts) if parts else None

    def _send_feature_grads(self, mb: int, grad) -> None:
        """First stage, TP rank 0, right after microbatch mb's backward:
        split its feature gradient per image and send each to its owner."""
        if rank_coords(self.grid, self.rank)[2] != 0:
            return
        me, entries = self._global_rank(), [e for e in self._image_rows if e[0] == mb]
        for (_, image, owner, _), part in zip(entries, grad.split([e[3] for e in entries])):
            if owner == me:
                self._mine[image] = part
            else:
                self.p2p.send_cpu(part, owner, _GRAD_KEYS + image)

    def _vision_backward(self, vision: dict, microbatches) -> None:
        """Receive the feature gradients of this rank's images in these
        microbatches and backpropagate them through its encoder."""
        if not microbatches:
            return
        me, p2p, mbs = self._global_rank(), self.p2p, set(microbatches)
        grads = {}
        for image, _, _, _, source in vision["own"]:
            if vision["own_mb"][image] not in mbs:
                continue
            grads[image] = (
                self._mine.pop(image) if source == me else p2p.recv_cpu(source, _GRAD_KEYS + image)
            )
        self.vision.backward(grads)

    # -- checkpoints -------------------------------------------------------------

    def _save(self, root: str, tag: str) -> dict:
        stage_dir = ckpt.stage_dir(root, tag, self.stage)
        self.adapter.save_shard(stage_dir, tag)
        self.adapter.save_rng(ckpt.rng_path(root, tag, self.stage, self.rank))
        if self.vision is not None:
            # every rank holds a slice of the encoder's optimizer state: all
            # gather it, and each stage keeps a full copy next to its shards
            state = self.vision.full_state()
            if self.rank == 0:
                torch.save(state, os.path.join(stage_dir, _VISION_FILE))
        # every rank's files are complete before rank 0 fingerprints the stage
        torch.distributed.barrier()
        # rank 0 fingerprints only what its node sees: a rank elsewhere must see the same directory
        record = {
            "rank": self.rank,
            "shares_dir": os.path.exists(ckpt.rng_path(root, tag, self.stage, 0)),
        }
        if self.rank == 0:
            record["files"] = ckpt.digest_tree(stage_dir)
        return record

    def _verify(self, root: str, tag: str, files_by_stage: dict):
        """Rank 0 re-checks this stage's shard digests on its own node.
        Returns True or the rejection text; never raises, so the reason
        reaches the driver without a transport-specific wrapper."""
        if self.rank != 0:
            return True
        files = files_by_stage.get(self.stage, files_by_stage.get(str(self.stage)))
        try:
            ckpt.verify_stage_files(ckpt.stage_dir(root, tag, self.stage), self.stage, files)
        except Exception as e:
            return str(e)
        return True

    def _load(
        self, root: str, tag: str, load_optimizer_states: bool, load_lr_scheduler_states: bool
    ) -> bool:
        self.adapter.load_shard(
            ckpt.stage_dir(root, tag, self.stage),
            tag,
            load_optimizer_states=load_optimizer_states,
            load_lr_scheduler_states=load_lr_scheduler_states,
        )
        self.adapter.load_rng(ckpt.rng_path(root, tag, self.stage, self.rank))
        if self.vision is not None:
            self.vision.load(
                os.path.join(ckpt.stage_dir(root, tag, self.stage), _VISION_FILE),
                load_optimizer_states,
                load_lr_scheduler_states,
            )
        return True
