# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""Runs one global step at a time over the stage workers; owns global_steps.

The coordinator never resolves tensors: only scalar losses and statuses. The
step protocol is a ready/apply barrier. A failure before any stage applies its
optimizer update raises StepFailed (retryable); a failure during apply poisons
the pipeline until load_checkpoint() restores it. Every attempt runs under a new
generation id, so results from failed attempts are never mistaken for current ones.
"""

import json
import os
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Protocol

import torch

from ray_deepspeed_pipeline import checkpoint as ckpt
from ray_deepspeed_pipeline.data import take_microbatch_entries
from ray_deepspeed_pipeline.errors import (
    CheckpointError,
    PipelinePoisoned,
    RdspError,
    StepFailed,
    ValidationError,
)
from ray_deepspeed_pipeline.plan import ExecutionPlan
from ray_deepspeed_pipeline.protocols import Command, resolve
from ray_deepspeed_pipeline.schedule import control_commands, generate_commands


def _shapes_differ(inputs) -> bool:
    """Whether a step's microbatches differ in shape (rows, or sequence
    length when each is padded to its own longest row); stages then send
    every boundary with its shape."""

    def shape(x):
        if isinstance(x, dict):
            x = x.get("input_ids", next((v for v in x.values() if torch.is_tensor(v)), None))
        return tuple(x.shape) if torch.is_tensor(x) else None

    return len({shape(x) for x in inputs}) > 1


class StageWorkerClient(Protocol):
    """One stage's endpoint (the stage's whole actor group).

    submit() returns a handle or future resolving to one stage-level result.
    A "step"/"eval_step" command carries the stage's whole op list in
    control["ops"] and resolves to {"losses": [...] (last stage) or None,
    "ready": bool}; apply/save/load resolve to a status or record.

    Also required: a static reconnect_p2p(clients, epoch) that rebuilds the
    stages' links after a failed step. Optional: alive() -> bool and
    shutdown() -> None, used by recovery.
    """

    def submit(self, command: Command, *, inputs=None, labels=None, control=None) -> Any: ...


class PipelineCoordinator:
    def __init__(
        self,
        plan: ExecutionPlan,
        workers: list[StageWorkerClient],
        *,
        rebuild: Callable[[], list[StageWorkerClient]] | None = None,
    ):
        """rebuild: recreates every stage's workers from scratch. load_checkpoint()
        needs it to recover from a dead worker."""
        if len(workers) != len(plan.stages):
            raise ValidationError(
                f"{len(plan.stages)} stages need {len(plan.stages)} "
                f"stage clients, got {len(workers)}"
            )
        self._plan = plan
        self._workers = workers
        self._rebuild = rebuild
        self._global_steps = 0
        self._generation = 0
        self._poisoned = False
        self._p2p_epoch = 0
        self._p2p_dirty = False  # a failed stage-local step left the p2p links unusable
        self._ahead = None  # (data iterator, future of the next step) under prefetch
        self._reader = None
        self._retry = None  # (data iterator, entries) of a step that raised StepFailed

    @property
    def global_steps(self) -> int:
        return self._global_steps

    @property
    def plan(self):
        """The ExecutionPlan this coordinator runs."""
        return self._plan

    def train_batch(self, data_iter: Iterator) -> float:
        return self._run(data_iter, train=True)

    def eval_batch(self, data_iter: Iterator) -> float:
        return self._run(data_iter, train=False)

    def save_checkpoint(
        self,
        save_dir: str,
        tag: str | None = None,
        *,
        client_state=None,
        data_position=None,
        save_latest: bool = True,
    ) -> bool:
        """Every rank of every stage saves; the manifest is published only after
        all of them succeed, so a failed save commits nothing.

        data_position: where the caller's data stands. Entries read but not
        trained on are taken off an engine-owned position's entries_consumed
        and recorded as read_ahead in a caller's: a failed step's, kept for
        its retry, and under prefetch the next step's M entries."""
        if self._poisoned:
            raise PipelinePoisoned(
                "refusing to checkpoint a poisoned pipeline: its stages may "
                "hold weights from different optimizer steps"
            )
        tag = ckpt.validate_tag(tag if tag is not None else f"global_step{self._global_steps}")
        held = self._entries_held()
        if held and data_position is not None and data_position.get("owner") == "engine":
            consumed = data_position["entries_consumed"] - held
            data_position = {**data_position, "entries_consumed": consumed}
        elif held:
            data_position = {**(data_position or {"owner": "caller"}), "read_ahead": held}
        if os.path.exists(os.path.join(save_dir, tag, ckpt.MANIFEST_NAME)):
            raise CheckpointError(f"checkpoint {tag!r} is already committed; tags are immutable")

        self._generation += 1
        cmds = control_commands(self._plan, self._generation, self._global_steps, "save")
        control = {"root": save_dir, "tag": tag}
        try:
            handles = {s: self._submit(c, control=control) for s, c in cmds.items()}
            records = [self._resolve(h, cmds[s]) for s, h in handles.items()]
        except RdspError as e:
            raise CheckpointError(
                f"save of {tag!r} failed; no manifest was published and the "
                f"last committed checkpoint is unchanged: {e}"
            ) from e

        manifest = ckpt.build_manifest(
            tag=tag,
            global_step=self._global_steps,
            plan_hash=self._plan.plan_hash(),
            plan_json=self._plan.canonical_json(),
            stage_records=records,
            data_position=data_position,
            client_state=client_state if client_state is not None else {},
        )
        ckpt.check_complete(manifest, list(range(len(self._plan.stages))))
        ckpt.publish(save_dir, manifest, update_latest=save_latest)
        return True

    def load_checkpoint(
        self,
        load_dir: str,
        tag: str | None = None,
        *,
        load_optimizer_states: bool = True,
        load_lr_scheduler_states: bool = True,
    ) -> dict:
        """Restore the whole pipeline from a committed manifest.

        Dead workers trigger a full rebuild; live ones reload in place. The
        pipeline is un-poisoned only after every rank of every stage has loaded.
        """
        manifest = ckpt.read_and_verify(
            load_dir,
            tag,
            plan_hash=self._plan.plan_hash(),
            plan_stages=list(range(len(self._plan.stages))),
        )
        self._drop_read_ahead()
        self._retry = None
        tag = manifest["tag"]

        # invalidates handles from failed generations
        self._generation += 1
        if not self._all_alive():
            self._poisoned = True  # fresh workers hold no committed state
            if self._rebuild is None:
                raise CheckpointError(
                    "a stage worker is unreachable and this pipeline has no "
                    "rebuild factory; cannot recover in place"
                )
            self._shutdown_workers()
            self._workers = self._rebuild()
            self._p2p_dirty = False  # fresh workers come with fresh groups
        # Two phases: every stage verifies its shards on its own node first, so
        # a rejected checkpoint leaves every worker untouched.
        files = {r["index"]: r["files"] for r in manifest["stages"]}
        self._load_phase(manifest, {"root": load_dir, "tag": tag, "verify": files}, "verification")
        self._poisoned = True  # until every stage has loaded
        self._generation += 1
        self._load_phase(
            manifest,
            {
                "root": load_dir,
                "tag": tag,
                "load_optimizer_states": load_optimizer_states,
                "load_lr_scheduler_states": load_lr_scheduler_states,
            },
            "load",
        )

        self._global_steps = manifest["global_step"]
        self._poisoned = False
        return {
            "tag": tag,
            "global_step": manifest["global_step"],
            "client_state": manifest["client_state"],
            "data_position": manifest["data_position"],
            "path": os.path.join(load_dir, tag),
        }

    def _load_phase(self, manifest: dict, control: dict, what: str) -> None:
        cmds = control_commands(self._plan, self._generation, manifest["global_step"], "load")
        try:
            handles = {s: self._submit(c, control=control) for s, c in cmds.items()}
            results = {s: self._resolve(h, cmds[s]) for s, h in handles.items()}
        except RdspError as e:
            raise CheckpointError(
                f"checkpoint {what} of {manifest['tag']!r} failed; the pipeline "
                f"stays poisoned until a load succeeds: {e}"
            ) from e
        for s, result in results.items():
            if isinstance(result, str):  # a stage's own rejection reason
                raise CheckpointError(result)
            if result is False:
                raise CheckpointError(f"stage {s} failed checkpoint {what}")

    def set_profile(self, on: bool) -> None:
        for w in self._workers:
            w.set_profile(on)

    def profile_records(self) -> list[list[dict]]:
        """Per stage, each rank's phase records and GPU memory (set_profile)."""
        return [w.profile_records() for w in self._workers]

    def shutdown(self) -> None:
        self._drop_read_ahead()
        self._shutdown_workers()

    def _all_alive(self) -> bool:
        return all(getattr(w, "alive", lambda: True)() for w in self._workers)

    def _shutdown_workers(self) -> None:
        for w in self._workers:
            shutdown = getattr(w, "shutdown", None)
            if shutdown is not None:
                shutdown()

    def _run(self, data_iter: Iterator, train: bool) -> float:
        if self._poisoned:
            raise PipelinePoisoned(
                "a previous generation partially applied optimizer updates; "
                "train/eval calls are rejected until load_checkpoint() restores "
                "the last committed global checkpoint"
            )
        prepared = None
        if train and self._retry is not None and self._retry[0] is data_iter:
            entries = self._retry[1]  # the failed step again, so the retry is exact
        elif train and self._ahead is not None:
            source, future = self._ahead
            if source is not data_iter:
                raise StepFailed(
                    "prefetch read this step's data ahead from another iterator; pass "
                    "the same iterator to every train_batch() call"
                )
            self._ahead = None
            entries, prepared = future.result()  # raises if the iterator ran out
        else:
            # consume exactly the step's microbatches before dispatching anything
            entries = take_microbatch_entries(data_iter, self._plan.global_microbatches)
        if train:
            self._retry = None
        inputs = [e[0] for e in entries]
        labels = [e[1] for e in entries]

        self._generation += 1
        sequences = generate_commands(self._plan, self._generation, self._global_steps, train=train)
        # a retry finds the next step already read ahead
        ahead = data_iter if train and self._plan.prefetch and self._ahead is None else None
        try:
            losses = self._run_step(sequences, inputs, labels, train, prepared, ahead)
        except StepFailed:
            if train:
                self._retry = (data_iter, entries)
            raise
        return self._finish(sequences, losses, train)

    def _entries_held(self) -> int:
        """Entries read from the caller's iterator but not yet trained on."""
        held = len(self._retry[1]) if self._retry is not None else 0
        if self._ahead is not None and self._ahead[1].exception() is None:
            held += len(self._ahead[1].result()[0])
        return held

    def _stage_payload(self, s: int, inputs, labels) -> dict:
        """What stage s is sent with a step: the inputs (first stage, or
        every stage under colocated vision) and the labels (last stage)."""
        colocated = self._plan.colocated_vision is not None
        return {
            "inputs": inputs if s == 0 or colocated else None,
            "labels": labels if s == len(self._plan.stages) - 1 else None,
        }

    def _drop_read_ahead(self) -> None:
        """Forget the step prefetch read ahead: after a load the caller
        resumes from its own position. Waits for the reader so it no longer
        touches the old iterator."""
        if self._ahead is not None:
            _, future = self._ahead
            self._ahead = None
            future.exception()  # wait; its entries and errors are dropped

    def _read_ahead(self, data_iter):
        """The next step's entries, and each stage's payload made ready."""
        entries = take_microbatch_entries(data_iter, self._plan.global_microbatches)
        inputs, labels = [e[0] for e in entries], [e[1] for e in entries]
        prepared = {}
        for s, worker in enumerate(self._workers):
            prepare = getattr(worker, "prepare", None)
            if prepare is not None:
                prepared[s] = prepare(**self._stage_payload(s, inputs, labels))
        return entries, prepared

    def _run_step(self, sequences, inputs, labels, train: bool, prepared=None, ahead=None) -> list:
        """Send each stage its whole op list as one command; stages exchange
        boundary tensors among themselves. Returns per-microbatch losses."""
        n_stages = len(self._plan.stages)
        terminal = n_stages - 1
        if self._p2p_dirty:
            self._reconnect_p2p(strict=True)
        kinds = ("forward", "backward", "eval", "ready")
        cmds, handles = {}, {}
        varying = _shapes_differ(inputs)
        started = time.perf_counter()
        # every payload before any dispatch: a stage left waiting on a
        # neighbour that was never sent its step hangs until the p2p timeout
        payloads = {}
        for s, worker in enumerate(self._workers):
            if prepared and s in prepared:
                payloads[s] = {"prepared": prepared[s]}
                continue
            payload = self._stage_payload(s, inputs, labels)
            prepare = getattr(worker, "prepare", None)
            try:
                payloads[s] = {"prepared": prepare(**payload)} if prepare else payload
            except Exception as e:
                raise StepFailed(
                    f"stage {s}: building its share of the step failed before any "
                    f"stage started: {type(e).__name__}: {e}"
                ) from e
        for s in range(n_stages):
            ops = [(c.kind, c.microbatch, c.command_id) for c in sequences[s] if c.kind in kinds]
            cmds[s] = Command(
                command_id=f"g{self._generation}.t{self._global_steps}.s{s}.step",
                generation=self._generation,
                global_step=self._global_steps,
                stage=s,
                kind="step" if train else "eval_step",
                microbatch=None,
                predecessors=(),
            )
            try:
                handles[s] = self._submit(
                    cmds[s], control={"ops": ops, "varying": varying}, **payloads[s]
                )
            except StepFailed:
                self._p2p_dirty = True
                raise
        dispatched = time.perf_counter()
        if ahead is not None:  # read and ship the next step while this one runs
            if self._reader is None:
                self._reader = ThreadPoolExecutor(max_workers=1)
            self._ahead = (ahead, self._reader.submit(self._read_ahead, ahead))
        results, errors = {}, []
        for s, handle in handles.items():  # resolve all, so no stage is left mid-step
            try:
                results[s] = self._resolve(handle, cmds[s])
            except StepFailed as e:
                errors.append(e)
        if errors:
            self._p2p_dirty = True
            # best effort now, to abort half-finished transfers instead of waiting
            # for a timeout; retried strictly before the next step
            self._reconnect_p2p(strict=False)
            raise errors[0]
        if train:
            for s, result in results.items():
                if not result["ready"]:
                    raise self._not_ready(s)
        if os.environ.get("RDSP_PROFILE") == "1":
            print(
                json.dumps(
                    {
                        "rdsp_profile": {"driver": "step"},
                        "dispatch": round((dispatched - started) * 1e3, 1),
                        "wait": round((time.perf_counter() - dispatched) * 1e3, 1),
                    }
                ),
                flush=True,
            )
        return results[terminal]["losses"]

    def _not_ready(self, stage: int) -> StepFailed:
        return StepFailed(
            f"stage {stage} not ready; generation {self._generation} "
            f"aborted before any optimizer update (safe to retry)"
        )

    def _reconnect_p2p(self, strict: bool) -> None:
        self._p2p_epoch += 1
        try:
            type(self._workers[0]).reconnect_p2p(self._workers, self._p2p_epoch)
            self._p2p_dirty = False
        except Exception as e:
            if strict:
                raise StepFailed(
                    f"could not rebuild stage-to-stage links after a failed step "
                    f"(a worker may be gone; load_checkpoint() rebuilds): {e}"
                ) from e

    def _finish(self, sequences, losses, train: bool) -> float:
        n_stages = len(self._plan.stages)
        mean_loss = sum(float(x) for x in losses) / len(losses)
        if not train:
            return mean_loss

        # once any apply is dispatched, stages may disagree on weights: poison on failure
        apply_cmds = {
            s: next(c for c in sequences[s] if c.kind == "apply") for s in range(n_stages)
        }
        started = time.perf_counter()
        try:
            applied = {s: self._submit(apply_cmds[s]) for s in range(n_stages)}
            for s, handle in applied.items():
                if self._resolve(handle, apply_cmds[s]) is False:
                    raise StepFailed(f"stage {s} failed its optimizer apply")
        except Exception as e:
            self._poisoned = True
            raise PipelinePoisoned(
                f"partial optimizer apply in generation {self._generation}: {e}; "
                f"restore from the last committed checkpoint"
            ) from e

        if os.environ.get("RDSP_PROFILE") == "1":
            print(
                json.dumps(
                    {
                        "rdsp_profile": {"driver": "apply"},
                        "apply": round((time.perf_counter() - started) * 1e3, 1),
                    }
                ),
                flush=True,
            )
        self._global_steps += 1
        return mean_loss

    def _submit(self, command: Command, **payload):
        """Submit to the command's stage; failures become StepFailed, except
        apply, whose failure the caller turns into PipelinePoisoned."""
        try:
            return self._workers[command.stage].submit(command, **payload)
        except Exception as e:
            if command.kind == "apply":
                raise
            raise StepFailed(f"stage {command.stage} failed on {command.command_id}: {e}") from e

    def _resolve(self, handle, command: Command | None):
        """Resolve a worker result; worker/transport failures become StepFailed."""
        try:
            return resolve(handle)
        except RdspError:
            raise
        except Exception as e:
            where = f"on {command.command_id}" if command else "resolving a result"
            raise StepFailed(f"worker failure {where}: {type(e).__name__}: {e}") from e
