# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""Checkpoint manifest and save/load protocol, on in-process fake stages.

Test names double as acceptance selectors; keep them stable."""

import json
import os

import pytest
import torch
from test_partition import ToyLM

import ray_deepspeed_pipeline as rdsp
from ray_deepspeed_pipeline import checkpoint as ckpt
from ray_deepspeed_pipeline.compiler import lower
from ray_deepspeed_pipeline.config import StageOverride
from ray_deepspeed_pipeline.coordinator import PipelineCoordinator
from ray_deepspeed_pipeline.errors import CheckpointError, PipelinePoisoned, StepFailed

DS = {"gradient_accumulation_steps": 2, "train_batch_size": 8}  # 4 rows per microbatch


class FakeStage:
    """One stage group: writes one shard per rank plus RNG, like a real
    stage (DeepSpeed files + rng/rank<r>.pt), and reports a stage record."""

    def __init__(self, stage, world_size, fail_save=False, drop_rank=None, is_last=False):
        self.stage, self.world_size, self.is_last = stage, world_size, is_last
        self.last_inputs = None
        self.fail_save, self.drop_rank = fail_save, drop_rank
        self.loaded = []
        self.is_alive = True
        self.fail_step = False

    def prepare(self, inputs=None, labels=None):
        return (inputs, labels)

    def submit(self, command, *, control=None, inputs=None, labels=None, prepared=None):
        if command.kind == "step":  # a training step, for prefetch tests
            if self.fail_step:
                raise RuntimeError("rank lost")
            self.last_inputs = prepared[0] if prepared is not None else inputs
            forwards = [mb for kind, mb, _ in control["ops"] if kind == "forward"]
            return {"losses": [0.5] * len(forwards) if self.is_last else None, "ready": True}
        if command.kind == "apply":
            return True
        if command.kind == "save":
            if self.fail_save:
                raise RuntimeError("disk full")
            root, tag = control["root"], control["tag"]
            sdir = ckpt.stage_dir(root, tag, self.stage)
            ranks = []
            for r in range(self.world_size):
                if r == self.drop_rank:
                    continue  # a rank that never reported
                os.makedirs(os.path.join(sdir, tag), exist_ok=True)
                with open(os.path.join(sdir, tag, f"shard_rank{r}.pt"), "wb") as f:
                    f.write(f"stage{self.stage}-rank{r}".encode())
                os.makedirs(os.path.join(sdir, "rng"), exist_ok=True)
                with open(ckpt.rng_path(root, tag, self.stage, r), "wb") as f:
                    f.write(b"rng")
                ranks.append({"rank": r})
            return {
                "index": self.stage,
                "world_size": self.world_size,
                "ranks": ranks,
                "files": ckpt.digest_tree(sdir),
            }
        if command.kind == "load" and "verify" in control:
            try:
                ckpt.verify_stage_files(
                    ckpt.stage_dir(control["root"], control["tag"], self.stage),
                    self.stage,
                    control["verify"][self.stage],
                )
            except CheckpointError as e:
                return str(e)
            return True
        if command.kind == "load":
            self.loaded.append(control["tag"])
            return True
        raise AssertionError(command.kind)

    def alive(self):
        return self.is_alive


def make_coordinator(world_sizes=(1, 2), stage_kwargs=None, prefetch=False):
    stage_kwargs = stage_kwargs or {}
    overrides = [StageOverride(stage=i, num_gpus=n) for i, n in enumerate(world_sizes)]
    cfg = rdsp.PipelineConfig(
        stages=len(world_sizes),
        partition=rdsp.UniformTransformerBlocks(),
        stage_overrides=overrides,
        prefetch=prefetch,
    )
    plan = lower(ToyLM(), cfg, DS)
    stages = [
        FakeStage(i, n, is_last=i == len(world_sizes) - 1, **stage_kwargs.get(i, {}))
        for i, n in enumerate(world_sizes)
    ]
    return plan, stages, PipelineCoordinator(plan, stages)


def test_all_stage_rank_save(tmp_path):
    plan, stages, coord = make_coordinator((1, 2))
    assert coord.save_checkpoint(
        str(tmp_path), "t1", client_state={"note": 1}, data_position={"owner": "caller"}
    )
    manifest = json.loads((tmp_path / "t1" / "manifest.json").read_text())
    assert (tmp_path / "latest").read_text() == "t1"
    assert manifest["plan_hash"] == plan.plan_hash()
    assert manifest["global_step"] == 0
    assert manifest["client_state"] == {"note": 1}
    assert [s["index"] for s in manifest["stages"]] == [0, 1]
    assert [[r["rank"] for r in s["ranks"]] for s in manifest["stages"]] == [[0], [0, 1]]
    # every shard and rng file of every rank is fingerprinted
    files = manifest["stages"][1]["files"]
    assert {"t1/shard_rank0.pt", "t1/shard_rank1.pt", "rng/rank0.pt", "rng/rank1.pt"} <= set(files)
    for rel, digest in files.items():
        assert ckpt.file_digest(str(tmp_path / "t1" / "stage1" / rel)) == digest


def test_default_tag_and_immutability(tmp_path):
    _, _, coord = make_coordinator()
    coord.save_checkpoint(str(tmp_path))
    assert (tmp_path / "latest").read_text() == "global_step0"
    with pytest.raises(CheckpointError, match="immutable"):
        coord.save_checkpoint(str(tmp_path))
    with pytest.raises(CheckpointError, match="plain directory name"):
        coord.save_checkpoint(str(tmp_path), "../escape")


def test_all_stage_rank_save_rejects_missing_rank(tmp_path):
    _, _, coord = make_coordinator((1, 2), {1: {"drop_rank": 1}})
    with pytest.raises(CheckpointError, match="expected ranks"):
        coord.save_checkpoint(str(tmp_path), "t1")
    assert not (tmp_path / "t1" / "manifest.json").exists()


def test_reject_partial_digest_plan_mismatch(tmp_path):
    plan, stages, coord = make_coordinator((1, 2))
    coord.save_checkpoint(str(tmp_path), "good")

    # 1. digest mismatch: a shard was modified after commit
    shard = tmp_path / "good" / "stage1" / "good" / "shard_rank1.pt"
    original = shard.read_bytes()
    shard.write_bytes(b"corrupted")
    with pytest.raises(CheckpointError, match="digest mismatch"):
        coord.load_checkpoint(str(tmp_path), "good")
    shard.write_bytes(original)

    # 2. partial: a rank's shard file is gone
    shard.unlink()
    with pytest.raises(CheckpointError, match="missing"):
        coord.load_checkpoint(str(tmp_path), "good")
    shard.write_bytes(original)

    # 3. plan mismatch: a pipeline with a different plan refuses it
    _, other_stages, other = make_coordinator((1, 1))
    with pytest.raises(CheckpointError, match="plan mismatch"):
        other.load_checkpoint(str(tmp_path), "good")

    # 4. an uncommitted tag (no manifest) is never loadable
    (tmp_path / "half" / "stage0").mkdir(parents=True)
    with pytest.raises(CheckpointError, match="never committed"):
        coord.load_checkpoint(str(tmp_path), "half")

    # rejected loads touched no worker
    assert all(not s.loaded for s in stages + other_stages)

    # and the intact checkpoint still loads
    assert coord.load_checkpoint(str(tmp_path))["tag"] == "good"
    assert all(s.loaded == ["good"] for s in stages)


def test_failure_keeps_manifest_unpublished(tmp_path):
    _, _, good = make_coordinator((1, 2))
    good.save_checkpoint(str(tmp_path), "committed")

    _, _, coord = make_coordinator((1, 2), {1: {"fail_save": True}})
    with pytest.raises(CheckpointError, match="no manifest was published"):
        coord.save_checkpoint(str(tmp_path), "broken")
    assert not (tmp_path / "broken" / "manifest.json").exists()
    assert (tmp_path / "latest").read_text() == "committed", "a failed save must not move `latest`"


def test_poisoned_pipeline_refuses_to_save(tmp_path):
    _, _, coord = make_coordinator()
    coord._poisoned = True
    with pytest.raises(PipelinePoisoned):
        coord.save_checkpoint(str(tmp_path), "t")
    assert not (tmp_path / "t").exists()


def test_load_restores_step_and_clears_poison(tmp_path):
    _, _, coord = make_coordinator()
    coord._global_steps = 7
    coord.save_checkpoint(str(tmp_path), "s7", client_state={"k": "v"})
    coord._global_steps = 9
    coord._poisoned = True
    result = coord.load_checkpoint(str(tmp_path))
    assert coord.global_steps == 7 and not coord._poisoned
    assert result["client_state"] == {"k": "v"}


def test_dead_worker_triggers_rebuild(tmp_path):
    plan, stages, _ = make_coordinator()
    fresh = [FakeStage(0, 1), FakeStage(1, 2)]
    coord = PipelineCoordinator(plan, stages, rebuild=lambda: fresh)
    coord.save_checkpoint(str(tmp_path), "t")
    stages[1].is_alive = False
    coord.load_checkpoint(str(tmp_path), "t")
    assert coord._workers is fresh
    assert all(s.loaded == ["t"] for s in fresh)


def test_reject_partial_digest_plan_mismatch_stage_side(tmp_path, monkeypatch):
    """Multi-node: the driver cannot see a stage's node-local shards, so the
    stage's own rank 0 must reject a corrupted file before anything loads,
    and a rejection leaves the pipeline's state exactly as it was."""
    plan, stages, coord = make_coordinator((1, 2))
    coord.save_checkpoint(str(tmp_path), "good")
    (tmp_path / "good" / "stage1" / "good" / "shard_rank0.pt").write_bytes(b"bad")
    monkeypatch.setattr(ckpt.os.path, "isdir", lambda p: False)  # driver blind
    with pytest.raises(CheckpointError, match="digest mismatch"):
        coord.load_checkpoint(str(tmp_path), "good")
    assert all(not s.loaded for s in stages) and not coord._poisoned


def test_dead_worker_without_rebuild_stays_poisoned(tmp_path):
    _, stages, coord = make_coordinator()
    coord.save_checkpoint(str(tmp_path), "t")
    stages[0].is_alive = False
    with pytest.raises(CheckpointError, match="rebuild"):
        coord.load_checkpoint(str(tmp_path), "t")
    with pytest.raises(PipelinePoisoned):
        coord.train_batch(iter([(torch.zeros(1), torch.zeros(1))] * 2))


def test_load_drops_the_step_prefetch_read_ahead(tmp_path):
    """After a load the caller hands in a new iterator; the step read ahead
    from the old one belongs to a run that no longer exists."""
    _, stages, coord = make_coordinator((1, 1), prefetch=True)
    coord.save_checkpoint(str(tmp_path), "t1")
    coord.train_batch(iter([(f"a{k}", f"y{k}") for k in range(4)]))  # reads a2, a3 ahead
    coord.load_checkpoint(str(tmp_path), "t1")
    coord.train_batch(iter([(f"b{k}", f"y{k}") for k in range(4)]))
    assert stages[0].last_inputs == ["b0", "b1"]


def test_a_save_after_a_failed_step_does_not_count_the_entries_kept_for_its_retry(tmp_path):
    _, stages, coord = make_coordinator((1, 1))
    stages[1].fail_step = True
    with pytest.raises(StepFailed):
        coord.train_batch(iter([(f"x{k}", f"y{k}") for k in range(2)]))
    coord.save_checkpoint(
        str(tmp_path), "t", data_position={"owner": "engine", "entries_consumed": 2}
    )
    manifest = json.loads((tmp_path / "t" / "manifest.json").read_text())
    assert manifest["data_position"] == {"owner": "engine", "entries_consumed": 0}


def test_a_save_under_prefetch_records_the_entries_read_ahead(tmp_path):
    _, _, coord = make_coordinator((1, 1), prefetch=True)
    coord.train_batch(iter([(f"x{k}", f"y{k}") for k in range(6)]))
    coord.save_checkpoint(str(tmp_path), "t", data_position={"owner": "caller"})
    manifest = json.loads((tmp_path / "t" / "manifest.json").read_text())
    assert manifest["data_position"] == {"owner": "caller", "read_ahead": 2}


def test_publish_creates_the_tag_directory(tmp_path):
    # node-local storage: no stage wrote under the driver's root
    manifest = ckpt.build_manifest(
        tag="t",
        global_step=0,
        plan_hash="h",
        plan_json="{}",
        stage_records=[],
        data_position=None,
        client_state={},
    )
    ckpt.publish(str(tmp_path / "root"), manifest)
    assert (tmp_path / "root" / "t" / "manifest.json").exists()
    blocker = tmp_path / "file"
    blocker.write_text("")
    with pytest.raises(CheckpointError, match="manifest"):
        ckpt.publish(str(blocker), manifest)
