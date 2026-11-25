# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""Global checkpoint manifest: layout, digests, publish, validate.

A global checkpoint is a directory per tag:

    <save_dir>/<tag>/stage<s>/...        each stage's own DeepSpeed checkpoint
                                         (written by DeepSpeed, never parsed here)
    <save_dir>/<tag>/stage<s>/rng/rank<r>.pt
    <save_dir>/<tag>/manifest.json       published LAST, only when every rank of
                                         every stage finished
    <save_dir>/latest                    tag of the last committed manifest

The manifest is the commit record: a tag without manifest.json is an
unfinished save and is never loaded. Committed tags are never overwritten.

Shards may live on node-local disks: each stage runs on one node, whose rank 0
digests the stage's files at save and re-verifies them at load. Recovering from
a lost node therefore needs a shared checkpoint directory.
"""

import hashlib
import json
import os
import time

from ray_deepspeed_pipeline.errors import CheckpointError

MANIFEST_FORMAT = "rdsp-manifest/1"
MANIFEST_NAME = "manifest.json"
LATEST_NAME = "latest"


def stage_dir(root: str, tag: str, stage: int) -> str:
    return os.path.join(root, tag, f"stage{stage}")


def rng_path(root: str, tag: str, stage: int, rank: int) -> str:
    return os.path.join(stage_dir(root, tag, stage), "rng", f"rank{rank}.pt")


def file_digest(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest_tree(directory: str) -> dict[str, str]:
    """sha256 of every file under directory, keyed by path relative to it."""
    out = {}
    for base, _, files in os.walk(directory):
        for name in files:
            path = os.path.join(base, name)
            out[os.path.relpath(path, directory)] = file_digest(path)
    return dict(sorted(out.items()))


def validate_tag(tag: str) -> str:
    """Return tag if it is a single path component; else raise CheckpointError."""
    if (
        not isinstance(tag, str)
        or not tag
        or tag in (".", "..")
        or os.sep in tag
        or (os.altsep and os.altsep in tag)
    ):
        raise CheckpointError(f"checkpoint tag must be a plain directory name, got {tag!r}")
    return tag


def build_manifest(
    *,
    tag: str,
    global_step: int,
    plan_hash: str,
    plan_json: str,
    stage_records: list[dict],
    data_position,
    client_state,
) -> dict:
    """Build the commit record. Raises CheckpointError if client_state is not JSON.

    stage_records: one per stage, {index, world_size, ranks, files: {relpath: sha256}}.
    """
    try:
        json.dumps(client_state)
    except TypeError as e:
        raise CheckpointError(f"client_state must be JSON-serializable: {e}") from e
    return {
        "format": MANIFEST_FORMAT,
        "tag": tag,
        "global_step": int(global_step),
        "plan_hash": plan_hash,
        "plan": json.loads(plan_json),
        "stages": sorted(stage_records, key=lambda r: r["index"]),
        "data_position": data_position,
        "client_state": client_state,
        "created_unix": time.time(),
    }


def check_complete(manifest: dict, plan_stages: list[int]) -> None:
    """Raise CheckpointError unless every planned stage and all its ranks finished."""
    by_index = {r["index"]: r for r in manifest["stages"]}
    for expected in plan_stages:
        if expected not in by_index:
            raise CheckpointError(f"stage {expected} missing from checkpoint")
    for record in manifest["stages"]:
        finished = sorted(r["rank"] for r in record["ranks"])
        if finished != list(range(record["world_size"])):
            raise CheckpointError(
                f"stage {record['index']}: expected ranks "
                f"{list(range(record['world_size']))}, finished {finished}"
            )


def publish(root: str, manifest: dict, *, update_latest: bool = True) -> str:
    """Atomically write the manifest, then advance `latest`.

    Call only after every shard is written: the manifest's presence is what marks
    the checkpoint complete. Raises CheckpointError if the tag is already committed
    or the manifest cannot be written.
    """
    tag = manifest["tag"]
    final = os.path.join(root, tag, MANIFEST_NAME)
    if os.path.exists(final):
        raise CheckpointError(f"checkpoint {tag!r} is already committed; tags are immutable")
    tmp = final + ".tmp"
    try:
        # on node-local storage no stage may have written under the driver's root
        os.makedirs(os.path.dirname(final), exist_ok=True)
        with open(tmp, "w") as f:
            json.dump(manifest, f, indent=1, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, final)
        if not update_latest:
            return final
        latest_tmp = os.path.join(root, LATEST_NAME + ".tmp")
        with open(latest_tmp, "w") as f:
            f.write(tag)
        os.replace(latest_tmp, os.path.join(root, LATEST_NAME))
    except OSError as e:
        raise CheckpointError(f"could not write the manifest of {tag!r} under {root!r}: {e}") from e
    return final


def resolve_tag(root: str, tag: str | None) -> str:
    if tag is not None:
        return validate_tag(tag)
    latest = os.path.join(root, LATEST_NAME)
    if not os.path.exists(latest):
        raise CheckpointError(f"no committed checkpoint under {root!r} (no 'latest' file)")
    with open(latest) as f:
        return validate_tag(f.read().strip())


def read_and_verify(root: str, tag: str | None, *, plan_hash: str, plan_stages: list[int]) -> dict:
    """Load a committed manifest, raising CheckpointError if it is incomplete,
    for another plan, or its driver-visible files are missing or changed."""
    tag = resolve_tag(root, tag)
    path = os.path.join(root, tag, MANIFEST_NAME)
    if not os.path.exists(path):
        raise CheckpointError(
            f"checkpoint {tag!r} has no manifest: it was never committed "
            f"(an interrupted or failed save)"
        )
    with open(path) as f:
        manifest = json.load(f)
    if manifest.get("format") != MANIFEST_FORMAT:
        raise CheckpointError(f"unknown manifest format {manifest.get('format')!r}")
    if manifest["plan_hash"] != plan_hash:
        raise CheckpointError(
            f"plan mismatch: checkpoint {tag!r} was written by plan "
            f"{manifest['plan_hash'][:12]}, this pipeline runs "
            f"{plan_hash[:12]}; stage-local shards are not convertible across "
            f"plans"
        )
    check_complete(manifest, plan_stages)
    for record in manifest["stages"]:
        directory = stage_dir(root, tag, record["index"])
        # early check when the driver can see the shards; stages always re-verify
        if os.path.isdir(directory):
            verify_stage_files(directory, record["index"], record["files"])
    return manifest


def verify_stage_files(sdir: str, stage: int, files: dict[str, str]) -> None:
    """Raise CheckpointError if any recorded shard under sdir is missing or changed."""
    for rel, digest in files.items():
        path = os.path.join(sdir, rel)
        if not os.path.exists(path):
            raise CheckpointError(f"stage {stage}: shard file {rel!r} is missing")
        if file_digest(path) != digest:
            raise CheckpointError(f"stage {stage}: digest mismatch for {rel!r}")
