# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""AutoTP end to end: HF checkpoint -> AutoTP + ZeRO training -> rollout at a new TP size.

1. Load a dense Hugging Face checkpoint into an AutoTP training engine (``--train_tp_size``,
   ZeRO ``--train_zero_stage``). Every weight is checked bit for bit against the safetensors files.
2. Train a few steps.
3. Move the trained weights into rollout engines with ``--rollout_tp_size``, three ways:

   - ``ucp``: ``save_checkpoint`` -> ``ds_to_universal`` -> ``load_checkpoint(load_universal)``.
   - ``gather``: what master offers on device: gather each whole parameter (``GatheredParameters``
     under ZeRO-3, then an all-gather of the TP shards) and slice the rollout shard out of it.
   - ``affine``: AutoTP storage maps of both engines -> ``plan_transfer`` -> ``execute_transfers``.
     The source is ZeRO's own storage (the ZeRO-3 partition, not a gathered parameter), and every
     element is sent once, from one holder to the ranks that need it.

   Rollout engines start from NaN weights, so any element a path fails to write shows up.
4. Generate greedily with ``HybridEngineRollout`` on the training engine and on every rollout engine.
5. ``--sync_steps`` times: train one more step and sync the ``gather`` and ``affine`` rollout engines
   again. The affine path reuses the plans of step 3, since a plan depends only on the two layouts.

Every step records its wall time and the device memory it added at its peak.

Oracles: the rollout weights equal the trained weights bit for bit, compared through whole
parameters rebuilt with plain collectives, independent of the affine code; and rollouts with the
same TP size differ in logits by no more than the forward's own run-to-run difference. Logits and
tokens against the training engine, whose TP size differs, are reported but not required to match.
"""

import argparse
import json
import math
import os
import platform
import shutil
import time
from types import SimpleNamespace

import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import deepspeed
import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.checkpoint.affine_transfer import PlanBudget, plan_transfer
from deepspeed.checkpoint.affine_transfer_executor import (StagingBudget, TransferEndpoints, TransferRequest,
                                                           execute_transfers)
from deepspeed.checkpoint.ds_to_universal import main as convert_to_universal
from deepspeed.checkpoint.zero_affine import autotp_storage_affine_maps, zero_param_storage_shard
from deepspeed.runtime.rollout import HybridEngineRollout, RolloutRequest, SamplingConfig
from deepspeed.utils import groups

DTYPE = torch.bfloat16
PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "Tensor parallelism splits each layer",
    "In 1969, the first humans",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_name", required=True, help="local HF checkpoint directory")
    parser.add_argument("--num_layers", type=int, default=0, help="truncate the model to this many layers (0: all)")
    parser.add_argument("--train_tp_size", type=int, default=2)
    parser.add_argument("--train_zero_stage", type=int, default=2, choices=(1, 2, 3))
    parser.add_argument("--rollout_tp_size", type=int, default=4)
    parser.add_argument("--paths", default="ucp,gather,affine")
    parser.add_argument("--train_steps", type=int, default=3)
    parser.add_argument("--sync_steps",
                        type=int,
                        default=3,
                        help="after the first sync, train one step and sync again this many times, "
                        "reusing the affine transfer plans")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--budget_mb", type=float, default=64.0, help="executor staging bytes per peer")
    parser.add_argument("--plan_budget_scale",
                        type=int,
                        default=1,
                        help="multiply every PlanBudget limit, for models whose maps outgrow the default")
    parser.add_argument("--replica_policy",
                        choices=["lowest", "local", "balanced"],
                        default="balanced",
                        help="which holder each target block reads from when ZeRO keeps several replicas")
    parser.add_argument("--work_dir", required=True, help="checkpoint directory for the UCP path")
    parser.add_argument("--output", default="autotp_rollout_e2e.json")
    parser.add_argument("--local_rank", type=int, default=-1)
    return parser.parse_args()


def device():
    return torch.device(get_accelerator().current_device_name())


def synchronize():
    get_accelerator().synchronize()


def max_over_ranks(value):
    tensor = torch.tensor([float(value)], dtype=torch.float64, device=device())
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return tensor.item()


def reduce_over_ranks(value, op):
    tensor = torch.tensor([float(value)], dtype=torch.float64, device=device())
    dist.all_reduce(tensor, op=op)
    return int(tensor.item())


def sum_over_ranks(value):
    return reduce_over_ranks(value, dist.ReduceOp.SUM)


class Timer:
    """Wall time of a step, as the slowest rank saw it, and the device memory the step added at its peak."""

    def __init__(self):
        self.seconds = {}
        self.memory_gb = {}

    def __call__(self, name):
        timer = self

        class Scope:

            def __enter__(self):
                dist.barrier()
                synchronize()
                accelerator = get_accelerator()
                accelerator.reset_peak_memory_stats()
                self.allocated = accelerator.memory_allocated()
                self.started = time.perf_counter()

            def __exit__(self, *exc):
                synchronize()
                timer.seconds[name] = max_over_ranks(time.perf_counter() - self.started)
                peak = get_accelerator().max_memory_allocated()
                timer.memory_gb[name] = {
                    # Memory held before the step plus whatever the step needed on top of it.
                    "peak": max_over_ranks(peak) / 2**30,
                    "added_at_peak": max_over_ranks(peak - self.allocated) / 2**30,
                }

        return Scope()


# ---------------------------------------------------------------------------
# Engines
# ---------------------------------------------------------------------------


def truncated_config(args):
    config = AutoConfig.from_pretrained(args.model_name)
    if args.num_layers:
        config.num_hidden_layers = args.num_layers
    return config


def ds_config(zero_stage, tp_size, lr, load_universal=False, optimizer=True):
    config = {
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 1,
        "bf16": {
            "enabled": True
        },
        "zero_optimization": {
            "stage": zero_stage
        },
        # The model's own tp_plan decides how each linear is split.
        "tensor_parallel": {
            "autotp_size": tp_size
        },
        "zero_allow_untested_optimizer": True,
    }
    if optimizer:
        config["optimizer"] = {"type": "AdamW", "params": {"lr": lr, "torch_adam": True}}
    if load_universal:
        config["checkpoint"] = {"load_universal": True}
    return config


# The process-global parallel state DeepSpeed fills in at ``initialize`` and reads back later.
GROUP_STATE = ("_TENSOR_MODEL_PARALLEL_GROUP", "_MODEL_PARALLEL_GROUP", "_DATA_PARALLEL_GROUP",
               "_MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE", "_MPU_TENSOR_MODEL_PARALLEL_RANK", "mesh_device", "mpu")


class Engine:
    """A DeepSpeed engine with the tensor-parallel group its AutoTP layers were built with."""

    def __init__(self, model, config):
        # DeepSpeed keeps one TP mesh per process and reuses it for every later engine. The training
        # and rollout engines differ in TP size, so drop the previous engine's references first; its
        # layers keep their own groups, and this script passes each engine's group explicitly.
        for group in (groups._TENSOR_MODEL_PARALLEL_GROUP, groups._DATA_PARALLEL_GROUP):
            if group is not None:
                groups._release_process_group_references(group)
        self.engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config)
        self.tp_group = groups.get_tensor_model_parallel_group()
        self.group_state = {name: getattr(groups, name) for name in GROUP_STATE}

    def activate(self):
        """Point the process-global parallel state back at this engine's groups before training it again."""
        for name, value in self.group_state.items():
            setattr(groups, name, value)

    def params(self):
        return dict(self.engine.module.named_parameters())


def build_rollout_engine(args, load_universal):
    """A rollout engine with every weight set to NaN, so that unwritten elements cannot pass."""
    model = AutoModelForCausalLM.from_config(truncated_config(args), dtype=DTYPE)
    with torch.no_grad():
        for param in model.parameters():
            param.fill_(float("nan"))
    # Universal checkpoints load through the ZeRO optimizer, so the rollout engine keeps one when the
    # UCP path runs; otherwise it has none, as a rollout engine would. Every path uses the same engine
    # configuration so that their logits are comparable bit for bit.
    with_optimizer = "ucp" in args.paths.split(",")
    config = ds_config(1 if with_optimizer else 0, args.rollout_tp_size, args.lr, load_universal, with_optimizer)
    return Engine(model, config)


# ---------------------------------------------------------------------------
# Whole parameters, rebuilt with plain collectives
# ---------------------------------------------------------------------------


def local_parameter(param):
    """This rank's tensor-parallel shard, gathered with ZeRO's own collective when it is partitioned."""
    if hasattr(param, "ds_id"):
        with deepspeed.zero.GatheredParameters([param]):
            return param.data.detach().clone()
    return param.data.detach().clone()


def all_ranks(value, group=None):
    gathered = [None] * dist.get_world_size(group=group)
    dist.all_gather_object(gathered, value, group=group)
    return gathered


def split_dim(local_shape, full_shape):
    """The axis AutoTP split this parameter along, or None when every TP rank holds all of it."""
    differing = [axis for axis, (local, full) in enumerate(zip(local_shape, full_shape)) if local != full]
    assert len(differing) <= 1, (local_shape, full_shape)
    return differing[0] if differing else None


def tp_slice(param, tp_group, full_shape):
    """(axis, start, length) of this rank's TP shard, assuming TP ranks hold consecutive slices."""
    local_shape = tuple(param.ds_shape if hasattr(param, "ds_id") else param.shape)
    axis = split_dim(local_shape, full_shape)
    if axis is None:
        return None
    lengths = [shape[axis] for shape in all_ranks(local_shape, tp_group)]
    tp_rank = dist.get_rank(group=tp_group)
    assert sum(lengths) == full_shape[axis], (lengths, full_shape)
    return axis, sum(lengths[:tp_rank]), lengths[tp_rank]


def full_tensor(param, tp_group, full_shape):
    """The whole logical parameter: ZeRO gather, then the TP shards concatenated in TP rank order."""
    local = local_parameter(param)
    axis = split_dim(tuple(local.shape), full_shape)
    if axis is None:
        return local
    shapes = all_ranks(tuple(local.shape), tp_group)
    padded = torch.zeros(max(math.prod(shape) for shape in shapes), dtype=local.dtype, device=local.device)
    padded[:local.numel()] = local.reshape(-1)
    parts = [torch.empty_like(padded) for _ in shapes]
    dist.all_gather(parts, padded, group=tp_group)
    return torch.cat([part[:math.prod(shape)].view(shape) for part, shape in zip(parts, shapes)], dim=axis)


def expected_shard(full, param, tp_group, full_shape):
    where = tp_slice(param, tp_group, full_shape)
    if where is None:
        return full
    axis, start, length = where
    return full.narrow(axis, start, length)


# ---------------------------------------------------------------------------
# Comparisons
# ---------------------------------------------------------------------------


def bit_mismatches(got, expected):
    got = got.reshape(-1)
    expected = expected.reshape(-1).to(got.device)
    assert got.numel() == expected.numel()
    return int((got.view(torch.int16) != expected.view(torch.int16)).sum().item())


class WeightCheck:
    """Counts of elements that differ bit for bit, summed over ranks."""

    def __init__(self):
        self.mismatched = 0
        self.checked = 0
        self.unmatched_names = []

    def add(self, got, expected):
        self.mismatched += bit_mismatches(got, expected)
        self.checked += got.numel()

    def result(self):
        return {
            "mismatched_elements": sum_over_ranks(self.mismatched),
            "checked_elements": sum_over_ranks(self.checked),
            "unmatched_names": self.unmatched_names,
        }


class HFCheckpoint:

    def __init__(self, path):
        self.path = path
        self.files = {}
        index = os.path.join(path, "model.safetensors.index.json")
        if os.path.exists(index):
            with open(index) as stream:
                self.weight_map = json.load(stream)["weight_map"]
            return
        # A checkpoint small enough to be saved unsharded has one file and no index.
        single = safe_open(os.path.join(path, "model.safetensors"), framework="pt", device="cpu")
        self.files["model.safetensors"] = single
        self.weight_map = {key: "model.safetensors" for key in single.keys()}

    def __contains__(self, key):
        return key in self.weight_map

    def get(self, key):
        file_name = self.weight_map[key]
        if file_name not in self.files:
            self.files[file_name] = safe_open(os.path.join(self.path, file_name), framework="pt", device="cpu")
        return self.files[file_name].get_tensor(key)


def check_against_hf(trained, checkpoint, full_shapes):
    """Every training shard equals its slice of the HF safetensors tensor of the same name."""
    check = WeightCheck()
    for name, param in trained.params().items():
        if name not in checkpoint:
            check.unmatched_names.append(name)
            continue
        full = checkpoint.get(name).to(device())
        check.add(local_parameter(param), expected_shard(full, param, trained.tp_group, full_shapes[name]))
    return check.result()


def changed_fraction(trained, checkpoint, full_shapes):
    """Fraction of trained weights that differ from the HF checkpoint, to show training moved them."""
    changed, total = 0, 0
    for name, param in trained.params().items():
        full = checkpoint.get(name).to(device())
        local = local_parameter(param)
        changed += bit_mismatches(local, expected_shard(full, param, trained.tp_group, full_shapes[name]))
        total += local.numel()
    return sum_over_ranks(changed) / max(sum_over_ranks(total), 1)


def check_rollout_weights(rollout, trained, full_shapes):
    """Each rollout shard equals its slice of the whole trained parameter."""
    check = WeightCheck()
    trained_params = trained.params()
    for name, param in rollout.params().items():
        full = full_tensor(trained_params[name], trained.tp_group, full_shapes[name])
        check.add(local_parameter(param), expected_shard(full, param, rollout.tp_group, full_shapes[name]))
    return check.result()


# ---------------------------------------------------------------------------
# Weight movement
# ---------------------------------------------------------------------------


def sync_ucp(trained, args, timer):
    tag = "trained"
    save_dir = os.path.join(args.work_dir, "checkpoint")
    with timer("ucp_save"):
        trained.engine.save_checkpoint(save_dir, tag=tag)
    with timer("ucp_convert"):
        if dist.get_rank() == 0:
            convert_to_universal(
                SimpleNamespace(input_folder=os.path.join(save_dir, tag),
                                output_folder=os.path.join(save_dir, f"{tag}_universal"),
                                num_extract_workers=4,
                                num_merge_workers=4,
                                keep_temp_folder=False,
                                strict=True,
                                inject_missing_state=False))
    rollout = build_rollout_engine(args, load_universal=True)
    with timer("ucp_load"):
        rollout.engine.load_checkpoint(save_dir,
                                       tag=f"{tag}_universal",
                                       load_optimizer_states=False,
                                       load_lr_scheduler_states=False)
    return rollout, {}, None


class GatherSync:
    """Gather each whole trained parameter and copy the rollout shard out of it."""

    def __init__(self, trained, rollout, full_shapes):
        self.trained = trained
        self.rollout = rollout
        self.full_shapes = full_shapes

    def __call__(self):
        trained_params = self.trained.params()
        for name, param in self.rollout.params().items():
            full = full_tensor(trained_params[name], self.trained.tp_group, self.full_shapes[name])
            param.data.copy_(expected_shard(full, param, self.rollout.tp_group, self.full_shapes[name]))
        return {}


class ReplicaPolicy:
    """Choose the source replica of each target block for ``plan_transfer``.

    ``lowest`` is the planner's own default. ``local`` reads from the target's own process when it
    holds a replica, which turns a send into a local copy. ``balanced`` does that too and otherwise
    picks the holder that has been asked to send the fewest elements so far, so ZeRO-0/1/2 replicas
    share the load instead of the lowest rank of every group sending everything. Every rank plans
    the same parameters in the same order, so every rank makes the same choices.
    """

    def __init__(self, name):
        self.name = name
        self.sent = {}

    def __call__(self, request):
        ranks = [candidate.source_rank for candidate in request.candidates]
        if self.name != "lowest" and request.target_rank in ranks:
            return request.target_rank
        if self.name != "balanced":
            return min(ranks)
        rank = min(ranks, key=lambda r: (self.sent.get(r, 0), r))
        self.sent[rank] = self.sent.get(rank, 0) + math.prod(request.shape)
        return rank


class AffineSync:
    """Transfer plans built once from both engines' storage maps, executed on every sync.

    A plan depends only on the two layouts, not on the values, so it stays valid as long as neither
    engine is rebuilt or repartitioned. Each sync only looks up the current storage views and runs
    the executor.
    """

    def __init__(self, trained, rollout, args, timer):
        self.trained = trained
        self.rollout = rollout
        world = dist.get_world_size()
        self.endpoints = TransferEndpoints(source={r: r for r in range(world)}, target={r: r for r in range(world)})
        self.budget = StagingBudget(bytes_per_peer=int(args.budget_mb * (1 << 20)))
        # The default budget suits block-shaped maps. A flattened row-parallel weight is one piece per
        # row, and the planner checks pieces pairwise, so its count limits are raised together.
        default = PlanBudget()
        scale = args.plan_budget_scale
        plan_budget = PlanBudget(max_pieces_per_map=default.max_pieces_per_map * scale,
                                 max_pair_checks=default.max_pair_checks * scale * scale,
                                 max_segments=default.max_segments * scale,
                                 max_plan_bytes=default.max_plan_bytes * scale,
                                 max_merge_checks=default.max_merge_checks * scale * scale)
        with timer("affine_maps"):
            self.source_maps = autotp_storage_affine_maps(trained.engine.module, tp_group=trained.tp_group)
            self.target_maps = autotp_storage_affine_maps(rollout.engine.module, tp_group=rollout.tp_group)
        selector = None if args.replica_policy == "lowest" else ReplicaPolicy(args.replica_policy)
        with timer("affine_plan"):
            self.plans = {
                name: plan_transfer(self.target_maps[name],
                                    self.source_maps[name],
                                    plan_budget,
                                    replica_selector=selector)
                for name in rollout.params()
            }

    def pieces(self, maps):
        return sum(len(pieces) for affine_map in maps.values() for pieces in affine_map.pieces_by_rank.values())

    def __call__(self):
        rank = dist.get_rank()
        # ZeRO-3 keeps persistent parameters gathered between steps; the shards must be their only copies.
        self.trained.engine.empty_partition_cache()
        self.rollout.engine.empty_partition_cache()
        trained_params = self.trained.params()
        transfers = []
        for name, param in self.rollout.params().items():
            # Views into the engines' live storage, looked up again on every sync; cheap, no planning.
            sources = {}
            if rank in self.source_maps[name].shard_shapes:
                sources[rank] = zero_param_storage_shard(trained_params[name])
            targets = {}
            if rank in self.target_maps[name].shard_shapes:
                targets[rank] = zero_param_storage_shard(param)
            transfers.append(TransferRequest(name, self.plans[name], sources, targets))
        stats = execute_transfers(transfers, self.endpoints, DTYPE, self.budget)
        return {
            "sent_bytes": sum_over_ranks(stats.sent_bytes),
            "local_copy_bytes": sum_over_ranks(stats.local_copy_bytes),
            "rounds": reduce_over_ranks(stats.rounds, dist.ReduceOp.MAX),
            "source_pieces": self.pieces(self.source_maps),
            "target_pieces": self.pieces(self.target_maps),
        }


def sync_gather(trained, args, timer, full_shapes):
    rollout = build_rollout_engine(args, load_universal=False)
    sync = GatherSync(trained, rollout, full_shapes)
    with timer("gather_copy"):
        stats = sync()
    return rollout, stats, sync


def sync_affine(trained, args, timer):
    rollout = build_rollout_engine(args, load_universal=False)
    sync = AffineSync(trained, rollout, args, timer)
    with timer("affine_execute"):
        stats = sync()
    return rollout, stats, sync


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------


def prompts(tokenizer):
    tokenizer.padding_side = "left"
    batch = tokenizer(PROMPTS, return_tensors="pt", padding=True)
    return batch["input_ids"].to(device()), batch["attention_mask"].to(device())


def generate(engine, tokenizer, prompt_ids, prompt_mask, args, timer, name):
    engine.eval()
    rollout = HybridEngineRollout(engine, tokenizer)
    with timer(f"{name}_generate"):
        batch = rollout.generate(RolloutRequest(prompt_ids, prompt_mask),
                                 SamplingConfig(max_new_tokens=args.max_new_tokens, temperature=0.0))
    # ZeRO-3 may leave prefetched parameters in flight after a forward; a later
    # GatheredParameters on them would fail.
    engine.empty_partition_cache()
    return batch.input_ids


@torch.no_grad()
def logits_of(wrapped, input_ids, attention_mask, vocab_size):
    engine = wrapped.engine
    engine.eval()
    logits = engine.module(input_ids=input_ids, attention_mask=attention_mask).logits.float()
    engine.empty_partition_cache()
    if logits.shape[-1] != vocab_size:
        # A vocabulary-parallel head leaves each TP rank its own slice of the vocabulary.
        shapes = all_ranks(logits.shape[-1], wrapped.tp_group)
        width = max(shapes)
        padded = torch.zeros(*logits.shape[:-1], width, dtype=logits.dtype, device=logits.device)
        padded[..., :logits.shape[-1]] = logits
        parts = [torch.empty_like(padded) for _ in shapes]
        dist.all_gather(parts, padded.contiguous(), group=wrapped.tp_group)
        logits = torch.cat([part[..., :size] for part, size in zip(parts, shapes)], dim=-1)
        assert logits.shape[-1] == vocab_size, (shapes, vocab_size)
    return logits


def compare_logits(got, expected, mask):
    """Every rank runs the same prompts; each compares its own outputs, and results are combined."""
    valid = mask.bool()
    difference = (got - expected).abs()[valid]
    argmax_differs = (got.argmax(-1) != expected.argmax(-1))[valid]
    by_rank = all_ranks({
        "bitwise_equal": bool(torch.equal(got, expected)),
        "max_abs_diff": difference.max().item(),
        "argmax_mismatched_positions": int(argmax_differs.sum().item()),
    })
    return {
        "bitwise_equal": all(entry["bitwise_equal"] for entry in by_rank),
        "max_abs_diff": max(entry["max_abs_diff"] for entry in by_rank),
        "reference_max_abs": expected[valid].abs().max().item(),
        "argmax_mismatched_positions": sum(entry["argmax_mismatched_positions"] for entry in by_rank),
        "positions": int(valid.sum().item()) * dist.get_world_size(),
    }


def compare_tokens(got, expected, prompt_len):
    same = (got[:, prompt_len:] == expected[:, prompt_len:])
    first = [int((~row).nonzero()[0]) if not row.all() else None for row in same]
    by_rank = all_ranks(first)
    return {
        "identical": all(divergence is None for rank_first in by_rank for divergence in rank_first),
        "first_divergence_by_rank": by_rank,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def batch_generator(args):
    # Every rank of a TP group must see the same batch; data-parallel replicas see different ones.
    dp_rank = dist.get_rank() // args.train_tp_size
    return torch.Generator().manual_seed(4321 + dp_rank)


def train(trained, config, args, steps, generator):
    trained.activate()
    engine = trained.engine
    engine.train()
    losses = []
    for _ in range(steps):
        tokens = torch.randint(0, config.vocab_size, (1, args.seq_len), generator=generator).to(device())
        loss = engine(input_ids=tokens, labels=tokens).loss
        engine.backward(loss)
        engine.step()
        losses.append(loss.item())
    return losses


def manifest(args):
    accelerator = get_accelerator()
    return {
        "args": vars(args),
        "world_size": dist.get_world_size(),
        "device_name": accelerator.get_device_name(0) if hasattr(accelerator, "get_device_name") else None,
        "torch": torch.__version__,
        "deepspeed": deepspeed.__version__,
        "host": platform.node(),
    }


def main():
    args = parse_args()
    get_accelerator().set_device(int(os.environ.get("LOCAL_RANK", 0)))
    deepspeed.init_distributed()
    rank = dist.get_rank()
    paths = args.paths.split(",")
    if rank == 0:
        shutil.rmtree(args.work_dir, ignore_errors=True)
        os.makedirs(args.work_dir)
    timer = Timer()
    report = {"manifest": manifest(args), "oracles": {}}
    oracles = report["oracles"]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    prompt_ids, prompt_mask = prompts(tokenizer)
    prompt_len = prompt_ids.shape[1]
    checkpoint = HFCheckpoint(args.model_name)
    config = truncated_config(args)
    with torch.device("meta"):
        full_shapes = {
            name: tuple(param.shape)
            for name, param in AutoModelForCausalLM.from_config(config).named_parameters()
        }

    # 1. HF checkpoint -> training engine.
    with timer("hf_load"):
        model = AutoModelForCausalLM.from_pretrained(args.model_name, config=config, dtype=DTYPE)
    trained = Engine(model, ds_config(args.train_zero_stage, args.train_tp_size, args.lr))
    oracles["hf_weights"] = check_against_hf(trained, checkpoint, full_shapes)

    # 2. Train.
    generator = batch_generator(args)
    with timer("train"):
        report["losses"] = train(trained, config, args, args.train_steps, generator)
    report["trained_changed_fraction"] = changed_fraction(trained, checkpoint, full_shapes)

    # Shared-weight rollout on the training engine itself: the reference sequences. The UCP path
    # saves through the training engine, so it runs before any rollout engine replaces the TP mesh.
    reference_ids = generate(trained.engine, tokenizer, prompt_ids, prompt_mask, args, timer, "trained")
    reference_mask = (reference_ids != tokenizer.pad_token_id).long()
    reference_mask[:, :prompt_len] = prompt_mask
    trained_logits = logits_of(trained, reference_ids, reference_mask, config.vocab_size)
    oracles["trained_logits_repeat"] = compare_logits(
        logits_of(trained, reference_ids, reference_mask, config.vocab_size), trained_logits, reference_mask)

    # 3. Move the weights, 4. roll out.
    rollouts = {}
    syncs = {}
    for path in paths:
        if path == "ucp":
            rollout, stats, sync = sync_ucp(trained, args, timer)
        elif path == "gather":
            rollout, stats, sync = sync_gather(trained, args, timer, full_shapes)
        elif path == "affine":
            rollout, stats, sync = sync_affine(trained, args, timer)
        else:
            raise ValueError(path)
        result = {"weights_vs_trained": check_rollout_weights(rollout, trained, full_shapes), "transfer": stats}
        ids = generate(rollout.engine, tokenizer, prompt_ids, prompt_mask, args, timer, path)
        logits = logits_of(rollout, reference_ids, reference_mask, config.vocab_size)
        # The same engine and inputs a second time: shows whether the forward itself is deterministic.
        result["logits_repeat"] = compare_logits(logits_of(rollout, reference_ids, reference_mask, config.vocab_size),
                                                 logits, reference_mask)
        result["tokens_vs_trained"] = compare_tokens(ids, reference_ids, prompt_len)
        result["logits_vs_trained"] = compare_logits(logits, trained_logits, reference_mask)
        rollouts[path] = (rollout, ids, logits)
        if sync is not None:
            syncs[path] = sync
        oracles[path] = result

    # 5. Keep training and sync again, as an RL loop would; the affine path reuses its plans. Before
    # each sync the rollout weights must be stale, after it they must equal the trained ones again.
    report["repeated_syncs"] = []
    for step in range(args.sync_steps):
        entry = {"loss": train(trained, config, args, 1, generator)[0]}
        for path, sync in syncs.items():
            rollout = rollouts[path][0]
            stale = check_rollout_weights(rollout, trained, full_shapes)
            name = f"repeat{step}_{path}"
            with timer(name):
                stats = sync()
            entry[path] = {
                "stale_mismatched_elements": stale["mismatched_elements"],
                "weights_vs_trained": check_rollout_weights(rollout, trained, full_shapes),
                "seconds": timer.seconds[name],
                "memory_gb": timer.memory_gb[name],
                "transfer": stats,
            }
        report["repeated_syncs"].append(entry)
    names = list(rollouts)
    for index, first in enumerate(names):
        for second in names[index + 1:]:
            oracles[f"{first}_vs_{second}"] = {
                "tokens": compare_tokens(rollouts[first][1], rollouts[second][1], prompt_len),
                "logits": compare_logits(rollouts[first][2], rollouts[second][2], reference_mask),
            }

    report["seconds"] = timer.seconds
    report["memory_gb"] = timer.memory_gb
    report["generations"] = {"trained": tokenizer.batch_decode(reference_ids[:, prompt_len:])}
    for path, (_, ids, _) in rollouts.items():
        report["generations"][path] = tokenizer.batch_decode(ids[:, prompt_len:])

    failures = []
    if oracles["hf_weights"]["mismatched_elements"] or oracles["hf_weights"]["unmatched_names"]:
        failures.append("hf_weights")
    if report["trained_changed_fraction"] == 0:
        failures.append("training did not change any weight")
    for path in rollouts:
        if oracles[path]["weights_vs_trained"]["mismatched_elements"]:
            failures.append(f"{path} weights")
    # Same engine configuration and bitwise equal weights, so two rollouts may differ only by the
    # forward's own run-to-run noise: zero where the forward is deterministic.
    repeats = [oracles["trained_logits_repeat"]] + [oracles[path]["logits_repeat"] for path in rollouts]
    noise = max(repeat["max_abs_diff"] for repeat in repeats)
    for index, first in enumerate(names):
        for second in names[index + 1:]:
            if oracles[f"{first}_vs_{second}"]["logits"]["max_abs_diff"] > noise:
                failures.append(f"{first}_vs_{second} logits beyond run-to-run noise")
    for step, entry in enumerate(report["repeated_syncs"]):
        for path in syncs:
            if entry[path]["stale_mismatched_elements"] == 0:
                failures.append(f"repeat{step} {path}: training left the rollout weights unchanged")
            if entry[path]["weights_vs_trained"]["mismatched_elements"]:
                failures.append(f"repeat{step} {path} weights")
    report["failures"] = failures

    if rank == 0:
        print(json.dumps({"oracles": oracles, "seconds": timer.seconds, "failures": failures}, indent=2))
        with open(args.output, "w") as stream:
            json.dump(report, stream, indent=2, default=str)
        print(f"wrote {args.output}")
    for rollout, _, _ in rollouts.values():
        rollout.engine.destroy()
    trained.engine.destroy()
    if failures:
        raise SystemExit(f"failed: {failures}")


if __name__ == "__main__":
    main()
