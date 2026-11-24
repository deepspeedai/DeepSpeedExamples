# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""AutoEP end to end: HF checkpoint -> ZeRO training -> rollout, through UCP or AffineIR.

1. Load a Hugging Face MoE checkpoint, truncated to ``--num_layers``, into an AutoEP training
   engine. Every weight is checked bit for bit against the safetensors files, and the logits
   against plain transformers.
2. Train a few steps.
3. Move the trained weights into rollout engines with a different expert layout, two ways:

   - ``ucp``: ``save_checkpoint`` -> ``ds_to_universal`` -> ``load_checkpoint(load_universal)``.
   - ``affine``: AutoEP storage maps -> ``plan_transfer`` -> ``execute_transfers`` for the experts,
     on device and without files. Dense weights are copied on device (gathered under ZeRO-3).

   Rollout engines start from NaN weights, so any element a path fails to write shows up.
4. Generate greedily with ``HybridEngineRollout`` on the training engine and on every rollout engine.

Oracles: the rollout weights equal the trained weights bit for bit (experts are matched by
expert id through plain collectives, independent of the affine code), and the two paths' logits
differ by no more than the forward's own run-to-run difference, which is zero when the forward is
deterministic. Logits and tokens against the training engine, whose expert layout differs, are
reported but not required to match bit for bit.
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
from deepspeed.module_inject.auto_ep_layer import AutoEPMoELayer
from deepspeed.runtime.rollout import HybridEngineRollout, RolloutRequest, SamplingConfig

DTYPE = torch.bfloat16
# AutoEP splits the HF fused gate/up projection into w1 (gate) and w3 (up).
HF_EXPERT_NAMES = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}
PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "Mixture-of-experts models route each token to",
    "In 1969, the first humans",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_name", required=True, help="local HF checkpoint directory")
    parser.add_argument("--num_layers", type=int, default=4, help="truncate the model to this many layers")
    parser.add_argument("--train_autoep_size", type=int, default=4)
    parser.add_argument("--train_zero_stage", type=int, default=2, choices=(1, 2, 3))
    parser.add_argument("--rollout_autoep_size", type=int, default=8)
    parser.add_argument("--paths", default="ucp,affine")
    parser.add_argument("--train_steps", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--budget_mb", type=float, default=64.0, help="executor staging bytes per peer")
    parser.add_argument("--work_dir", required=True, help="checkpoint directory for the UCP path")
    parser.add_argument("--output", default="autoep_rollout_e2e.json")
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
    """Wall time of a step, as the slowest rank saw it."""

    def __init__(self):
        self.seconds = {}

    def __call__(self, name):
        timer = self

        class Scope:

            def __enter__(self):
                dist.barrier()
                synchronize()
                self.started = time.perf_counter()

            def __exit__(self, *exc):
                synchronize()
                timer.seconds[name] = max_over_ranks(time.perf_counter() - self.started)

        return Scope()


# ---------------------------------------------------------------------------
# Engines
# ---------------------------------------------------------------------------


def truncated_config(args):
    config = AutoConfig.from_pretrained(args.model_name)
    config.num_hidden_layers = args.num_layers
    return config


def ds_config(zero_stage, autoep_size, lr, load_universal=False):
    config = {
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 1,
        "bf16": {
            "enabled": True
        },
        "zero_optimization": {
            "stage": zero_stage
        },
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": lr,
                "torch_adam": True
            }
        },
        "expert_parallel": {
            "enabled": True,
            "autoep_size": autoep_size
        },
        "zero_allow_untested_optimizer": True,
    }
    if load_universal:
        config["checkpoint"] = {"load_universal": True}
    return config


def initialize(model, config):
    engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config)
    return engine


def build_rollout_engine(args, load_universal):
    """A rollout engine with every weight set to NaN, so that unwritten elements cannot pass."""
    model = AutoModelForCausalLM.from_config(truncated_config(args), dtype=DTYPE)
    with torch.no_grad():
        for param in model.parameters():
            param.fill_(float("nan"))
    # Universal checkpoints load through the ZeRO optimizer, so the rollout engine keeps one. Both
    # paths use the same engine so that their layouts, and therefore their logits, are identical.
    return initialize(model, ds_config(1, args.rollout_autoep_size, args.lr, load_universal))


def autoep_layers(engine):
    return {name: module for name, module in engine.module.named_modules() if isinstance(module, AutoEPMoELayer)}


def expert_param_ids(engine):
    return {id(getattr(layer.experts, name)) for layer in autoep_layers(engine).values() for name in HF_EXPERT_NAMES}


def dense_params(engine):
    expert_ids = expert_param_ids(engine)
    return {name: param for name, param in engine.module.named_parameters() if id(param) not in expert_ids}


def full_parameter(param):
    """The whole local parameter, gathered with ZeRO's own collective when it is partitioned."""
    if hasattr(param, "ds_id"):
        with deepspeed.zero.GatheredParameters([param]):
            return param.data.detach().clone()
    return param.data.detach().clone()


def flat_storage(param, shard_numel):
    # ZeRO-3 keeps the rank's partition in ds_tensor with trailing padding; the map excludes it.
    if hasattr(param, "ds_id"):
        return param.ds_tensor.data.view(-1)[:shard_numel]
    return param.data.view(-1)


def local_expert_ids(layer):
    # AutoEP packs EP rank r's experts as the contiguous range r * L .. (r + 1) * L - 1. The HF
    # weight check below confirms this independently on the training engine.
    count = layer.num_local_experts
    return range(layer.ep_rank * count, (layer.ep_rank + 1) * count)


def gather_experts(layer, name):
    """The layer's whole ``[num_experts, ...]`` tensor, gathered with plain collectives."""
    packed = full_parameter(getattr(layer.experts, name)).contiguous()
    parts = [torch.empty_like(packed) for _ in range(layer.ep_size)]
    dist.all_gather(parts, packed, group=layer.ep_group)
    return torch.cat(parts, dim=0)


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
        with open(os.path.join(path, "model.safetensors.index.json")) as stream:
            self.weight_map = json.load(stream)["weight_map"]
        self.files = {}

    def __contains__(self, key):
        return key in self.weight_map

    def get(self, key):
        file_name = self.weight_map[key]
        if file_name not in self.files:
            self.files[file_name] = safe_open(os.path.join(self.path, file_name), framework="pt", device="cpu")
        return self.files[file_name].get_tensor(key)


def hf_dense_key(name):
    return name.replace(".mlp.router.gate.", ".mlp.gate.")


def check_against_hf(engine, checkpoint):
    """Every training weight equals the HF safetensors tensor of the same name or expert id."""
    check = WeightCheck()
    for name, param in dense_params(engine).items():
        key = hf_dense_key(name)
        if key not in checkpoint:
            check.unmatched_names.append(name)
            continue
        check.add(full_parameter(param), checkpoint.get(key))
    for prefix, layer in autoep_layers(engine).items():
        for name, hf_name in HF_EXPERT_NAMES.items():
            local = full_parameter(getattr(layer.experts, name))
            for index, expert in enumerate(local_expert_ids(layer)):
                check.add(local[index], checkpoint.get(f"{prefix}.experts.{expert}.{hf_name}.weight"))
    return check.result()


def changed_fraction(engine, checkpoint):
    """Fraction of trained weights that differ from the HF checkpoint, to show training moved them."""
    changed, total = 0, 0
    for name, param in dense_params(engine).items():
        value = full_parameter(param)
        changed += bit_mismatches(value, checkpoint.get(hf_dense_key(name)))
        total += value.numel()
    return sum_over_ranks(changed) / max(sum_over_ranks(total), 1)


def check_rollout_weights(rollout, trained):
    """Rollout weights equal the trained ones: dense by name, experts by expert id."""
    check = WeightCheck()
    trained_dense = dense_params(trained)
    for name, param in dense_params(rollout).items():
        check.add(full_parameter(param), full_parameter(trained_dense[name]))
    trained_layers = autoep_layers(trained)
    for prefix, layer in autoep_layers(rollout).items():
        for name in HF_EXPERT_NAMES:
            reference = gather_experts(trained_layers[prefix], name)
            local = full_parameter(getattr(layer.experts, name))
            expected = reference[local_expert_ids(layer).start:local_expert_ids(layer).stop]
            check.add(local, expected)
    return check.result()


# ---------------------------------------------------------------------------
# Weight movement
# ---------------------------------------------------------------------------


def sync_ucp(trained, args, timer):
    tag = "trained"
    save_dir = os.path.join(args.work_dir, "checkpoint")
    with timer("ucp_save"):
        trained.save_checkpoint(save_dir, tag=tag)
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
        rollout.load_checkpoint(save_dir,
                                tag=f"{tag}_universal",
                                load_optimizer_states=False,
                                load_lr_scheduler_states=False)
    return rollout


def sync_affine(trained, args, timer):
    rollout = build_rollout_engine(args, load_universal=False)
    world = dist.get_world_size()
    rank = dist.get_rank()
    endpoints = TransferEndpoints(source={r: r for r in range(world)}, target={r: r for r in range(world)})
    budget = StagingBudget(bytes_per_peer=int(args.budget_mb * (1 << 20)))

    trained_layers = autoep_layers(trained)
    with timer("affine_plan"):
        transfers = []
        for prefix, layer in autoep_layers(rollout).items():
            source_layer = trained_layers[prefix]
            source_maps = source_layer.expert_storage_affine_maps()
            target_maps = layer.expert_storage_affine_maps()
            for name in HF_EXPERT_NAMES:
                plan = plan_transfer(target_maps[name], source_maps[name], PlanBudget())
                source_param = getattr(source_layer.experts, name)
                sources = {}
                if rank in source_maps[name].shard_shapes:
                    numel = math.prod(source_maps[name].shard_shapes[rank])
                    sources[rank] = flat_storage(source_param, numel)
                targets = {}
                if rank in target_maps[name].shard_shapes:
                    shape = target_maps[name].shard_shapes[rank]
                    targets[rank] = getattr(layer.experts, name).data.view(shape)
                transfers.append(TransferRequest(f"{prefix}.{name}", plan, sources, targets))
    with timer("affine_execute"):
        stats = execute_transfers(transfers, endpoints, DTYPE, budget)
    with timer("affine_dense_copy"):
        # Under ZeRO-0/1/2 every rank holds the whole dense weight, so this is a device copy.
        # ZeRO-3 shards it; gathering here stands in for a ZeRO storage map.
        trained_dense = dense_params(trained)
        for name, param in dense_params(rollout).items():
            param.data.copy_(full_parameter(trained_dense[name]))
    return rollout, {
        "sent_bytes": sum_over_ranks(stats.sent_bytes),
        "local_copy_bytes": sum_over_ranks(stats.local_copy_bytes),
        "rounds": reduce_over_ranks(stats.rounds, dist.ReduceOp.MAX),
    }


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
def logits_of(engine, input_ids, attention_mask):
    engine.eval()
    logits = engine.module(input_ids=input_ids, attention_mask=attention_mask).logits.float()
    engine.empty_partition_cache()
    return logits


def all_ranks(value):
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, value)
    return gathered


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
        "bitwise_equal_by_rank": [entry["bitwise_equal"] for entry in by_rank],
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


def train(engine, config, args):
    engine.train()
    generator = torch.Generator().manual_seed(4321 + dist.get_rank())
    losses = []
    for _ in range(args.train_steps):
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

    # 1. HF checkpoint -> training engine.
    with timer("hf_load"):
        model = AutoModelForCausalLM.from_pretrained(args.model_name, config=config, dtype=DTYPE)
    with torch.no_grad():
        model.to(device()).eval()
        hf_logits = model(input_ids=prompt_ids, attention_mask=prompt_mask).logits.float()
    trained = initialize(model, ds_config(args.train_zero_stage, args.train_autoep_size, args.lr))
    oracles["hf_weights"] = check_against_hf(trained, checkpoint)
    oracles["hf_logits"] = compare_logits(logits_of(trained, prompt_ids, prompt_mask), hf_logits, prompt_mask)
    del hf_logits

    # 2. Train.
    with timer("train"):
        report["losses"] = train(trained, config, args)
    report["trained_dense_changed_fraction"] = changed_fraction(trained, checkpoint)

    # Shared-weight rollout on the training engine itself: the reference sequences.
    reference_ids = generate(trained, tokenizer, prompt_ids, prompt_mask, args, timer, "trained")
    reference_mask = (reference_ids != tokenizer.pad_token_id).long()
    reference_mask[:, :prompt_len] = prompt_mask
    trained_logits = logits_of(trained, reference_ids, reference_mask)
    oracles["trained_logits_repeat"] = compare_logits(logits_of(trained, reference_ids, reference_mask),
                                                      trained_logits, reference_mask)

    # 3. Move the weights, 4. roll out.
    rollouts = {}
    for path in paths:
        if path == "ucp":
            rollout = sync_ucp(trained, args, timer)
        elif path == "affine":
            rollout, report["affine_transfer"] = sync_affine(trained, args, timer)
        else:
            raise ValueError(path)
        result = {"weights_vs_trained": check_rollout_weights(rollout, trained)}
        ids = generate(rollout, tokenizer, prompt_ids, prompt_mask, args, timer, path)
        logits = logits_of(rollout, reference_ids, reference_mask)
        # The same engine and inputs a second time: shows whether the forward itself is deterministic.
        result["logits_repeat"] = compare_logits(logits_of(rollout, reference_ids, reference_mask), logits,
                                                 reference_mask)
        result["tokens_vs_trained"] = compare_tokens(ids, reference_ids, prompt_len)
        result["logits_vs_trained"] = compare_logits(logits, trained_logits, reference_mask)
        rollouts[path] = (rollout, ids, logits)
        oracles[path] = result
    if len(rollouts) == 2:
        (_, ids_a, logits_a), (_, ids_b, logits_b) = rollouts.values()
        oracles["ucp_vs_affine"] = {
            "tokens": compare_tokens(ids_a, ids_b, prompt_len),
            "logits": compare_logits(logits_a, logits_b, reference_mask),
        }

    report["seconds"] = timer.seconds
    report["generations"] = {"trained": tokenizer.batch_decode(reference_ids[:, prompt_len:])}
    for path, (_, ids, _) in rollouts.items():
        report["generations"][path] = tokenizer.batch_decode(ids[:, prompt_len:])

    failures = []
    if oracles["hf_weights"]["mismatched_elements"] or oracles["hf_weights"]["unmatched_names"]:
        failures.append("hf_weights")
    if report["trained_dense_changed_fraction"] == 0:
        failures.append("training did not change any weight")
    for path in rollouts:
        if oracles[path]["weights_vs_trained"]["mismatched_elements"]:
            failures.append(f"{path} weights")
    if "ucp_vs_affine" in oracles:
        # Same layout and bitwise equal weights, so the two paths may differ only by the forward's
        # own run-to-run noise: zero where the forward is deterministic.
        repeats = [oracles["trained_logits_repeat"]] + [oracles[path]["logits_repeat"] for path in rollouts]
        noise = max(repeat["max_abs_diff"] for repeat in repeats)
        if oracles["ucp_vs_affine"]["logits"]["max_abs_diff"] > noise:
            failures.append("ucp_vs_affine logits beyond run-to-run noise")
    report["failures"] = failures

    if rank == 0:
        print(json.dumps({"oracles": oracles, "seconds": timer.seconds, "failures": failures}, indent=2))
        with open(args.output, "w") as stream:
            json.dump(report, stream, indent=2, default=str)
        print(f"wrote {args.output}")
    for rollout, _, _ in rollouts.values():
        rollout.destroy()
    trained.destroy()
    if failures:
        raise SystemExit(f"failed: {failures}")


if __name__ == "__main__":
    main()
