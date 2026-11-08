# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""GRPO retool training entry point.

Launch with the DeepSpeed launcher, or with a script in ``scripts/``::

    deepspeed --num_gpus 8 main.py --config configs/qwen3_8b_8gpu.json

The DeepSpeed launcher sets ``LOCAL_RANK``, ``RANK``, and ``WORLD_SIZE`` in
the environment; we call :func:`deepspeed.init_distributed` to take that over.
"""

import argparse
import os
import random

import deepspeed
import numpy as np
import torch
from deepspeed import comm as dist
from deepspeed.accelerator import get_accelerator
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import GRPOConfig
from data import PromptDataset, collate_rows, load_prompts
from deepspeed.runtime.rollout import build_rollout
from policy import PolicyModel, RolloutEngineView, register_attention
from tools import CodeInterpreter
from trainer import GRPOTrainer


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if get_accelerator().is_available():
        get_accelerator().manual_seed_all(seed)


def _resolve_dtype(name: str) -> torch.dtype:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to GRPOConfig JSON")
    parser.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", 0)))
    args = parser.parse_args()

    deepspeed.init_distributed()
    world_size = dist.get_world_size()

    cfg = GRPOConfig.from_json(args.config)
    cfg.validate(world_size)
    _seed_everything(cfg.training.seed)

    # --- tokenizer (shared between rollout and training) -------------------
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name_or_path,
                                              trust_remote_code=cfg.model.trust_remote_code,
                                              padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- policy + DeepSpeed hybrid engine ----------------------------------
    register_attention()
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.model_name_or_path,
        dtype=_resolve_dtype(cfg.model.dtype),
        attn_implementation=cfg.model.attn_implementation,
        trust_remote_code=cfg.model.trust_remote_code,
    )
    if cfg.model.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    policy = PolicyModel(model, chunk_size=cfg.algorithm.logprob_chunk_size)

    # Every step is one optimizer update over all of a rank's trajectories.
    samples_per_rank = cfg.training.global_batch_size // world_size
    prompts_per_rank = samples_per_rank // cfg.rollout.n_samples_per_prompt
    ds_config = cfg.load_deepspeed_config()
    ds_config["train_micro_batch_size_per_gpu"] = cfg.training.micro_batch_size // world_size
    ds_config["gradient_accumulation_steps"] = cfg.training.global_batch_size // cfg.training.micro_batch_size
    ds_config.pop("train_batch_size", None)
    if not ds_config.get("hybrid_engine", {}).get("enabled", False):
        raise ValueError("This example rolls out with the hybrid engine; enable it in the DeepSpeed config")

    engine, *_ = deepspeed.initialize(model=policy, model_parameters=policy.parameters(), config=ds_config)

    # --- rollout engine with the code interpreter --------------------------
    tool = CodeInterpreter(timeout=cfg.tool.timeout,
                           memory_limit_gb=cfg.tool.memory_limit_gb,
                           max_concurrency=cfg.tool.max_concurrency,
                           max_output_chars=cfg.tool.max_output_chars,
                           show_last_value=cfg.tool.show_last_value)
    rollout = build_rollout(cfg.rollout,
                            generation_engine=RolloutEngineView(engine),
                            tokenizer=tokenizer,
                            tools=[tool])

    # --- data: each rank rolls out its own share of every step's prompts ----
    dataset = PromptDataset(cfg.data.path, cfg.data.prompt_key, cfg.data.label_key)
    sampler = DistributedSampler(dataset,
                                 num_replicas=world_size,
                                 rank=dist.get_rank(),
                                 shuffle=cfg.data.shuffle,
                                 seed=cfg.training.seed)
    loader = DataLoader(dataset, batch_size=prompts_per_rank, sampler=sampler, collate_fn=collate_rows, drop_last=True)
    eval_rows = load_prompts(cfg.eval.path, cfg.data.prompt_key, cfg.data.label_key) if cfg.eval.path else []

    trainer = GRPOTrainer(
        cfg=cfg,
        engine=engine,
        tokenizer=tokenizer,
        rollout=rollout,
        dataloader=loader,
        eval_rows=eval_rows,
    )
    if cfg.training.resume_from:
        trainer.resume(cfg.training.resume_from)
    trainer.train()


if __name__ == "__main__":
    main()
