# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""OPSD training entry point.

Launch with the DeepSpeed launcher::

    deepspeed --num_gpus 8 main.py --config configs/opsd_hybrid_engine.json

The DeepSpeed launcher sets ``LOCAL_RANK``, ``RANK``, and ``WORLD_SIZE`` in
the environment; we call :func:`deepspeed.init_distributed` to take that over.
"""

import argparse
import json
import os
import random

import deepspeed
import numpy as np
import torch
from deepspeed.accelerator import get_accelerator
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import OPSDConfig
from data import LeftPaddedPromptCollator, PromptDataset
from deepspeed.runtime.rollout import build_rollout
from teacher import TeacherWrapper
from trainer import OPSDTrainer
from losses import require_vocab_parallel_kl


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if get_accelerator().is_available():
        get_accelerator().manual_seed_all(seed)


def _resolve_dtype(name: str) -> torch.dtype:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def _load_ds_config(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _feature_enabled(value):
    return value.get("enabled", False) if isinstance(value, dict) else bool(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to OPSDConfig JSON")
    parser.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", 0)))
    args = parser.parse_args()

    cfg = OPSDConfig.from_json(args.config)
    cfg.validate()
    _seed_everything(cfg.training.seed)

    deepspeed.init_distributed()

    # --- tokenizer (shared between data + rollout) -------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.student.model_name_or_path,
        trust_remote_code=cfg.student.trust_remote_code,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- student model + DeepSpeed engine ----------------------------------
    student_dtype = _resolve_dtype(cfg.student.dtype)
    student_model = AutoModelForCausalLM.from_pretrained(
        cfg.student.model_name_or_path,
        dtype=student_dtype,
        trust_remote_code=cfg.student.trust_remote_code,
    )

    ds_config = _load_ds_config(cfg.deepspeed_config)
    vocab_parallel = cfg.distillation.loss_backend == "vocab_parallel"
    if vocab_parallel:
        require_vocab_parallel_kl()
        tp = ds_config.setdefault("tensor_parallel", {}).get("autotp_size", 1)
        if cfg.teacher.autotp_size != tp:
            raise ValueError("vocab_parallel requires equal student and teacher AutoTP sizes")
        if cfg.teacher.offload_to_cpu:
            raise ValueError("vocab_parallel teacher CPU offload is not validated; disable offload_to_cpu")
        if ds_config.get("zero_optimization", {}).get("stage", 0) not in (0, 2):
            raise ValueError("vocab_parallel currently supports student ZeRO-0 and ZeRO-2")
        if _feature_enabled(ds_config.get("hybrid_engine", {})) or _feature_enabled(ds_config.get("compile", {})):
            raise ValueError("vocab_parallel requires eager HF rollout and uncompiled training")
        if ds_config.get("pipeline", {}).get("pipeline_parallel_size", 1) != 1:
            raise ValueError("vocab_parallel pipeline parallelism is not validated")
        if ds_config.get("sequence_parallel_size", 1) != 1:
            raise ValueError("vocab_parallel sequence parallelism is not validated")
        ds_config["tensor_parallel"]["vocab_parallel_lm_head"] = True
    elif ds_config.get("tensor_parallel", {}).get("autotp_size", 1) > 1:
        if ds_config["tensor_parallel"].get("vocab_parallel_lm_head") is True:
            raise ValueError("Dense distillation requires full vocabulary logits; select vocab_parallel for sharded heads")
        ds_config["tensor_parallel"]["vocab_parallel_lm_head"] = False
    ds_config["train_micro_batch_size_per_gpu"] = cfg.training.micro_batch_size_per_gpu
    ds_config["gradient_accumulation_steps"] = cfg.training.gradient_accumulation_steps

    student_engine, *_ = deepspeed.initialize(
        model=student_model,
        model_parameters=student_model.parameters(),
        config=ds_config,
    )

    # --- frozen teacher ----------------------------------------------------
    teacher = TeacherWrapper(cfg.teacher, world_size=dist_world_size(), vocab_parallel=vocab_parallel)
    if vocab_parallel:
        from vocab_parallel import prepare_vocab_parallel

        teacher_tokenizer = AutoTokenizer.from_pretrained(cfg.teacher.model_name_or_path,
                                                         trust_remote_code=cfg.teacher.trust_remote_code)
        if teacher_tokenizer.pad_token is None:
            teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
        student_layout, teacher.vocab_layout = prepare_vocab_parallel(
            student_engine.module, teacher._callable.module, tokenizer, teacher_tokenizer)
        student_engine._opsd_vocab_layout = student_layout
    elif any(getattr(model.get_output_embeddings(), "is_vocab_parallel_lm_head", False)
             for model in [student_engine.module, teacher._callable.module]):
        raise ValueError("Dense distillation cannot consume vocabulary shards; select vocab_parallel or disable sharded heads")

    # --- rollout engine ----------------------------------------------------
    rollout = build_rollout(
        cfg.rollout,
        student_engine=student_engine,
        tokenizer=tokenizer,
        student_model_path=cfg.student.model_name_or_path,
    )

    # --- dataloader --------------------------------------------------------
    dataset = PromptDataset(
        path=cfg.data.path,
        tokenizer=tokenizer,
        max_prompt_length=cfg.rollout.max_prompt_length,
        prompt_field=cfg.data.prompt_field,
        chat_template=cfg.data.chat_template,
    )
    collator = LeftPaddedPromptCollator(tokenizer=tokenizer, max_prompt_length=cfg.rollout.max_prompt_length)
    # In pure-TP mode (autotp_size == world_size), every rank must see the
    # same data — a DistributedSampler would split prompts across TP ranks
    # and deadlock the TP all-reduce.  Only shard when there is real DP.
    tp_size = student_engine.autotp_size() if hasattr(student_engine, 'autotp_size') else 1
    tp_size = tp_size or 1  # autotp_size() returns 0 when AutoTP is unconfigured
    dp_size = dist_world_size() // tp_size
    if dp_size > 1:
        # Use the engine's actual DP group so TP peers consume the same
        # prompts without assuming a global-rank ordering.
        sampler = DistributedSampler(
            dataset,
            shuffle=cfg.data.shuffle,
            num_replicas=dp_size,
            rank=deepspeed.comm.get_rank(student_engine.data_parallel_group),
        )
    else:
        sampler = None
    loader = DataLoader(
        dataset,
        batch_size=cfg.training.micro_batch_size_per_gpu,
        shuffle=cfg.data.shuffle if sampler is None else False,
        sampler=sampler,
        collate_fn=collator,
        drop_last=True,
    )

    OPSDTrainer(
        cfg=cfg,
        student_engine=student_engine,
        teacher=teacher,
        tokenizer=tokenizer,
        rollout=rollout,
        dataloader=loader,
    ).train()


def dist_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


if __name__ == "__main__":
    main()
