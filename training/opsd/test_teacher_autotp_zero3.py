"""Isolated teacher test: AutoTP + ZeRO-3 (offload) forward_to_cache.

Env knobs:
  TEACHER_MODEL (default local path), OFFLOAD (1/0), AUTOTP (int)
Launch: deepspeed --num_gpus N test_teacher_autotp_zero3.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import deepspeed
from deepspeed.accelerator import get_accelerator

from teacher import TeacherWrapper
from config import TeacherConfig

MODEL = os.environ.get("TEACHER_MODEL", "/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct")


def main():
    deepspeed.init_distributed()
    cfg = TeacherConfig(
        model_name_or_path=MODEL,
        dtype="bfloat16",
        offload_to_cpu=os.environ.get("OFFLOAD", "1") == "1",
        autotp_size=int(os.environ.get("AUTOTP", "2")),
    )
    ws = int(os.environ.get("WORLD_SIZE", 1))
    teacher = TeacherWrapper(cfg, world_size=ws)

    dev = get_accelerator().current_device_name()
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]], device=dev)
    attn = torch.ones_like(input_ids)
    cache = teacher.forward_to_cache(input_ids, attn)

    if deepspeed.comm.get_rank() == 0:
        print(
            f"TEACHER_OK shape={cache.shape} "
            f"mem={torch.cuda.max_memory_allocated()/1024**3:.2f}GB "
            f"offload={cfg.offload_to_cpu} autotp={cfg.autotp_size} ws={ws}",
            flush=True,
        )
    cache.free()


if __name__ == "__main__":
    main()
