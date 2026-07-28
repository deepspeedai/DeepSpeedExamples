"""Isolated student test: AutoTP + ZeRO-3 (+offload) forward/backward/step.

Env knobs:
  STUDENT_MODEL (default local path), OFFLOAD (1/0), AUTOTP (int)
Launch: deepspeed --num_gpus N test_student_autotp_zero3.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import deepspeed
from deepspeed.accelerator import get_accelerator
from transformers import AutoModelForCausalLM

MODEL = os.environ.get("STUDENT_MODEL", "/root/autodl-tmp/models/Qwen2.5-0.5B-Instruct")


def build_ds_config():
    offload = os.environ.get("OFFLOAD", "1") == "1"
    autotp = int(os.environ.get("AUTOTP", "2"))
    z = {"stage": 3}
    if offload:
        z["offload_param"] = {"device": "cpu"}
        z["offload_optimizer"] = {"device": "cpu"}
    return {
        "train_micro_batch_size_per_gpu": 1,
        "bf16": {"enabled": True},
        "zero_optimization": z,
        "tensor_parallel": {"autotp_size": autotp},
        "optimizer": {"type": "AdamW", "params": {"lr": 1e-6, "torch_adam": True}},
    }


def main():
    deepspeed.init_distributed()
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
    engine, *_ = deepspeed.initialize(
        model=model, model_parameters=model.parameters(), config=build_ds_config()
    )

    dev = get_accelerator().current_device_name()
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]], device=dev)
    attn = torch.ones_like(input_ids)
    out = engine(input_ids=input_ids, attention_mask=attn)
    loss = out.logits.float().mean()
    engine.backward(loss)
    engine.step()

    if deepspeed.comm.get_rank() == 0:
        print(
            f"STUDENT_OK loss={loss.item():.4f} "
            f"mem={torch.cuda.max_memory_allocated()/1024**3:.2f}GB "
            f"offload={os.environ.get('OFFLOAD','1')} autotp={os.environ.get('AUTOTP','2')} "
            f"ws={os.environ.get('WORLD_SIZE','1')}",
            flush=True,
        )


if __name__ == "__main__":
    main()
