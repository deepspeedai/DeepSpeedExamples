# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import time
import argparse

import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
import torch

import deepspeed
from deepspeed.runtime.zero.offload_config import OffloadDeviceEnum, OffloadStateTypeEnum


class SimpleModel(torch.nn.Module):

    def __init__(self, hidden_dim, empty_grad=False, nlayers=1):
        super(SimpleModel, self).__init__()
        self.linears = torch.nn.ModuleList([torch.nn.Linear(hidden_dim, hidden_dim) for _ in range(nlayers)])
        if empty_grad:
            self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

    def forward(self, x, y):
        for l in self.linears:
            x = l(x)
        return self.cross_entropy_loss(x, y)


def random_dataset(total_samples, hidden_dim, device, dtype):
    train_data = torch.randn(total_samples, hidden_dim, device=device, dtype=dtype)
    train_label = torch.empty(total_samples, dtype=torch.long, device=device).random_(hidden_dim)
    train_dataset = torch.utils.data.TensorDataset(train_data, train_label)
    return train_dataset


def random_dataloader(model, total_samples, hidden_dim, device, dtype):
    batch_size = model.train_micro_batch_size_per_gpu()
    train_dataset = random_dataset(total_samples, hidden_dim, device, dtype=dtype)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    return train_loader


def run_model(model, config_dict, hidden_dim, dtype, include, pin_memory, non_blocking, iteration, warmup):
    model, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config_dict)
    data_loader = random_dataloader(model=model,
                                    total_samples=iteration,
                                    hidden_dim=hidden_dim,
                                    device=model.device,
                                    dtype=dtype)

    time_offload_list = []
    time_load_list = []

    dist.barrier()
    for i, batch in enumerate(data_loader):
        loss = model(batch[0], batch[1])
        model.backward(loss)
        model.step()

        # Start offloading
        alloc_before_offload = get_accelerator().memory_allocated()
        dist.barrier()

        time_start = time.time()
        model.offload_states(include=include,
                             device=OffloadDeviceEnum.cpu,
                             pin_memory=pin_memory,
                             non_blocking=non_blocking)
        dist.barrier()
        time_after_offload = time.time()
        alloc_after_offload = get_accelerator().memory_allocated()
        assert alloc_after_offload < alloc_before_offload, f"Allocated memory should decrease after offload"

        # Load offloaded states back
        model.reload_states()
        dist.barrier()
        time_after_load = time.time()

        time_offload_list.append(time_after_offload - time_start)
        time_load_list.append(time_after_load - time_after_offload)

        assert alloc_after_offload < get_accelerator().memory_allocated(
        ), f"Allocated memory should increase after offload back"

        if dist.get_rank() == 0:
            print(
                f"Memory usage ({i}): include={include}, pin_memory={pin_memory}, non_blocking={non_blocking} alloc_before_offload={alloc_before_offload} alloc_after_offload={alloc_after_offload}"
            )

    # remove warmup
    time_offload_list = time_offload_list[warmup:]
    time_load_list = time_load_list[warmup:]

    if dist.get_rank() == 0:
        with open("offload_states.log", "a") as f:
            offload_time = sum(time_offload_list) / len(time_offload_list)
            load_time = sum(time_load_list) / len(time_load_list)
            msg = f"{1 if pin_memory else 0},{1 if non_blocking else 0},{offload_time},{load_time}"
            f.write(f"{msg}\n")
        print(f"Summary: pin_memory={pin_memory} non_blocking={non_blocking} offload={offload_time} load={load_time}")

    # Needed in ZeRO 3. Not doing so can give memory leak
    model.destroy()


def main():
    parser = argparse.ArgumentParser(description="Test Offload States")
    parser.add_argument("--included_state", type=str, choices=[e.name for e in OffloadStateTypeEnum] + [None], default=None, help="State to include")
    parser.add_argument("--pin_memory", action='store_true', help="Pin memory")
    parser.add_argument("--non_blocking", action='store_true', help="Non blocking")
    parser.add_argument("--nlayers", type=int, default=1, help="Number of layers")
    parser.add_argument("--hidden_dim", type=int, default=1024, help="Hidden dimension")
    parser.add_argument('--dtype', choices=['torch.bfloat16', 'torch.float16', 'torch.float32'], default='torch.bfloat16', help='Data type')
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank")
    parser.add_argument("--iteration", type=int, default=10, help="Warmup")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup")

    args = parser.parse_args()

    dtype = eval(args.dtype)
    hidden_dim = args.hidden_dim

    config_dict = {
        "train_micro_batch_size_per_gpu": 1,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 1e-6
            }
        },
        "zero_optimization": {
            "stage": 3,
        },
    }

    if dtype == torch.float16:
        config_dict["fp16"] = {"enabled": True, "initial_scale_power": 8}
    elif dtype == torch.bfloat16:
        config_dict["bf16"] = {"enabled": True}

    with deepspeed.zero.Init(config_dict_or_path=config_dict):
        model = SimpleModel(hidden_dim, nlayers=args.nlayers)

    included_state = None if args.included_state is None else [OffloadStateTypeEnum[args.included_state]]
    run_model(model, config_dict, hidden_dim, dtype, included_state, args.pin_memory, args.non_blocking, args.iteration, args.warmup)


if __name__ == "__main__":
    main()
