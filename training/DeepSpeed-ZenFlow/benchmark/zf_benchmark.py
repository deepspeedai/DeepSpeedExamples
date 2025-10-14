# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import argparse
import torch
import deepspeed.comm as dist
import time

import deepspeed

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


def run_model(model, config_dict, hidden_dim, dtype, pin_memory, topk_ratio, update_interval, overlap_step, iteration):

    model, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config_dict)
    

    data_loader = random_dataloader(model=model,
                                    total_samples=iteration,
                                    hidden_dim=hidden_dim,
                                    device=model.device,
                                    dtype=dtype)

    time_step_list = []
    accumulation_step_time_list = []
    update_step_time_list = []

    dist.barrier()
    for i, batch in enumerate(data_loader):
        step_start_time = time.time()
        loss = model(batch[0], batch[1])
        model.backward(loss)
        model.step()
        step_end_time = time.time()
        step_time = step_end_time - step_start_time
        if dist.get_rank() == 0:
            print(f"Step {i} time: {step_time*1000:.2f}ms")
        if i >= update_interval:
            time_step_list.append(step_time)
            if (i + 1) % update_interval == 0:
                update_step_time_list.append(step_time)
            else:
                accumulation_step_time_list.append(step_time)
        
    if dist.get_rank() == 0:
        with open("zenflow_report.log", "a") as f:
            msg = f"{1 if pin_memory else 0}," \
                f"{topk_ratio}," \
                f"{update_interval}," \
                f"{overlap_step}," \
                f"{sum(accumulation_step_time_list) / len(accumulation_step_time_list):.2f}," \
                f"{sum(update_step_time_list) / len(update_step_time_list):.2f}"
            f.write(f"{msg}\n")
        print(f"[Summary] pin_memory={pin_memory} topk_ratio={topk_ratio} update_interval={update_interval} overlap_step={overlap_step} avg_accumulation_step={sum(accumulation_step_time_list) * 1000 / len(accumulation_step_time_list):.2f}ms avg_update_step={sum(update_step_time_list) * 1000 / len(update_step_time_list):.2f}ms")

    model.destroy()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nlayers", type=int, default=1)
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--dtype", choices=['torch.bfloat16', 'torch.float16', 'torch.float32'], default='torch.bfloat16')
    parser.add_argument("--iteration", type=int, default=5)
    parser.add_argument("--local_rank", type=int, default=-1)

    parser.add_argument("--pin_memory_opts", type=int, required=True)
    parser.add_argument("--topk_ratios", type=float, required=True)
    parser.add_argument("--update_intervals", type=int, required=True)
    parser.add_argument("--overlap_steps", type=int, required=True)

    # Optional: explicitly receive master_port (though deepspeed handles it via env)
    parser.add_argument("--master_port", type=int, default=None)

    args = parser.parse_args()
    dtype = eval(args.dtype)


    pin_memory = bool(args.pin_memory_opts)
    topk_ratio = args.topk_ratios
    update_interval = args.update_intervals
    overlap_step = bool(args.overlap_steps)
    total_iteration = args.iteration * update_interval

    config_dict = {
        "train_micro_batch_size_per_gpu": 1,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 1e-6
            }
        },
        "zero_optimization": {
            "stage": 2,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": pin_memory
            },
            "zenflow": {
                "topk_ratio": topk_ratio,
                "update_interval": update_interval,
                "full_warm_up_rounds": 0,
                "overlap_step": overlap_step
            },
        },
        "wall_clock_breakdown": True,
        "zero_allow_untested_optimizer": True
    }

    if dtype == torch.float16:
        config_dict["fp16"] = {"enabled": True, "initial_scale_power": 8}
    elif dtype == torch.bfloat16:
        config_dict["bf16"] = {"enabled": True}

    model = SimpleModel(args.hidden_dim, nlayers=args.nlayers)
    run_model(model, config_dict, args.hidden_dim, dtype,
                pin_memory, topk_ratio, update_interval, overlap_step,
                total_iteration)


if __name__ == "__main__":
    main()
