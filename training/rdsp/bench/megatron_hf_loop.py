# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""Megatron-Core pipeline training from the same pretrained Hugging Face Qwen3
weights and the same token file as rdsp_bench.py, for matched loss and
throughput comparisons.

    torchrun --standalone --nproc_per_node 4 megatron_hf_loop.py \
        --hf /cache/qwen3-0.6b-untied --data /cache/text_s512_b4_m8.pt --steps 300 --out x.json

Weights are mapped by hand (no Megatron-Bridge): HF q/k/v are interleaved per
query group into Megatron's fused linear_qkv, gate/up stacked into linear_fc1,
norms into the TE layer-norm slots. Every Megatron parameter must be written
exactly once or the run aborts.
"""

import argparse
import gc
import json
import os
import time

import torch
import torch.nn.functional as F
from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.distributed.finalize_model_grads import finalize_model_grads
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from safetensors.torch import load_file

p = argparse.ArgumentParser()
p.add_argument("--hf", required=True)
p.add_argument("--data", required=True)
p.add_argument("--steps", type=int, default=300)
p.add_argument("--lr", type=float, default=1e-5)
p.add_argument("--out", required=True)
p.add_argument(
    "--grad_bf16",
    type=int,
    default=0,
    help="sum gradients across microbatches in bf16 (--grad-reduce-in-bf16), "
    "as DeepSpeed ZeRO-0 bf16 does; Megatron's default sums in fp32",
)
p.add_argument(
    "--ce_impl",
    default="native",
    choices=["native", "te"],
    help="Megatron's fused cross-entropy implementation",
)
p.add_argument("--prof_out", default="", help="torch.profiler kernel table for steps [20, 30)")
p.add_argument("--layout", default="", help="Megatron layer layout, e.g. PP=8: 'Et*4|...|t*3,L'")
a = p.parse_args()

H, L, HEADS, GROUPS, HD, FFN, V = 1024, 28, 16, 8, 128, 3072, 151936

torch.distributed.init_process_group("nccl")
local = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local)
pp = torch.distributed.get_world_size()
parallel_state.initialize_model_parallel(1, pp)
model_parallel_cuda_manual_seed(1234)
first, last = parallel_state.is_pipeline_first_stage(), parallel_state.is_pipeline_last_stage()

X = torch.load(a.data)  # [steps, m, b, s+1] token ids, the same file rdsp reads
_, M, B, S1 = X.shape
S = S1 - 1

cfg = TransformerConfig(
    num_layers=L,
    hidden_size=H,
    ffn_hidden_size=FFN,
    num_attention_heads=HEADS,
    num_query_groups=GROUPS,
    kv_channels=HD,
    normalization="RMSNorm",
    layernorm_epsilon=1e-6,
    gated_linear_unit=True,
    activation_func=F.silu,
    add_bias_linear=False,
    add_qkv_bias=False,
    qk_layernorm=True,
    hidden_dropout=0.0,
    attention_dropout=0.0,
    bf16=True,
    params_dtype=torch.bfloat16,
    pipeline_dtype=torch.bfloat16,
    pipeline_model_parallel_size=pp,
    pipeline_model_parallel_layout=a.layout or None,
    # Megatron's standard fusions (pretrain_gpt.py defaults); rdsp runs the
    # matching Liger kernels (RoPE, RMSNorm, SwiGLU, cross-entropy) and fused
    # Adam. Weight-gradient accumulation fusion has no rdsp equivalent;
    # Megatron keeps it.
    apply_rope_fusion=True,
    bias_activation_fusion=True,
    cross_entropy_loss_fusion=True,
    cross_entropy_fusion_impl=a.ce_impl,
    gradient_accumulation_fusion=True,
)
model = GPTModel(
    config=cfg,
    transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(qk_layernorm=True),
    vocab_size=V,
    max_sequence_length=40960,
    pre_process=first,
    post_process=last,
    share_embeddings_and_output_weights=False,
    position_embedding_type="rope",
    rotary_percent=1.0,
    rotary_base=1000000,
).cuda()

hf = load_file(os.path.join(a.hf, "model.safetensors"))
params = dict(model.named_parameters())
written = set()


def put(name, tensor):
    assert name in params, f"no Megatron parameter {name}"
    assert params[name].shape == tensor.shape, (name, params[name].shape, tensor.shape)
    with torch.no_grad():
        params[name].copy_(tensor.to(params[name].dtype))
    written.add(name)


if first:
    put("embedding.word_embeddings.weight", hf["model.embed_tokens.weight"])
if last:
    put("decoder.final_layernorm.weight", hf["model.norm.weight"])
    put("output_layer.weight", hf["lm_head.weight"])
per_group = HEADS // GROUPS
for i, layer in enumerate(model.decoder.layers):
    g = layer.layer_number - 1  # global HF layer index
    pre, loc = f"model.layers.{g}.", f"decoder.layers.{i}."
    q = hf[pre + "self_attn.q_proj.weight"].view(GROUPS, per_group * HD, H)
    k = hf[pre + "self_attn.k_proj.weight"].view(GROUPS, HD, H)
    v = hf[pre + "self_attn.v_proj.weight"].view(GROUPS, HD, H)
    put(loc + "self_attention.linear_qkv.weight", torch.cat([q, k, v], 1).reshape(-1, H))
    put(loc + "self_attention.linear_qkv.layer_norm_weight", hf[pre + "input_layernorm.weight"])
    put(loc + "self_attention.q_layernorm.weight", hf[pre + "self_attn.q_norm.weight"])
    put(loc + "self_attention.k_layernorm.weight", hf[pre + "self_attn.k_norm.weight"])
    put(loc + "self_attention.linear_proj.weight", hf[pre + "self_attn.o_proj.weight"])
    put(
        loc + "mlp.linear_fc1.weight",
        torch.cat([hf[pre + "mlp.gate_proj.weight"], hf[pre + "mlp.up_proj.weight"]], 0),
    )
    put(loc + "mlp.linear_fc1.layer_norm_weight", hf[pre + "post_attention_layernorm.weight"])
    put(loc + "mlp.linear_fc2.weight", hf[pre + "mlp.down_proj.weight"])
missing = set(params) - written
assert not missing, f"Megatron parameters with no HF weight: {sorted(missing)[:8]}"
# every HF layer lives on exactly one rank (catches a wrong layout / offset)
mine = torch.zeros(L, device="cuda")
for layer in model.decoder.layers:
    mine[layer.layer_number - 1] += 1
torch.distributed.all_reduce(mine)
assert bool((mine == 1).all()), f"layer coverage {mine.tolist()}"
print(
    f"rank {torch.distributed.get_rank()}: layers "
    f"{[layer.layer_number - 1 for layer in model.decoder.layers]}",
    flush=True,
)
del hf

ddp = DistributedDataParallel(
    config=cfg,
    ddp_config=DistributedDataParallelConfig(
        grad_reduce_in_fp32=not a.grad_bf16,
        overlap_grad_reduce=False,
        use_distributed_optimizer=False,
    ),
    module=model,
)
opt = get_megatron_optimizer(
    OptimizerConfig(
        optimizer="adam",
        lr=a.lr,
        min_lr=a.lr,
        weight_decay=0.0,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_eps=1e-8,
        clip_grad=0.0,
        bf16=True,
        use_distributed_optimizer=False,
    ),
    [ddp],
)
# printed so each run's log records which gradient-summing precision it used
print(f"main_grad dtype: {next(p for p in model.parameters()).main_grad.dtype}", flush=True)
for group in opt.param_groups:
    group["lr"], group["weight_decay"] = a.lr, 0.0
cfg.finalize_model_grads_func = finalize_model_grads
cfg.no_sync_func = ddp.no_sync

pos = torch.arange(S, device="cuda").unsqueeze(0).expand(B, -1)


def batches(step):
    for k in range(M):
        x = X[step, k]
        yield x[:, :-1].contiguous().cuda(), x[:, 1:].contiguous().cuda()


def forward_step(it, mdl):
    tokens, labels = next(it)
    out = mdl(tokens, pos, None, labels=labels)  # per-token CE on the last stage

    def loss_func(o):
        loss = o.float().mean()  # mean over this microbatch's tokens, like rdsp's loss_fn
        # clone: Megatron divides `loss` by num_microbatches IN PLACE afterwards,
        # and a detached view would share that storage (logged value /M)
        return loss, {"lm loss": loss.detach().clone()}

    return out, loss_func


fb = get_forward_backward_func()
gc.disable()  # Megatron's --manual-gc, as in the M-defaults pretrain_gpt.py runs
losses, times = [], []
prof, t_prof = None, 0.0
for step in range(a.steps):
    if a.prof_out and step == 20:  # same window and tool as bench_hooks.ProfiledEngine
        torch.cuda.synchronize()
        prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        )
        prof.__enter__()
        t_prof = time.perf_counter()
    if prof is not None and step == 30:
        torch.cuda.synchronize()
        wall = time.perf_counter() - t_prof
        prof.__exit__(None, None, None)
        if torch.distributed.get_rank() == 0:
            from bench_hooks import kernel_table

            with open(a.prof_out, "w") as f:
                json.dump({"steps": 10, "wall_s": wall, "kernels": kernel_table(prof)}, f)
        prof = None
    ddp.zero_grad_buffer()
    opt.zero_grad()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fb(
        forward_step_func=forward_step,
        data_iterator=batches(step),
        model=ddp,
        num_microbatches=M,
        seq_length=S,
        micro_batch_size=B,
        forward_only=False,
    )
    opt.step()
    torch.cuda.synchronize()
    times.append(time.perf_counter() - t0)
    if last:
        losses.append(sum(float(d["lm loss"]) for d in out) / M)
        if step % 25 == 0:
            print(
                f"megatron step {step} loss {losses[-1]:.4f} {times[-1] * 1000:.0f} ms", flush=True
            )

if last:
    with open(a.out, "w") as f:
        json.dump(
            {
                "system": "megatron",
                "pp": pp,
                "m": M,
                "b": B,
                "seq": S,
                "layout": a.layout,
                "grad_bf16": a.grad_bf16,
                "ce_impl": a.ce_impl,
                "loss": losses,
                "step_s": times,
            },
            f,
        )
torch.distributed.barrier()
torch.distributed.destroy_process_group()
