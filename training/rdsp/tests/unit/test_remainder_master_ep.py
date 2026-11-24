# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""--master-dtype remainder with expert parallelism: two CPU ranks (gloo), a tiny
Qwen3.5-MoE text model whose experts DeepSpeed's AutoEP splits over the two ranks.
RemainderAdamW then steps DeepSpeed's MoE groups (the experts reduced and split over
their own data-parallel group) and must train as DeepSpeed's own fp32 master does.
Needs DeepSpeed and a transformers with Qwen3.5-MoE."""

import os
import sys

import pytest
import torch

os.environ.setdefault("DS_ACCELERATOR", "cpu")  # not the Mac's MPS
deepspeed = pytest.importorskip("deepspeed")
transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "Qwen3_5MoeForCausalLM"):
    pytest.skip("transformers without Qwen3.5-MoE", allow_module_level=True)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))

VOCAB, SEQ, STEPS, EXPERTS = 160, 16, 3, 8  # VOCAB: no layer width (q_proj is 128)
LR = 1e-2


def tiny_config():
    return transformers.Qwen3_5MoeTextConfig(
        vocab_size=VOCAB,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        num_experts=EXPERTS,
        num_experts_per_tok=2,
        layer_types=["linear_attention", "full_attention"],
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.5,
        },
    )


def train_rank(rank, master, port, out_dir, world, tp):
    """One rank of a run on `world` ranks, AutoTP over `tp` of them and the experts
    split over all (folded onto the TP ranks when tp > 1): 3 steps, then its losses,
    its bf16 weights, its optimizer groups and each group's fp32 master after every step."""
    os.environ.update(
        DS_ACCELERATOR="cpu",
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        WORLD_SIZE=str(world),
        LOCAL_RANK=str(rank),
    )
    import deepspeed.comm.torch  # noqa: F401  (deepspeed.comm's `torch` is torch itself)
    import torch.nn.functional as F

    import ray_deepspeed_pipeline as rdsp
    import train_vl
    from ray_deepspeed_pipeline.deepspeed_adapter import _deepspeed_engine_factory
    from ray_deepspeed_pipeline.hf_stage import build_hf_stage
    from ray_deepspeed_pipeline.partition import partition_parameters
    from ray_deepspeed_pipeline.remainder_adamw import join_master

    # some DeepSpeed versions build a shared-memory op for CPU collectives on init (a C++
    # build through ninja); gloo does not use it
    sys.modules["deepspeed.comm.torch"].build_shm_op = lambda: None
    deepspeed.init_distributed(dist_backend="gloo")
    dp = world // tp
    flags = [
        "--stages",
        "1",
        "--stage",
        f"0:gpus={world}",
        "--zero",
        "1",
        "--lr",
        str(LR),
        "--betas",
        "0.9,0.95",
        "--weight-decay",
        "0.0",
        "--rows",
        str(dp),
        "--microbatches",
        "1",
        "--grad-dtype",
        "bf16",
        "--master-dtype",
        master,
    ]
    args = train_vl.build_parser().parse_args(flags)
    assert train_vl.optimizer_problem(args) is None
    conf = train_vl.ds_config(args)
    conf.update(train_micro_batch_size_per_gpu=1, train_batch_size=dp)
    conf["expert_parallel"] = {"enabled": True, "autoep_size": world}
    if tp > 1:
        conf["tensor_parallel"] = {"autotp_size": tp}
    torch.manual_seed(0)
    model = transformers.Qwen3_5MoeForCausalLM(tiny_config()).to(torch.bfloat16)
    part = partition_parameters(model, rdsp.ExplicitCuts(()), 1)[0]
    stage = build_hf_stage(model, part.block_start, part.block_stop, part.parameter_names)
    engine = _deepspeed_engine_factory(stage, conf)

    opt = engine.optimizer

    def masters():
        out = []
        for m in opt.single_partition_of_fp32_groups:
            state = opt.optimizer.state.get(m, {})
            out.append(
                join_master(m.detach(), state["remainder"])
                if "remainder" in state
                else m.detach().float().clone()
            )
        return out

    g = torch.Generator().manual_seed(10 + rank // tp)  # each TP group its own rows
    losses, stepped = [], []
    for _ in range(STEPS):
        ids = torch.randint(0, VOCAB, (1, SEQ), generator=g)
        logits = engine(ids)
        loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, VOCAB), ids[:, 1:].reshape(-1))
        engine.backward(loss)
        engine.step()
        losses.append(loss.item())
        stepped.append(masters())
    groups = [
        {"moe": bool(grp.get("moe", False)), "name": grp.get("name")}
        for grp in opt.optimizer.param_groups
    ]
    torch.save(
        {
            "losses": losses,
            "groups": groups,
            "masters": stepped,
            "weights": {n: p.detach().float().clone() for n, p in engine.module.named_parameters()},
            "basic": type(engine.basic_optimizer).__name__,
        },
        os.path.join(out_dir, f"{master}-{rank}.pt"),
    )


def run(master, port, out_dir, world=2, tp=1):
    torch.multiprocessing.spawn(
        train_rank, args=(master, port, str(out_dir), world, tp), nprocs=world
    )
    return [torch.load(out_dir / f"{master}-{r}.pt") for r in range(world)]


@pytest.mark.parametrize(
    "world,tp,port", [(2, 1, 29571), (4, 2, 29575)], ids=["ep2", "tp2-ep4-folded"]
)
def test_remainder_master_trains_split_experts_as_the_fp32_master(tmp_path, world, tp, port):
    fp32 = run("fp32", port, tmp_path, world, tp)
    rem = run("remainder", port + 1, tmp_path, world, tp)
    for r in range(world):
        assert rem[r]["basic"] == "RemainderAdamW"
        # the experts in a group of their own, as DeepSpeed's own optimizer has them
        assert [g["moe"] for g in rem[r]["groups"]] == [g["moe"] for g in fp32[r]["groups"]]
        assert any(g["moe"] for g in rem[r]["groups"]) and not all(
            g["moe"] for g in rem[r]["groups"]
        )
        # from the same weights and gradients the first step gives the same fp32 master,
        # bit for bit, in every group
        assert rem[r]["losses"][0] == fp32[r]["losses"][0]
        for i, (a, b) in enumerate(zip(rem[r]["masters"][0], fp32[r]["masters"][0])):
            assert a.shape == b.shape
            assert torch.equal(a.view(torch.int32), b.view(torch.int32)), f"rank {r} group {i}"
        # later steps: the bf16 weights differ where a master sits exactly halfway between
        # two bf16 values (rounded away from zero, not to even), and in bf16 one such weight
        # moves many gradients a little; Adam moves a weight by about lr a step, so the
        # masters stay within 2 lr a step of each other
        assert rem[r]["losses"] == pytest.approx(fp32[r]["losses"], rel=1e-2)
        for i, (a, b) in enumerate(zip(rem[r]["masters"][-1], fp32[r]["masters"][-1])):
            assert float((a - b).abs().max()) < 2 * LR * (STEPS - 1), f"rank {r} group {i}"
        # and the bf16 weights are those masters rounded: within the same, plus one bf16 step
        for name, w in rem[r]["weights"].items():
            bound = 2 * LR * (STEPS - 1) + float(w.abs().max()) * 2**-7
            assert float((w - fp32[r]["weights"][name]).abs().max()) < bound, name
    # each rank holds its own share of the experts: rank 0's and rank `tp`'s (same TP
    # slice, other data-parallel copy) expert weights differ, the rest match
    experts = [n for n in rem[0]["weights"] if ".experts." in n or n.endswith(".experts")]
    assert experts, list(rem[0]["weights"])[:20]
    assert any(not torch.equal(rem[0]["weights"][n], rem[tp]["weights"][n]) for n in experts)
    for name, w in rem[0]["weights"].items():
        if name not in experts:
            assert torch.equal(w, rem[tp]["weights"][name]), name
