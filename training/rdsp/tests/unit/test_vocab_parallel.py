# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""Vocab-parallel embedding and output head: two processes stand in for two
tensor-parallel ranks; each holds half of the vocabulary rows. Outputs and
gradients must equal the unsplit modules'."""

import os
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F

from ray_deepspeed_pipeline.vocab_parallel import next_token_loss_sum, shard_vocab

VOCAB, HIDDEN, TP = 12, 4, 2


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(VOCAB, HIDDEN)
        self.pos = nn.Embedding(5, HIDDEN)  # not vocab-sized: left alone
        self.lm_head = nn.Linear(HIDDEN, VOCAB, bias=False)

    def forward(self, ids):
        return self.lm_head(self.embed_tokens(ids) + self.pos.weight[: ids.shape[1]])


def _rank(rank, init_file, out_dir):
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=TP)
    torch.manual_seed(0)
    full = Tiny()
    sharded = Tiny()
    sharded.load_state_dict(full.state_dict())
    shard_vocab(sharded, VOCAB, rank=rank, world=TP, group=lambda: dist.group.WORLD)
    ids = torch.tensor([[0, 5, 6, 11]])
    target = torch.randn(1, 4, VOCAB, generator=torch.Generator().manual_seed(1))
    (full(ids) * target).sum().backward()
    out = sharded(ids)
    (out * target).sum().backward()
    rows = slice(rank * VOCAB // TP, (rank + 1) * VOCAB // TP)
    torch.save(
        {
            "out_close": torch.allclose(out, full(ids), atol=1e-6),
            "embed_grad": torch.allclose(
                sharded.embed_tokens.weight.grad, full.embed_tokens.weight.grad[rows], atol=1e-6
            ),
            "head_grad": torch.allclose(
                sharded.lm_head.weight.grad, full.lm_head.weight.grad[rows], atol=1e-6
            ),
            "pos_whole": sharded.pos.weight.shape == (5, HIDDEN),
            "shapes": (
                tuple(sharded.embed_tokens.weight.shape),
                tuple(sharded.lm_head.weight.shape),
            ),
        },
        os.path.join(out_dir, f"r{rank}.pt"),
    )
    dist.destroy_process_group()


def test_sharded_embedding_and_head_match_the_unsplit_modules():
    with tempfile.TemporaryDirectory() as d:
        mp.spawn(_rank, args=(os.path.join(d, "init"), d), nprocs=TP)
        for rank in range(TP):
            r = torch.load(os.path.join(d, f"r{rank}.pt"))
            assert r["shapes"] == ((VOCAB // TP, HIDDEN), (VOCAB // TP, HIDDEN))
            assert r["out_close"] and r["embed_grad"] and r["head_grad"] and r["pos_whole"]


def _rank_sharded_loss(rank, init_file, out_dir):
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=TP)
    torch.manual_seed(0)
    full = Tiny()
    sharded = Tiny()
    sharded.load_state_dict(full.state_dict())
    shard_vocab(
        sharded, VOCAB, rank=rank, world=TP, group=lambda: dist.group.WORLD, gather_logits=False
    )
    ids = torch.tensor([[0, 5, 6, 11], [3, 3, 9, 1]])
    labels = torch.tensor([[-100, 5, 7, 11], [-100, 2, -100, 6]])  # 0..5 on rank 0
    expected = F.cross_entropy(
        full(ids)[:, :-1].reshape(-1, VOCAB),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )
    expected.backward()
    logits = sharded(ids)
    got = next_token_loss_sum(logits, labels)
    got.backward()
    rows = slice(rank * VOCAB // TP, (rank + 1) * VOCAB // TP)
    torch.save(
        {
            "local_width": logits.shape[-1],
            "loss_close": torch.allclose(got, expected, atol=1e-5),
            "head_grad": torch.allclose(
                sharded.lm_head.weight.grad, full.lm_head.weight.grad[rows], atol=1e-5
            ),
            "embed_grad": torch.allclose(
                sharded.embed_tokens.weight.grad, full.embed_tokens.weight.grad[rows], atol=1e-5
            ),
            "pos_grad": torch.allclose(sharded.pos.weight.grad, full.pos.weight.grad, atol=1e-5),
            "full_logits": torch.allclose(next_token_loss_sum(full(ids), labels), expected),
        },
        os.path.join(out_dir, f"r{rank}.pt"),
    )
    dist.destroy_process_group()


def test_vocab_parallel_loss_matches_the_full_vocabulary_loss():
    """The head keeps its shard's logits; the loss exchanges only per-token
    max, exp-sum and target logit, and equals the full-vocabulary loss, with
    the same gradients. The same loss takes ordinary full logits too."""
    with tempfile.TemporaryDirectory() as d:
        mp.spawn(_rank_sharded_loss, args=(os.path.join(d, "init"), d), nprocs=TP)
        for rank in range(TP):
            r = torch.load(os.path.join(d, f"r{rank}.pt"))
            assert r["local_width"] == VOCAB // TP
            assert all(
                r[k] for k in ("loss_close", "head_grad", "embed_grad", "pos_grad", "full_logits")
            ), r


def test_a_token_mean_loss_takes_shards_when_its_sum_does():
    from ray_deepspeed_pipeline import TokenMeanLoss

    assert TokenMeanLoss(next_token_loss_sum, len).takes_vocab_shards
    assert not TokenMeanLoss(lambda logits, labels: 0, len).takes_vocab_shards


def _rank_tied(rank, init_file, out_dir):
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=TP)
    torch.manual_seed(0)
    full = Tiny()
    full.lm_head.weight = full.embed_tokens.weight
    sharded = Tiny()
    sharded.lm_head.weight = sharded.embed_tokens.weight
    sharded.load_state_dict(full.state_dict())
    shard_vocab(
        sharded, VOCAB, rank=rank, world=TP, group=lambda: dist.group.WORLD, gather_logits=False
    )
    ids = torch.tensor([[0, 5, 6, 11]])
    labels = torch.tensor([[-100, 5, 7, 11]])
    next_token_loss_sum(full(ids), labels).backward()
    next_token_loss_sum(sharded(ids), labels).backward()
    rows = slice(rank * VOCAB // TP, (rank + 1) * VOCAB // TP)
    torch.save(
        {
            "shared": sharded.lm_head.weight is sharded.embed_tokens.weight,
            "params": len(list(sharded.parameters())),
            "grad": torch.allclose(
                sharded.embed_tokens.weight.grad, full.embed_tokens.weight.grad[rows], atol=1e-5
            ),
        },
        os.path.join(out_dir, f"r{rank}.pt"),
    )
    dist.destroy_process_group()


def test_tied_embedding_and_head_stay_one_parameter():
    """A head tied to the embedding keeps sharing its rows after sharding:
    one parameter, whose gradient sums both uses, as in the unsplit model."""
    with tempfile.TemporaryDirectory() as d:
        mp.spawn(_rank_tied, args=(os.path.join(d, "init"), d), nprocs=TP)
        for rank in range(TP):
            r = torch.load(os.path.join(d, f"r{rank}.pt"))
            assert r["shared"] and r["params"] == 2 and r["grad"], r
