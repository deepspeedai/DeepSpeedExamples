# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""Vocab-parallel input embedding and output head for tensor-parallel stages.

A large vocabulary makes the embedding and the output head the biggest single
matrices in a stage (Qwen3.5: 248k x hidden each). AutoTP leaves them whole
on every tensor-parallel rank, so each rank carries their full weights,
gradients and optimizer state. Here each rank keeps 1/tp of the vocabulary
rows instead, as Megatron does:

- embedding: a rank looks up the token ids in its row range, zeros the rest,
  and the ranks' outputs are summed (each id is in exactly one range);
- head: a rank computes the logits of its rows, and they are gathered along
  the vocabulary, so the loss sees ordinary full logits. With a loss that
  takes vocabulary shards (next_token_loss_sum), the head skips the gather:
  each rank keeps its rows' logits and the loss exchanges only three numbers
  per token (VocabShard below).
"""

from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


class SumOverRanks(torch.autograd.Function):
    """Sum over the group forward; backward passes the gradient on as is
    (every rank's output feeds the same downstream computation)."""

    @staticmethod
    def forward(ctx, x, group):
        x = x.clone()
        dist.all_reduce(x, group=group)
        return x

    @staticmethod
    def backward(ctx, grad):
        return grad, None


class SumGradOverRanks(torch.autograd.Function):
    """Identity forward; backward sums the gradient over the group: each
    rank's head shard sees only its vocabulary rows' share of the input
    gradient."""

    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        grad = grad.contiguous().clone()
        dist.all_reduce(grad, group=ctx.group)
        return grad, None


class _GatherLastDim(torch.autograd.Function):
    """Concatenate the ranks' tensors along the last dim; backward keeps this
    rank's slice of the gradient."""

    @staticmethod
    def forward(ctx, x, group):
        ctx.group, ctx.width = group, x.shape[-1]
        parts = [torch.empty_like(x) for _ in range(dist.get_world_size(group))]
        dist.all_gather(parts, x.contiguous(), group=group)
        return torch.cat(parts, dim=-1)

    @staticmethod
    def backward(ctx, grad):
        rank = dist.get_rank(ctx.group)
        return grad[..., rank * ctx.width : (rank + 1) * ctx.width].contiguous(), None


@dataclass(frozen=True)
class VocabShard:
    """Which vocabulary rows a logits tensor holds: [start, start + width)."""

    start: int
    group: object  # the tensor-parallel process group


class _ShardedCrossEntropy(torch.autograd.Function):
    """Per-token cross entropy over logits split along the vocabulary. The
    ranks agree on each token's max logit, exp-sum and target logit; the
    gradient, softmax minus the target's one-hot, needs no exchange and is
    kept from forward in the logits' dtype."""

    @staticmethod
    def forward(ctx, logits, target, start, group, ignore_index):
        x = logits.float()
        top = x.max(dim=-1).values
        dist.all_reduce(top, op=dist.ReduceOp.MAX, group=group)
        x.sub_(top.unsqueeze(-1))
        local = target - start
        mine = (local >= 0) & (local < x.shape[-1])
        index = local.clamp(0, x.shape[-1] - 1).unsqueeze(-1)
        picked = x.gather(-1, index).squeeze(-1).masked_fill(~mine, 0)
        dist.all_reduce(picked, group=group)
        x.exp_()
        total = x.sum(dim=-1)
        dist.all_reduce(total, group=group)
        counted = target != ignore_index
        loss = (total.log() - picked) * counted
        x.div_(total.unsqueeze(-1))
        x.scatter_add_(-1, index, -mine.to(x.dtype).unsqueeze(-1))
        x.mul_(counted.unsqueeze(-1))
        ctx.save_for_backward(x.to(logits.dtype))
        return loss

    @staticmethod
    def backward(ctx, grad):
        (softmax_minus_target,) = ctx.saved_tensors
        return (
            softmax_minus_target * grad.unsqueeze(-1).to(softmax_minus_target.dtype),
            None,
            None,
            None,
            None,
        )


def next_token_loss_sum(logits, labels, ignore_index: int = -100):
    """Summed next-token cross entropy (position t predicts labels[t + 1];
    ignore_index labels do not count), for full logits or a vocab-parallel
    head's shard. As a stage's loss (or a TokenMeanLoss's sum_fn) it lets
    tensor-parallel stages skip gathering the logits."""
    shard = getattr(logits, "vocab_shard", None)
    logits, target = logits[:, :-1], labels[:, 1:].to(logits.device)
    width = logits.shape[-1]
    if shard is None:
        return F.cross_entropy(
            logits.float().reshape(-1, width),
            target.reshape(-1),
            ignore_index=ignore_index,
            reduction="sum",
        )
    return _ShardedCrossEntropy.apply(
        logits.reshape(-1, width), target.reshape(-1), shard.start, shard.group, ignore_index
    ).sum()


next_token_loss_sum.takes_vocab_shards = True


class VocabShardedEmbedding(nn.Module):
    def __init__(self, embedding: nn.Embedding, rank: int, world: int, group):
        super().__init__()
        rows = embedding.num_embeddings // world
        self.start, self.stop = rank * rows, (rank + 1) * rows
        self.num_embeddings, self.embedding_dim = embedding.num_embeddings, embedding.embedding_dim
        self.weight = nn.Parameter(
            embedding.weight.detach()[self.start : self.stop].clone(),
            requires_grad=embedding.weight.requires_grad,
        )
        self._group = group

    def forward(self, ids):
        outside = (ids < self.start) | (ids >= self.stop)
        out = F.embedding((ids - self.start).masked_fill(outside, 0), self.weight)
        out = out.masked_fill(outside.unsqueeze(-1), 0)
        return SumOverRanks.apply(out, self._group())


class VocabShardedHead(nn.Module):
    def __init__(self, head: nn.Linear, rank: int, world: int, group, gather: bool = True):
        super().__init__()
        rows = head.out_features // world
        self.start, self.gather = rank * rows, gather
        self.in_features, self.out_features = head.in_features, head.out_features
        self.weight = nn.Parameter(
            head.weight.detach()[rank * rows : (rank + 1) * rows].clone(),
            requires_grad=head.weight.requires_grad,
        )
        self._group = group

    def forward(self, x):
        group = self._group()
        x = SumGradOverRanks.apply(x, group)
        logits = F.linear(x, self.weight)
        if self.gather:
            return _GatherLastDim.apply(logits, group)
        logits.vocab_shard = VocabShard(self.start, group)
        return logits


def shard_vocab(
    module: nn.Module, vocab_size: int, rank: int, world: int, group, gather_logits: bool = True
) -> int:
    """Replace, in place, every bias-free nn.Embedding / nn.Linear whose
    vocabulary dimension is vocab_size by its rank's shard. group: a
    callable returning the tensor-parallel process group (resolved at call
    time, so it can be created after this). gather_logits=False: heads
    return their shard's logits, tagged with .vocab_shard, for a loss that
    takes vocabulary shards. A head tied to an embedding (one weight) stays
    tied: both get the same shard parameter. Returns how many were replaced."""
    if vocab_size % world:
        return 0
    replaced = 0
    shards = {}  # id(original weight) -> its shard, so tied modules share it

    def shared(new, original):
        new.weight = shards.setdefault(id(original), new.weight)
        return new

    for parent in list(module.modules()):
        for name, child in list(parent.named_children()):
            if (
                isinstance(child, nn.Embedding)
                and child.num_embeddings == vocab_size
                and child.padding_idx is None
                and not child.weight.is_meta
            ):
                setattr(
                    parent,
                    name,
                    shared(VocabShardedEmbedding(child, rank, world, group), child.weight),
                )
                replaced += 1
            elif (
                isinstance(child, nn.Linear)
                and child.out_features == vocab_size
                and child.bias is None
                and not child.weight.is_meta
            ):
                setattr(
                    parent,
                    name,
                    shared(
                        VocabShardedHead(child, rank, world, group, gather=gather_logits),
                        child.weight,
                    ),
                )
                replaced += 1
    return replaced
