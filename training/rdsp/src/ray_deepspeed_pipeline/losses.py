# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""Loss averaging across the step.

A plain loss_fn returns each data-parallel rank's mean over its rows, and the
step loss is the mean of those means over ranks and microbatches. When rows
hold different numbers of counted tokens (captions of different lengths),
that weights a token by how few tokens its rank had. TokenMeanLoss instead
averages over every counted token of the step, as Megatron's per-token loss
does.
"""

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class TokenMeanLoss:
    """loss_fn whose step loss is (summed token loss over all microbatches
    and ranks) / (their token count). sum_fn(outputs, labels): the summed
    per-token loss of one rank's rows; count_fn(labels): how many tokens it
    sums. The driver counts the step's tokens from the labels before the
    step, so every backward already carries the final weight.

    per_microbatch: train instead on each rank's microbatch's own token mean,
    averaged over microbatches and ranks (Megatron's default, without
    calculate_per_token_loss), while still reporting the step's token mean,
    as Megatron logs it."""

    sum_fn: Callable
    count_fn: Callable
    per_microbatch: bool = False

    def __call__(self, outputs, labels):
        return self.sum_fn(outputs, labels)

    def split(self, outputs, labels, weight: float):
        """(loss to train on, value to report) for one rank's microbatch;
        weight: token_weight() for the step."""
        total = self.sum_fn(outputs, labels)
        reported = total.detach() * weight
        if self.per_microbatch:
            return total / max(int(self.count_fn(labels)), 1), reported
        return total * weight, reported

    @property
    def takes_vocab_shards(self) -> bool:
        return bool(getattr(self.sum_fn, "takes_vocab_shards", False))


def token_weight(token_total: int, dp: int, n_microbatches: int) -> float:
    """Factor on one rank's summed loss so that the adapter's usual handling
    (divide by the microbatch count, then DeepSpeed's average over dp) leaves
    sum / token_total."""
    return dp * n_microbatches / max(token_total, 1)
