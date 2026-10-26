# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

"""CPU regression checks for masked PPO policy log-ratios."""

import pytest
import torch

from dschat.rlhf.ppo_trainer import DeepSpeedPPOTrainer


@pytest.mark.parametrize("padding_case", ["negative_infinity", "nan"])
def test_actor_loss_ignores_nonfinite_masked_logprobs(padding_case):
    trainer = DeepSpeedPPOTrainer.__new__(DeepSpeedPPOTrainer)
    trainer.cliprange = 0.2

    old = torch.tensor([[-1.0, -1.0, -1.0]])
    logprobs = old.clone()
    if padding_case == "negative_infinity":
        old[0, 0] = float("-inf")
        logprobs[0, 0] = float("-inf")
    else:
        logprobs[0, 0] = float("nan")
    logprobs.requires_grad_()

    mask = torch.tensor([[0.0, 1.0, 1.0]])
    loss = trainer.actor_loss_fn(logprobs, old, torch.ones_like(logprobs), mask)
    loss.backward()

    torch.testing.assert_close(loss, torch.tensor(-1.0))
    torch.testing.assert_close(logprobs.grad, torch.tensor([[0.0, -0.5, -0.5]]))
    assert torch.isfinite(logprobs.grad).all()


def test_actor_loss_keeps_normal_unmasked_gradient():
    trainer = DeepSpeedPPOTrainer.__new__(DeepSpeedPPOTrainer)
    trainer.cliprange = 0.2
    old = torch.tensor([[-1.0, -1.0]])
    logprobs = old.clone().requires_grad_(True)
    loss = trainer.actor_loss_fn(
        logprobs,
        old,
        torch.ones_like(logprobs),
        torch.ones_like(logprobs),
    )
    loss.backward()
    torch.testing.assert_close(loss, torch.tensor(-1.0))
    torch.testing.assert_close(logprobs.grad, torch.tensor([[-0.5, -0.5]]))
