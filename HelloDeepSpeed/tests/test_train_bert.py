import tempfile

import numpy as np
import pytest
import torch
import tqdm
from transformers import AutoTokenizer

from train_bert import create_data_iterator, create_model, load_model_checkpoint, train


@pytest.fixture(scope="function")
def checkpoint_dir() -> str:
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield tmpdirname


def test_masking_stats(tol: float = 1e-3):
    """Test to check that the masking probabilities
    match what we expect them to be.
    """
    kwargs = {
        "mask_prob": 0.15,
        "random_replace_prob": 0.1,
        "unmask_replace_prob": 0.1,
        "batch_size": 8,
    }
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    dataloader = create_data_iterator(**kwargs)
    num_samples = 10000
    total_tokens = 0
    masked_tokens = 0
    random_replace_tokens = 0
    unmasked_replace_tokens = 0
    for ix, batch in tqdm.tqdm(enumerate(dataloader, start=1), total=num_samples):
        # Since we don't mask the BOS / EOS tokens, we subtract them from the total tokens
        total_tokens += batch["attention_mask"].sum().item() - (
            2 * batch["attention_mask"].size(0)
        )
        masked_tokens += (batch["tgt_tokens"] != tokenizer.pad_token_id).sum().item()
        random_or_unmasked = (
            batch["tgt_tokens"] != tokenizer.pad_token_id
        ).logical_and(batch["src_tokens"] != tokenizer.mask_token_id)
        unmasked = random_or_unmasked.logical_and(
            batch["src_tokens"] == batch["tgt_tokens"]
        )
        unmasked_replace_tokens += unmasked.sum().item()
        random_replace_tokens += random_or_unmasked.sum().item() - unmasked.sum().item()
        if ix == num_samples:
            break
    estimated_mask_prob = masked_tokens / total_tokens
    estimated_random_tokens = random_replace_tokens / total_tokens
    estimated_unmasked_tokens = unmasked_replace_tokens / total_tokens
    assert np.isclose(estimated_mask_prob, kwargs["mask_prob"], atol=tol)
    assert np.isclose(
        estimated_random_tokens,
        kwargs["random_replace_prob"] * kwargs["mask_prob"],
        atol=tol,
    )
    assert np.isclose(
        estimated_unmasked_tokens,
        kwargs["unmask_replace_prob"] * kwargs["mask_prob"],
        atol=tol,
    )


def test_model_checkpointing(checkpoint_dir: str):
    """Training a small model, and ensuring
    that both checkpointing and resuming from
    a checkpoint work.
    """
    # First train a tiny model for 5 iterations
    train_params = {
        "checkpoint_dir": checkpoint_dir,
        "checkpoint_every": 2,
        "num_layers": 2,
        "num_heads": 4,
        "ff_dim": 64,
        "h_dim": 64,
        "num_iterations": 5,
    }
    exp_dir = train(**train_params)
    # now check that we have 3 checkpoints
    assert len(list(exp_dir.glob("*.pt"))) == 3
    model = create_model(
        num_layers=train_params["num_layers"],
        num_heads=train_params["num_heads"],
        ff_dim=train_params["ff_dim"],
        h_dim=train_params["h_dim"],
        dropout=0.1,
    )
    optimizer = torch.optim.Adam(model.parameters())
    step, model, optimizer = load_model_checkpoint(exp_dir, model, optimizer)
    assert step == 5
    model_state_dict = model.state_dict()
    # the saved checkpoint would be for iteration 5
    correct_state_dict = torch.load(exp_dir / "checkpoint.iter_5.pt")
    correct_model_state_dict = correct_state_dict["model"]
    assert set(model_state_dict.keys()) == set(correct_model_state_dict.keys())
    assert all(
        torch.allclose(model_state_dict[key], correct_model_state_dict[key])
        for key in model_state_dict.keys()
    )
    # Finally, try training with the checkpoint
    train_params.pop("checkpoint_dir")
    train_params["load_checkpoint_dir"] = str(exp_dir)
    train_params["num_iterations"] = 10
    train(**train_params)
