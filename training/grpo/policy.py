# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""The policy as DeepSpeed trains it: a causal LM whose forward returns token log-probs.

Full logits of a long rollout are too large to keep (16k tokens x 152k
vocab is 9.7 GB in fp32), so the LM head runs over the sequence in chunks
under activation checkpointing and only each target token's log-prob is
kept. Training goes through ``engine(...)`` rather than a
submodule, because DeepSpeed hooks the engine's forward outputs to prepare
ZeRO for the backward pass.

Rollout instead drives the causal LM itself: ``generate``, CUDA graph capture
and the forward profiler all call ``engine.module`` as a Hugging Face model, so
the rollout gets a :class:`RolloutEngineView` whose ``module`` is that model.
"""

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import AttentionInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.masking_utils import AttentionMaskInterface, sdpa_mask

# SDPA that reads grouped KV heads in place on decode steps; see _sdpa_gqa_decode.
SDPA_GQA_DECODE = "sdpa_gqa_decode"


def _sdpa_gqa_decode(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kwargs):
    """Hugging Face SDPA attention, except that one-token decode steps read grouped KV heads in place.

    With an attention mask, Hugging Face's SDPA copies each KV head once per query head it serves before
    attending. Rollout decodes with a mask over each row's whole KV cache, so for Qwen3 (8 KV heads for 32
    query heads) that copy reads and writes the full cache four times over at every step, and dominates
    decode time. ``enable_gqa`` attends to the shared heads directly. Prefill and training, where the query
    is longer than one token, keep Hugging Face's SDPA path.
    """
    if query.shape[2] != 1 or getattr(module, "num_key_value_groups", 1) == 1:
        return sdpa_attention_forward(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling,
                                      **kwargs)
    output = F.scaled_dot_product_attention(query,
                                            key,
                                            value,
                                            attn_mask=attention_mask,
                                            dropout_p=dropout,
                                            scale=scaling,
                                            enable_gqa=True)
    return output.transpose(1, 2).contiguous(), None


def register_attention() -> None:
    """Make ``SDPA_GQA_DECODE`` available as an ``attn_implementation``, with SDPA's attention masks."""
    AttentionInterface.register(SDPA_GQA_DECODE, _sdpa_gqa_decode)
    AttentionMaskInterface.register(SDPA_GQA_DECODE, sdpa_mask)


def _chunk_log_probs(hidden, lm_head, targets, temperature):
    logits = lm_head(hidden).float() / temperature
    return F.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)


class PolicyModel(torch.nn.Module):
    """Wraps a Hugging Face causal LM; its forward returns token log-probs for training."""

    def __init__(self, model, chunk_size: int = 1024):
        super().__init__()
        self.model = model
        self.chunk_size = chunk_size

    def forward(self, input_ids, attention_mask, temperature: float = 1.0):
        """Log-prob of each next token under ``softmax(logits / temperature)``.

        Rollout samples from ``softmax(logits / temperature)``, so the
        policy being trained is that tempered distribution.

        Returns:
            ``[B, T - 1]`` log-probs; position ``t`` scores ``input_ids[:, t + 1]``.
        """
        hidden = self.model.get_decoder()(input_ids=input_ids, attention_mask=attention_mask,
                                          use_cache=False).last_hidden_state
        hidden = hidden[:, :-1]
        targets = input_ids[:, 1:]
        lm_head = self.model.get_output_embeddings()
        log_probs = []
        for start in range(0, hidden.shape[1], self.chunk_size):
            end = start + self.chunk_size
            log_probs.append(
                checkpoint(_chunk_log_probs,
                           hidden[:, start:end],
                           lm_head,
                           targets[:, start:end],
                           temperature,
                           use_reentrant=False))
        return torch.cat(log_probs, dim=1)


class RolloutEngineView:
    """The DeepSpeed engine as the rollout sees it: ``module`` is the wrapped causal LM.

    Everything else (the hybrid engine's ZeRO and shared-prefill methods) is the
    engine's own, so rollout and training still share one set of weights.
    """

    def __init__(self, engine):
        self._engine = engine
        self.module = engine.module.model

    def __getattr__(self, name):
        return getattr(self._engine, name)
