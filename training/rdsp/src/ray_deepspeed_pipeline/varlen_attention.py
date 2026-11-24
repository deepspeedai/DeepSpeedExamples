# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""Vision attention as one variable-length flash attention call per layer.

Hugging Face's Qwen-VL vision encoders attend within each image. Under SDPA
they split each layer's q, k and v per image on the host (`lengths.tolist()`:
three GPU syncs in every layer) and run one attention call per image. Under a
flash implementation they hand the image boundaries (cu_seqlens) to one
kernel instead, and read the longest image's length once per forward.
PyTorch's own varlen_attn (FlashAttention, with a backward) is such a kernel,
so rdsp registers it under a name containing "flash" and points only the
vision encoder's config at it; the language model's attention is unchanged.

The registry is per process: stage workers unpickle modules built on the
driver, so they register the function again (register_if_used).
"""

import torch

from ray_deepspeed_pipeline.errors import ValidationError

NAME = "rdsp_flash_varlen"


def varlen_attention_forward(
    module,
    query,
    key,
    value,
    attention_mask=None,
    *,
    scaling=None,
    dropout=0.0,
    cu_seq_lens_q=None,
    cu_seq_lens_k=None,
    max_length_q=None,
    max_length_k=None,
    is_causal=False,
    **kwargs,
):
    """Hugging Face's attention-function signature: query, key and value are
    (1, heads, tokens, head_dim), the packed sequences cu_seq_lens bounds;
    returns ((1, tokens, heads, head_dim), None)."""
    from torch.nn.attention.varlen import varlen_attn

    if cu_seq_lens_q is None or attention_mask is not None or dropout:
        raise ValidationError(
            f"{NAME} runs packed sequences only: cu_seq_lens, no mask, no dropout"
        )
    q, k, v = (t[0].transpose(0, 1) for t in (query, key, value))
    out = varlen_attn(
        q,
        k,
        v,
        cu_seq_lens_q,
        cu_seq_lens_k,
        max_length_q,
        max_length_k,
        scale=scaling,
        window_size=(-1, 0) if is_causal else (-1, -1),
    )
    return out.unsqueeze(0), None


def register() -> None:
    from transformers import AttentionInterface

    AttentionInterface.register(NAME, varlen_attention_forward)


def _configs(module: torch.nn.Module) -> list:
    from transformers import PretrainedConfig

    found = {}
    for m in module.modules():
        config = getattr(m, "config", None)
        if isinstance(config, PretrainedConfig):
            found[id(config)] = config
    return list(found.values())


def use_varlen_vision_attention(model: torch.nn.Module) -> None:
    """Run the model's vision encoder attention through NAME: every config
    object the encoder's modules hold."""
    from ray_deepspeed_pipeline.vision import find_vision_encoder

    register()
    for config in _configs(model.get_submodule(find_vision_encoder(model))):
        config._attn_implementation = NAME


def register_if_used(module: torch.nn.Module) -> None:
    """Register NAME in this process if a config in `module` names it."""
    try:
        configs = _configs(module)
    except ImportError:  # no transformers: no Hugging Face configs either
        return
    if any(c._attn_implementation == NAME for c in configs):
        register()
