# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
# This file is adapted from language_model.py in Megatron-LM

from typing import Literal, Optional

import torch
from torch import einsum, nn
from domino.arguments import get_args
from domino.modules.enums import ModelType
import domino.parallel_state as mpu
from domino.modules.module import DominoModule
from domino.tensor_parallel.comm import GatherFromModelParallelRegion
from domino.tensor_parallel.partition  import VocabParallelEmbedding, linear_with_grad_accumulation_and_async_allreduce
from domino.modules.fused_layer_norm import MixedFusedLayerNorm as fused_layer_norm
from domino.modules.fused_func import bias_dropout_add_fused_train, bias_dropout_add_fused_inference, apply_rotary_pos_emb
from domino.tensor_parallel.partition import _initialize_affine_weight_gpu, set_tensor_model_parallel_attributes
from domino.tensor_parallel.partition import ColumnParallelLinear, RowParallelLinearNoComm

from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.model.utils import get_norm

from deepspeed.runtime.domino.transformer import DominoTransformer

def parallel_lm_logits(input_, word_embeddings_weight, parallel_output,
                       bias=None):
    """LM logits using word embedding weights."""
    args = get_args()
    # Parallel logits.
    if args.async_tensor_model_parallel_allreduce:
        input_parallel = input_
        model_parallel = mpu.get_tensor_model_parallel_world_size() > 1
        async_grad_allreduce = args.async_tensor_model_parallel_allreduce and model_parallel

    # Matrix multiply.
    logits_parallel = linear_with_grad_accumulation_and_async_allreduce(
        input=input_parallel,
        weight=word_embeddings_weight,
        bias=bias,
        gradient_accumulation_fusion=args.gradient_accumulation_fusion,
        async_grad_allreduce=async_grad_allreduce,
        sequence_parallel=False)
    # Gather if needed.

    if parallel_output:
        return logits_parallel

    return GatherFromModelParallelRegion.apply(logits_parallel)


def get_language_model(config, num_tokentypes,
                       encoder_attn_mask_type,
                       pre_process=True, post_process=True):
    args = get_args()
    language_model = TransformerLanguageModel(
        config,
        encoder_attn_mask_type,
        num_tokentypes=num_tokentypes,
        pre_process=pre_process,
        post_process=post_process,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
        rotary_base=args.rotary_base,
        rope_scaling=args.use_rope_scaling,
        seq_len_interpolation_factor = args.rotary_seq_len_interpolation_factor,
    )

    return language_model


class Embedding(DominoModule):
    def __init__(self, hidden_dim, vocab_size, max_seq_len, dropout_prob, config):
        super(Embedding, self).__init__()
        self.hidden_dim = hidden_dim
        self.init_method = config.init_method
        args = get_args()
        self.word_embeddings = VocabParallelEmbedding(
            vocab_size, self.hidden_dim, config=config, init_method=config.init_method
        )
        self.use_position_embedding = args.position_embedding_type == 'learned_absolute'
        if self.use_position_embedding:
            self.position_embeddings = torch.nn.Embedding(max_seq_len, self.hidden_dim)
            self.init_method(self.position_embeddings.weight)
        self.embedding_dropout = torch.nn.Dropout(dropout_prob)

    def forward(self, input_ids, position_ids):
        word_embeds = self.word_embeddings(input_ids)
        if self.use_position_embedding:
            pos_embeds = self.position_embeddings(position_ids)
            combined_embeds = word_embeds + pos_embeds
        else:
            combined_embeds = word_embeds

        combined_embeds = combined_embeds.transpose(0, 1).contiguous()
        combined_embeds = self.embedding_dropout(combined_embeds)

        return combined_embeds


class TransformerLanguageModel(DominoModule):
    def __init__(self,
                 config,
                 encoder_attn_mask_type,
                 num_tokentypes=0,
                 pre_process=True,
                 post_process=True,
                 position_embedding_type: Literal['learned_absolute', 'rope', 'none'] = 'learned_absolute',
                 rotary_percent: float = 1.0,
                 rotary_base: int = 10000,
                 rope_scaling: bool = False,
                 seq_len_interpolation_factor: Optional[float] = None,):
        args = get_args()
        super(TransformerLanguageModel, self).__init__(share_embeddings_and_output_weights=True)

        self.pre_process = pre_process
        self.post_process = post_process
        self.hidden_size = config.hidden_size
        self.num_tokentypes = num_tokentypes
        self.init_method = config.init_method
        self.encoder_attn_mask_type = encoder_attn_mask_type
        self.encoder_hidden_state = None
        self.position_embedding_type = position_embedding_type
        self.rotary_percent = rotary_percent
        self.rotary_base = rotary_base
        self.rotary_scaling = rope_scaling
        self.seq_length = config.seq_length

        if self.pre_process:
            self.embedding = Embedding(self.hidden_size,
                                       args.padded_vocab_size,
                                       args.max_position_embeddings,
                                       args.hidden_dropout,
                                       config)

        self.use_rotary_position_embeddings = \
            args.position_embedding_type == 'rope'
        if self.use_rotary_position_embeddings:
            self.rotary_pos_emb = RotaryEmbedding(
                kv_channels=config.kv_channels,
                rotary_percent=rotary_percent,
                rotary_interleaved=config.rotary_interleaved,
                seq_len_interpolation_factor=seq_len_interpolation_factor,
                rotary_base=rotary_base,
                rope_scaling=rope_scaling,
            )

        self.encoder = DominoTransformer(
            config, ModelType.encoder_or_decoder, mpu,
            get_norm, _initialize_affine_weight_gpu,
            ColumnParallelLinear, RowParallelLinearNoComm, apply_rotary_pos_emb,
            bias_dropout_add_fused_train, bias_dropout_add_fused_inference,
            self_attn_mask_type=self.encoder_attn_mask_type,
            pre_process=self.pre_process,
            post_process=self.post_process,
        )

    def set_input_tensor(self, input_tensor):
        pass
        # self.encoder.set_input_tensor(input_tensor[0])

    def forward(self, enc_input_ids, enc_position_ids, enc_attn_mask,
                inference_params=None):

        if self.pre_process:
            encoder_input = self.embedding(enc_input_ids, enc_position_ids)
        else:
            encoder_input = None

        rotary_pos_emb = None
        if self.use_rotary_position_embeddings:
            rotary_pos_emb = self.rotary_pos_emb(self.seq_length)
        rotary_pos_emb = ((rotary_pos_emb,) * 2)

        encoder_out_size = encoder_input.shape
        p_batch_size = encoder_out_size[1] // 2
        dtype = encoder_input.dtype
        encoder_output_t = torch.empty(encoder_out_size, dtype=dtype, device=torch.cuda.current_device())
        intra_partitions = 2
        encoder_inputs = torch.tensor_split(encoder_input, intra_partitions, dim=1)
        encoder_outputs = self.encoder(
            encoder_inputs,
            enc_attn_mask,
            rotary_pos_emb=rotary_pos_emb)
        encoder_output_t[:, 0:p_batch_size, :] = encoder_outputs[0]
        encoder_output_t[:, p_batch_size:2*p_batch_size, :] = encoder_outputs[1]
        encoder_output = encoder_output_t

        return encoder_output
        