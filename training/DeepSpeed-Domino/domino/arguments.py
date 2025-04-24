# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
# This file is adapted from arguments.py in Megatron-LM

"""Domino arguments."""

import argparse
import os
import types
import math
import torch
import torch.nn.functional as F

import dataclasses
from dataclasses import dataclass
from typing import Callable
from domino.timer import Timers
from megatron.tokenizer import build_tokenizer


_GLOBAL_ARGS = None
_GLOBAL_TOKENIZER = None
_GLOBAL_TIMERS = None


def get_args():
    """Return arguments."""
    return _GLOBAL_ARGS


def set_args(args):
    global _GLOBAL_ARGS
    _GLOBAL_ARGS = args


def build_tokenizer_g(args):
    """Initialize tokenizer."""
    global _GLOBAL_TOKENIZER
    _GLOBAL_TOKENIZER = build_tokenizer(args)
    return _GLOBAL_TOKENIZER


def get_tokenizer():
    """Return tokenizer."""
    return _GLOBAL_TOKENIZER


def get_num_microbatches():
    return 1


def init_method_normal(std_dev):
    def initialize(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=std_dev)
    return initialize


def scaled_init_method_normal(std_dev, layer_count):
    scaled_std_dev = std_dev / math.sqrt(2.0 * layer_count)
    def initialize(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=scaled_std_dev)
    return initialize


def get_timers():
    """Return timers."""
    return _GLOBAL_TIMERS


def set_timers():
    """Initialize timers."""
    global _GLOBAL_TIMERS
    _GLOBAL_TIMERS = Timers(0, "maxmin")


def parse_args():
    """Parse all arguments."""
    parser = argparse.ArgumentParser(description='Domino Arguments', allow_abbrev=False)
    parser.add_argument('--num-layers', type=int, default=None,
                       help='Number of transformer layers.')
    parser.add_argument('--hidden-size', type=int, default=None,
                       help='Tansformer hidden size.')
    parser.add_argument('--num-attention-heads', type=int, default=None,
                       help='Number of transformer attention heads.')
    parser.add_argument('--ffn-hidden-size', type=int, default=None,
                       help='Transformer Feed-Forward Network hidden size. '
                       'This is set to 4*hidden-size if not provided')
    parser.add_argument('--seq-length', type=int, default=None,
                       help='Maximum sequence length to process.')
    parser.add_argument('--max-position-embeddings', type=int, default=None,
                       help='Maximum number of position embeddings to use. '
                       'This is the size of position embedding.')
    parser.add_argument('--position-embedding-type', type=str, default='learned_absolute',
                       choices=['learned_absolute', 'rope'],
                       help='Position embedding type.')
    parser.add_argument('--rotary-percent', type=float, default=1.0,
                       help='Percent of rotary dimension to use, default 100%')
    parser.add_argument('--rotary-seq-len-interpolation-factor', type=int, default=None,
                       help='Sequence length interpolation factor for rotary embeddings.')
    parser.add_argument('--hidden-dropout', type=float, default=0.1,
                       help='Dropout probability for hidden state transformer.')
    parser.add_argument('--attention-dropout', type=float, default=0.1,
                       help='Post attention dropout probability.')
    parser.add_argument('--no-masked-softmax-fusion',
                       action='store_false',
                       help='Disable fusion of query_key_value scaling, '
                       'masking, and softmax.',
                       dest='masked_softmax_fusion')
    parser.add_argument('--tensor-model-parallel-size', type=int, default=1,
                    help='Degree of tensor model parallelism.')
    parser.add_argument('--local_rank', type=int, default=None,
                       help='local rank passed from distributed launcher.')
    parser.add_argument('--distributed-backend', default='nccl',
                       choices=['nccl', 'gloo'],
                       help='Which backend to use for distributed training.')
    parser.add_argument('--seed', type=int, default=1234,
                       help='Random seed used for python, numpy, pytorch, and cuda.')
    parser.add_argument('--train-iters', type=int, default=None,
                       help='Total number of iterations to train over all '
                       'training runs. Note that either train-iters or '
                       'train-samples should be provided.')
    parser.add_argument('--micro-batch-size', type=int, default=None,
                       help='Batch size per model instance (local batch size). '
                       'Global batch size is local batch size times data '
                       'parallel size times number of micro batches.')
    parser.add_argument('--global-batch-size', type=int, default=None,
                   help='Training batch size. If set, it should be a '
                   'multiple of micro-batch-size times data-parallel-size. '
                   'If this value is None, then '
                   'use micro-batch-size * data-parallel-size as the '
                   'global batch size. This choice will result in 1 for '
                   'number of micro-batches.')
    parser.add_argument('--lr', type=float, default=None,
                       help='Initial learning rate. Depending on decay style '
                       'and initial warmup, the learing rate at each '
                       'iteration would be different.')
    parser.add_argument('--min-lr', type=float, default=0.0,
                       help='Minumum value for learning rate. The scheduler'
                       'clip values below this threshold.')
    parser.add_argument('--lr-warmup-fraction', type=float, default=None,
                       help='fraction of lr-warmup-(iters/samples) to use '
                       'for warmup (as a float)')
    parser.add_argument('--lr-decay-style', type=str, default='linear',
                       choices=['constant', 'linear', 'cosine', 'inverse-square-root'],
                       help='Learning rate decay function.')
    parser.add_argument('--lr-decay-iters', type=int, default=None,
                       help='number of iterations to decay learning rate over,'
                       ' If None defaults to `--train-iters`')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                       help='Weight decay coefficient for L2 regularization.')
    parser.add_argument('--clip-grad', type=float, default=1.0,
                       help='Gradient clipping based on global L2 norm.')
    parser.add_argument('--data-path', nargs='*', default=None,
                       help='Path to the training dataset. Accepted format:'
                       '1) a single data path, 2) multiple datasets in the'
                       'form: dataset1-weight dataset1-path dataset2-weight '
                       'dataset2-path ... It is used with --split when a '
                       'single dataset used for all three: train, valid '
                       'and test. It is exclusive to the other '
                       '--*-data-path args')
    parser.add_argument('--split', type=str, default='969, 30, 1',
                       help='Comma-separated list of proportions for training,'
                       ' validation, and test split. For example the split '
                       '`90,5,5` will use 90%% of data for training, 5%% for '
                       'validation and 5%% for test.')
    parser.add_argument('--vocab-file', type=str, default=None,
                       help='Path to the vocab file.')
    parser.add_argument('--merge-file', type=str, default=None,
                    help='Path to the BPE merge file.')
    parser.add_argument('--data-impl', type=str, default='infer',
                       choices=['mmap', 'infer'],
                       help='Implementation of indexed datasets.')
    parser.add_argument('--fp16', action='store_true',
                       help='Run model in fp16 mode.')
    parser.add_argument('--bf16', action='store_true',
                       help='Run model in bfloat16 mode.')
    parser.add_argument('--tokenizer-type', type=str,
                       default='GPT2BPETokenizer',
                       choices=['BertWordPieceLowerCase',
                                'BertWordPieceCase',
                                'GPT2BPETokenizer',
                                'SentencePieceTokenizer',
                                'GPTSentencePieceTokenizer',
                                'NullTokenizer'],
                       help='What type of tokenizer to use.')
    parser.add_argument('--make-vocab-size-divisible-by', type=int, default=128,
                       help='Pad the vocab size to be divisible by this value.'
                       'This is added for computational efficieny reasons.')
    parser.add_argument('--llama-model', action='store_true', help='Use LLaMA model.')
    parser.add_argument('--swiglu', action='store_true',
                       help='Use gated linear units and SiLU activation instead of default gelu')
    parser.add_argument('--add-bias-linear', action='store_true',
                       help='Enable bias in the linear layers')
    parser.add_argument('--normalization', default='LayerNorm',
                       choices=['LayerNorm', 'RMSNorm'],
                       help='Which normalization technique to use.',
                       dest='normalization')
    parser.add_argument('--layernorm-epsilon', type=float, default=1e-5,
                       help='Layer norm epsilon.')
    parser.add_argument('--eval-iters', type=int, default=100,
                    help='Number of iterations to run for evaluation'
                    'validation/test for.')
    parser.add_argument('--eval-interval', type=int, default=1000,
                       help='Interval between running evaluation on '
                       'validation set.')
    parser.add_argument('--log-interval', type=int, default=100,
                       help='Report loss and timing interval.')
    parser.add_argument('--save-interval', type=int, default=None,
                       help='Number of iterations between checkpoint saves.')
    parser.add_argument('--input-split-dim', type=str, default='batch',
                       help='Dimension for input split.')
    
    args = parser.parse_args()

    args.rank = int(os.getenv('RANK', '0'))
    args.world_size = int(os.getenv("WORLD_SIZE", '1'))

    if args.ffn_hidden_size is None:
        args.ffn_hidden_size = 4 * args.hidden_size
    if args.swiglu:
        args.ffn_hidden_size = int((4 * args.hidden_size * 2 / 3) / 64) * 64

    args.kv_channels = args.hidden_size // args.num_attention_heads

    args.perform_initialization = True
    args.apply_residual_connection_post_layernorm = False
    args.no_persist_layer_norm = False

    args.activation_func = F.gelu
    args.add_bias_linear = True
    args.gated_linear_unit = False
    if args.swiglu:
        args.activation_func = F.silu
        args.gated_linear_unit = True
        args.bias_gelu_fusion = False

    init_method_std = 0.02
    args.init_method = init_method_normal(init_method_std)
    args.output_layer_init_method = scaled_init_method_normal(
        init_method_std, args.num_layers)

    args.optimizer = 'adam'
    args.adam_beta1 = 0.9
    args.adam_beta2 = 0.999
    args.adam_eps = 1e-8
    args.weight_decay = 0.01
    args.lr_warmup_init = 0.0
    args.lr_decay_style = 'cosine'
    args.start_weight_decay = 0.1
    args.end_weight_decay = 0.1
    args.weight_decay_incr_style ='constant'
    args.start_weight_decay = args.weight_decay
    args.end_weight_decay = args.weight_decay
    args.use_checkpoint_opt_param_scheduler = False
    args.override_opt_param_scheduler = False

    args.mmap_warmup = False

    args.num_workers = 1
    args.dataloader_type = 'single'
    args.train_data_path = None
    args.valid_data_path = None
    args.test_data_path = None
    args.data_cache_path = None
    args.train_samples = None
    args.consumed_train_samples = 0
    args.consumed_valid_samples = 0
    args.decoder_seq_length = None
    args.reset_position_ids = False
    args.reset_attention_mask = False
    args.eod_mask_loss = False
    args.empty_unused_memory_level = 1
    args.tokenizer_type = 'GPT2BPETokenizer'

    args.loss_scale = 1024
    args.initial_loss_scale = 2**32
    args.min_loss_scale = 1.0
    args.loss_scale_window = 1000
    args.hysteresis = 2
    args.use_distributed_optimizer = False
    args.log_num_zeros_in_grad = False

    args.rampup_batch_size = None
    # Parameters dtype.
    args.accumulate_allreduce_grads_in_fp32 = False
    args.params_dtype = torch.float
    if args.fp16:
        args.params_dtype = torch.half
    if args.bf16:
        args.params_dtype = torch.bfloat16
        # bfloat16 requires gradient accumulation and all-reduce to
        # be done in fp32.
        if not args.accumulate_allreduce_grads_in_fp32:
            args.accumulate_allreduce_grads_in_fp32 = True
            if args.rank == 0:
                print('accumulate and all-reduce gradients in fp32 for '
                      'bfloat16 data type.', flush=True)

    args.async_tensor_model_parallel_allreduce = True
    if torch.cuda.is_available() and torch.version.hip:
        args.gradient_accumulation_fusion = False
    elif torch.cuda.is_available() and torch.version.cuda:
        args.gradient_accumulation_fusion = True
    
    args.padded_vocab_size = 0 # tokenizer.py
    args.model_type = 1
    args.data_parallel_size = 1
    args.DDP_impl = 'local'
    args.use_contiguous_buffers_in_local_ddp = True
    args.data_parallel_random_init = False

    return args


@dataclass
class TransformerConfig():
    """Configuration object for transformers.
    """
    sequence_parallel: bool = False
    llama_model: bool = False
    apply_residual_connection_post_layernorm = False
    no_persist_layer_norm = False

    # Initialization
    perform_initialization: bool = True
    use_cpu_initialization: bool = False

    # Training
    fp16: bool = False
    bf16: bool = False
    params_dtype: torch.dtype = torch.float32
    timers: Callable = None

    # Optimizations
    gradient_accumulation_fusion: bool = True
    async_tensor_model_parallel_allreduce: bool = True

    # model architecture
    num_layers: int = 0
    hidden_size: int = 0
    num_attention_heads: int = 0
    ffn_hidden_size: int = None
    kv_channels: int = None
    hidden_dropout: float = 0.1
    attention_dropout: float = 0.1
    layernorm_epsilon: float = 1e-5
    layernorm_zero_centered_gamma: bool = False
    add_bias_linear: bool = True
    swiglu = False
    gated_linear_unit: bool = False
    activation_func: Callable = F.gelu
    bias_gelu_fusion = False

    # initialization
    init_method: Callable = None
    output_layer_init_method: Callable = None
    init_method_std: float = 0.02

    enable_autocast: bool = False
    # autocast_dtype: torch.dtype = None
    deallocate_pipeline_outputs: bool = False
    no_sync_func: Callable = None
    # grad_sync_func: Callable = None
    # param_sync_func: Callable = None
    
    input_split_dim: str = 'batch'

    def __post_init__(self):
        """ Python dataclass method that is used to modify attributes after initialization.
            See https://docs.python.org/3/library/dataclasses.html#post-init-processing for more details.
        """
        if self.fp16 and self.bf16:
            raise ValueError(
                f'Only one of self.fp16: {self.fp16} and self.bf16 {self.bf16} should be True.'
            )

        # if self.num_attention_heads % self.tensor_model_parallel_size != 0:
        #     raise ValueError(
        #         f"num_attention_heads ({self.num_attention_heads}) must be a multiple of "
        #         f"tensor_model_parallel_size ({self.tensor_model_parallel_size})."
        #     )

        if self.ffn_hidden_size is None:
            self.ffn_hidden_size = 4 * self.hidden_size

        if self.kv_channels is None:
            self.kv_channels = self.hidden_size // self.num_attention_heads

        if self.init_method is None:
            self.init_method = init_method_normal(self.init_method_std)

        if self.output_layer_init_method is None:
            self.output_layer_init_method = scaled_init_method_normal(
                self.init_method_std, self.num_layers
            )

def core_transformer_config_from_args(args):
    # Translate args to core transformer configuration
    kw_args = {}
    for f in dataclasses.fields(TransformerConfig):
        if hasattr(args, f.name):
            kw_args[f.name] = getattr(args, f.name)
 
    kw_args['hidden_size'] = args.hidden_size
    kw_args['init_method'] = args.init_method
    kw_args['output_layer_init_method'] = args.init_method
    kw_args['params_dtype'] = args.params_dtype
    kw_args['input_split_dim'] = args.input_split_dim

    return TransformerConfig(**kw_args)
