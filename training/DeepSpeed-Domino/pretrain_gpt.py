# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
# This file is adapted from pretrain_gpt.py in Megatron-LM

import time
import torch
from domino.utils import print_rank_0
from domino.initialize import initialize_domino, set_jit_fusion_options
from domino.arguments import get_args, core_transformer_config_from_args
from domino.data.gpt_dataset import build_train_valid_test_datasets
from domino.training import pretrain
from domino.modules.module import DominoModule
from domino.modules.enums import AttnMaskType
from domino.language_model import parallel_lm_logits
from domino.language_model import get_language_model
from domino.tensor_parallel.cross_entropy import vocab_parallel_cross_entropy

from domino.tensor_parallel.cross_entropy import LigerFusedLinearCrossEntropyFunction

_TRAIN_START_TIME = time.time()

def post_language_model_processing(lm_output, labels, logit_weights, parallel_output):
    output = parallel_lm_logits(lm_output, logit_weights, parallel_output)
    labels = labels.transpose(0, 1).contiguous()
    loss = vocab_parallel_cross_entropy(output.float(), labels)
    loss = loss.transpose(0, 1).contiguous()
    return loss

def post_language_model_processing_with_liger(lm_output, labels, logit_weights, parallel_output):
    b, s = labels.shape
    lm_output = lm_output.flatten(0, 1)
    labels = labels.transpose(0, 1).flatten(0, 1)
    loss, _ = LigerFusedLinearCrossEntropyFunction.apply(lm_output, logit_weights, labels)
    loss = loss.view(s, b).transpose(0, 1).contiguous()
    return loss


class GPTModel(DominoModule):
    def __init__(
        self,
        config,
        num_tokentypes=0,
        parallel_output=True,
        pre_process=True,
        post_process=True,
    ):
        super().__init__(config=config)
        self.parallel_output = parallel_output
        self.pre_process = pre_process
        self.post_process = post_process
        self.language_model = get_language_model(
            config=config,
            num_tokentypes=num_tokentypes,
            encoder_attn_mask_type=AttnMaskType.causal,
            pre_process=self.pre_process,
            post_process=self.post_process,
        )
        self.initialize_word_embeddings()
        self.config = config

    def set_input_tensor(self, input_tensor):
        self.language_model.set_input_tensor(input_tensor)

    def forward(
        self,
        input_ids,
        position_ids,
        attention_mask,
        labels=None,
        inference_params=None,
    ):
        lm_output = self.language_model(
            input_ids,
            position_ids,
            attention_mask,
            inference_params=inference_params,
        )

        if self.post_process:
            if self.config.fused_linear_loss:
                return post_language_model_processing_with_liger(
                    lm_output,
                    labels,
                    self.shared_embedding_or_output_weight(),
                    self.parallel_output,
                )
            else:
                return post_language_model_processing(
                    lm_output,
                    labels,
                    self.shared_embedding_or_output_weight(),
                    self.parallel_output,
                )
        else:
            return lm_output

def main():
    initialize_domino()

    # Set pytorch JIT layer fusion options and warmup JIT functions.
    set_jit_fusion_options()

    # Adjust the startup time so it reflects the largest value.
    # This will be closer to what scheduler will see (outside of
    # image ... launches.
    global _TRAIN_START_TIME
    start_time_tensor = torch.cuda.DoubleTensor([_TRAIN_START_TIME])
    torch.distributed.all_reduce(start_time_tensor,
                                 op=torch.distributed.ReduceOp.MIN)
    _TRAIN_START_TIME = start_time_tensor.item()
    print_rank_0('time to initialize megatron (seconds): {:.3f}'.format(
        time.time() - _TRAIN_START_TIME))

    print_rank_0('Building GPT model ...')
    config = core_transformer_config_from_args(get_args())
    model = GPTModel(
        config,
        num_tokentypes=0,
        parallel_output=True,
        pre_process=True,
        post_process=True
    )

    args = get_args()
    print_rank_0('Load GPT dataset ...')
    # Number of train/valid/test samples.
    if args.train_samples:
        train_samples = args.train_samples
    else:
        train_samples = args.train_iters * args.global_batch_size
    eval_iters = (args.train_iters // args.eval_interval + 1) * \
        args.eval_iters
    test_iters = args.eval_iters
    train_val_test_num_samples = [train_samples,
                                  eval_iters * args.global_batch_size,
                                  test_iters * args.global_batch_size]
    print_rank_0(' > datasets target sizes (minimum size):')
    print_rank_0('    train:      {}'.format(train_val_test_num_samples[0]))
    print_rank_0('    validation: {}'.format(train_val_test_num_samples[1]))
    print_rank_0('    test:       {}'.format(train_val_test_num_samples[2]))
    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        data_prefix=args.data_path,
        data_impl=args.data_impl,
        splits_string=args.split,
        train_valid_test_num_samples=train_val_test_num_samples,
        seq_length=args.seq_length,
        seed=args.seed,
        skip_warmup=(not args.mmap_warmup),
        train_data_prefix=args.train_data_path,
        valid_data_prefix=args.valid_data_path,
        test_data_prefix=args.test_data_path,
        data_cache_path=args.data_cache_path)
    
    pretrain(model, train_ds, valid_ds, test_ds)


if __name__ == "__main__":
    main()
