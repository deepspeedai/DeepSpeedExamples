# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
# This file is adapted from training.py in Megatron-LM

import torch
from domino.arguments import get_args, get_tokenizer, get_num_microbatches, get_timers
from domino.utils import print_rank_0, get_model_config, get_ltor_masks_and_position_ids
import domino.parallel_state as mpu
from domino.tensor_parallel.partition import set_defaults_if_not_set_tensor_model_parallel_attributes
from domino.modules.enums import ModelType
from domino.schedules import get_forward_backward_func
from domino.data.data_samplers import build_pretraining_data_loader
from domino.modules.distributed import DistributedDataParallel as LocalDDP
from domino.modules.module import Float16Module
from domino.optimizer import get_megatron_optimizer
from domino.optimizer_param_scheduler import OptimizerParamScheduler
from domino.initialize import set_jit_fusion_options
from domino.tensor_parallel.data import broadcast_data


from megatron.checkpointing import load_checkpoint
from megatron.checkpointing import save_checkpoint

def is_rank_0():
    # if torch.cuda.current_device() == 0:
    if torch.distributed.get_rank() == 0:
        return True
    

def forward_step(data_iterator, model):
    input_tokens, target_labels, loss_mask, attention_mask, position_ids = prepare_batch(data_iterator)
    model_output = model(input_tokens, position_ids, attention_mask, labels=target_labels)
    return model_output, lambda output: compute_loss(loss_mask, output)


def prepare_batch(data_iterator):
    args = get_args()
    tokenizer = get_tokenizer()

    data_keys = ['text']
    data_type = torch.int64

    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None

    broadcasted_data = broadcast_data(data_keys, data, data_type)
    full_tokens = broadcasted_data['text'].long()
    input_tokens = full_tokens[:, :-1].contiguous()
    target_labels = full_tokens[:, 1:].contiguous()

    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        input_tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss
    )

    return input_tokens, target_labels, loss_mask, attention_mask, position_ids


def compute_loss(loss_mask, model_output):
    flattened_output = model_output.view(-1).float()
    flattened_loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(flattened_output * flattened_loss_mask) / flattened_loss_mask.sum()
    return loss


def pretrain(base_model, train_ds, valid_ds, test_ds):
    args = get_args()

    # Model, optimizer, and learning rate.
    model, optimizer, opt_param_scheduler = setup_model_and_optimizer(
        base_model, ModelType.encoder_or_decoder)
    config = get_model_config(model)

    # Do not use virtual pipeline parallelism for data parallel
    train_data_iterator, valid_data_iterator, test_data_iterator \
        = get_dataset_iterator(train_ds, valid_ds, test_ds)

    # Train and eval.
    print_rank_0('training ...')

    if args.do_train and args.train_iters > 0:
        train(forward_step,
              model, optimizer, opt_param_scheduler,
              train_data_iterator, valid_data_iterator, config)

    # if args.do_valid:
    #     total_loss_dict = evaluate(forward_step, valid_data_iterator, model, config, True)
    #     print_rank_0(total_loss_dict)

    # if args.do_test:
    #     total_loss_dict = evaluate(forward_step, test_data_iterator, model, config, True)
    #     print_rank_0(total_loss_dict)


def setup_model_and_optimizer(base_model,
                              model_type,
                              no_wd_decay_cond=None,
                              scale_lr_cond=None):
    """Setup model and optimizer."""
    args = get_args()

    model = get_model(base_model, model_type)

    if isinstance(model, list):
        models = model
    else:
        models = [model]
    optimizer = get_megatron_optimizer(models, no_wd_decay_cond, scale_lr_cond)
    opt_param_scheduler = get_optimizer_param_scheduler(optimizer)

    if args.load is not None:
        args.iteration = load_checkpoint(model, optimizer, opt_param_scheduler)
    else:
        args.iteration = 0

    return model, optimizer, opt_param_scheduler


def get_model(base_model, model_type=ModelType.encoder_or_decoder, wrap_with_ddp=True):
    """Build the model."""
    args = get_args()
    args.model_type = model_type

    # Build model.
    model = base_model 
    model.model_type = model_type

    for param in model.parameters():
        set_defaults_if_not_set_tensor_model_parallel_attributes(param)

    # Print number of parameters.
    if mpu.get_data_parallel_rank() == 0:
        print(' > number of parameters on (tensor, pipeline) '
              'model parallel rank ({}, {}): {}'.format(
                  mpu.get_tensor_model_parallel_rank(),
                  mpu.get_pipeline_model_parallel_rank(),
                  sum([p.nelement() for p in model.parameters()])), flush=True)

    # GPU allocation.
    model.cuda(torch.cuda.current_device())

    # Fp16 conversion.
    if args.fp16 or args.bf16:
        model = Float16Module(model, args)

    if wrap_with_ddp:
        if args.DDP_impl == 'local':
            model = LocalDDP(model,
                             args.accumulate_allreduce_grads_in_fp32,
                             args.use_contiguous_buffers_in_local_ddp)
            # broad cast params from data parallel src rank to other data parallel ranks
            if args.data_parallel_random_init:
                model.broadcast_params()
        else:
            raise NotImplementedError('Unknown DDP implementation specified: '
                                      '{}. Exiting.'.format(args.DDP_impl))
    return model


def get_optimizer_param_scheduler(optimizer):
    """Build the learning rate scheduler."""
    args = get_args()

    # Iteration-based training.
    # Remove sample-based training.
    if args.train_iters:
        if args.lr_decay_iters is None:
            args.lr_decay_iters = args.train_iters
        lr_decay_steps = args.lr_decay_iters * args.global_batch_size
        wd_incr_steps = args.train_iters * args.global_batch_size
        if args.lr_warmup_fraction is not None:
            lr_warmup_steps = args.lr_warmup_fraction * lr_decay_steps
        else:
            lr_warmup_steps = args.lr_warmup_iters * args.global_batch_size
    else:
        raise Exception(
            'either train-iters or train-samples should be provided.')

    opt_param_scheduler = OptimizerParamScheduler(
        optimizer,
        init_lr=args.lr_warmup_init,
        max_lr=args.lr,
        min_lr=args.min_lr,
        lr_warmup_steps=lr_warmup_steps,
        lr_decay_steps=lr_decay_steps,
        lr_decay_style=args.lr_decay_style,
        start_wd=args.start_weight_decay,
        end_wd=args.end_weight_decay,
        wd_incr_steps=wd_incr_steps,
        wd_incr_style=args.weight_decay_incr_style,
        use_checkpoint_opt_param_scheduler=args.use_checkpoint_opt_param_scheduler,
        override_opt_param_scheduler=args.override_opt_param_scheduler)

    return opt_param_scheduler


def get_dataset_iterator(train_ds, valid_ds, test_ds):
    """Build pretraining data iterators."""
    args = get_args()

    # Build loaders.
    train_dataloader, valid_dataloader, test_dataloader = \
        get_data_loader(train_ds, valid_ds, test_ds)

    # Build iterators.
    dl_type = args.dataloader_type
    assert dl_type == 'single'

    if train_dataloader is not None:
        train_data_iterator = iter(train_dataloader)
    else:
        train_data_iterator = None

    if valid_dataloader is not None:
        valid_data_iterator = iter(valid_dataloader)
    else:
        valid_data_iterator = None

    if test_dataloader is not None:
        test_data_iterator = iter(test_dataloader)
    else:
        test_data_iterator = None

    return train_data_iterator, valid_data_iterator, test_data_iterator


def get_data_loader(train_ds, valid_ds, test_ds):
    """Build pretraining data loaders."""
    args = get_args()

    (train_dataloader, valid_dataloader, test_dataloader) = (None, None, None)

    print_rank_0('> building train, validation, and test datasets ...')

    # Backward compatibility, assume fixed batch size.
    if args.iteration > 0 and args.consumed_train_samples == 0:
        assert args.train_samples is None, \
            'only backward compatiblity support for iteration-based training'
        args.consumed_train_samples = args.iteration * args.global_batch_size
    if args.iteration > 0 and args.consumed_valid_samples == 0:
        if args.train_samples is None:
            args.consumed_valid_samples = (args.iteration // args.eval_interval) * \
                args.eval_iters * args.global_batch_size

    # Data loader only on rank 0 of each model parallel group.
    if mpu.get_tensor_model_parallel_rank() == 0:
        # Build dataloders.
        train_dataloader = build_pretraining_data_loader(
            train_ds, args.consumed_train_samples)
        valid_dataloader = build_pretraining_data_loader(
            valid_ds, args.consumed_valid_samples)
        test_dataloader = build_pretraining_data_loader(test_ds, 0)

        # Flags to know if we need to do training/validation/testing.
        do_train = train_dataloader is not None and args.train_iters > 0
        do_valid = valid_dataloader is not None and args.eval_iters > 0
        do_test = test_dataloader is not None and args.eval_iters > 0
        # Need to broadcast num_tokens and num_type_tokens.
        flags = torch.cuda.LongTensor(
            [int(do_train), int(do_valid), int(do_test)])
    else:
        flags = torch.cuda.LongTensor([0, 0, 0])

    # Broadcast num tokens.
    torch.distributed.broadcast(flags,
                                mpu.get_tensor_model_parallel_src_rank(),
                                group=mpu.get_tensor_model_parallel_group())
    args.do_train = flags[0].item()
    args.do_valid = flags[1].item()
    args.do_test = flags[2].item()

    return train_dataloader, valid_dataloader, test_dataloader


def train(forward_step_func, model, optimizer, opt_param_scheduler,
          train_data_iterator, valid_data_iterator, config):
    """Train the model function."""
    args = get_args()
    timers = get_timers()

    model.train()

    # Iterations.
    iteration = args.iteration

    # Setup some training config params
    config.grad_scale_func = optimizer.scale_loss
    config.timers = None

    timers('ite-time', log_level=0).start(barrier=True)
    while iteration < args.train_iters:
        args.curr_iteration = iteration
        loss_dict, skipped_iter, grad_norm, num_zeros_in_grad = \
            train_step(forward_step_func,
                        train_data_iterator,
                        model,
                        optimizer,
                        opt_param_scheduler,
                        config)

        iteration += 1

        if args.save and args.save_interval and \
           iteration % args.save_interval == 0:
            save_checkpoint(iteration, model, optimizer, opt_param_scheduler)

        args.consumed_train_samples += mpu.get_data_parallel_world_size() * \
            args.micro_batch_size * get_num_microbatches()
        
        ite_time = timers('ite-time').elapsed(barrier=True)
        if iteration % args.log_interval == 0 and is_rank_0():
            loss = loss_dict['lm loss'].item()
            print( 'iteration: {} | loss: {:.3f} | iteration time (ms): {} '.format(iteration, loss, ite_time*1000.0))
            # loss_scale = optimizer.cur_scale
            # lr = optimizer.param_groups[0]['lr']
            # print( 'lr: {} loss scale: {:.1f} |'.format(lr, loss_scale))'

    return iteration


def train_step(forward_step_func, data_iterator,
               model, optimizer, opt_param_scheduler, config):
    """Single training step."""
    args = get_args()

    # Set grad to zero.
    if args.DDP_impl == 'local' and args.use_contiguous_buffers_in_local_ddp:
        model.zero_grad_buffer()
    optimizer.zero_grad()

    forward_backward_func = get_forward_backward_func()

    losses_reduced = forward_backward_func(
        forward_step_func=forward_step_func,
        data_iterator=data_iterator,
        model=model,
        num_microbatches=get_num_microbatches(),
        seq_length=args.seq_length,
        micro_batch_size=args.micro_batch_size,
        decoder_seq_length=args.decoder_seq_length,
        forward_only=False)

    timers = None
    # reset timers if necessary
    if config.timers is None:
        config.timers = timers

    # Empty unused memory.
    if args.empty_unused_memory_level >= 1:
        torch.cuda.empty_cache()

    update_successful, grad_norm, num_zeros_in_grad = optimizer.step(args, timers)

    # Update learning rate.
    if update_successful:
        increment = get_num_microbatches() * \
                    args.micro_batch_size * \
                    args.data_parallel_size
        opt_param_scheduler.step(increment=increment)
        skipped_iter = 0
    else:
        skipped_iter = 1

    # Empty unused memory.
    if args.empty_unused_memory_level >= 2:
        torch.cuda.empty_cache()

    if mpu.is_pipeline_last_stage(ignore_virtual=True):
        # Average loss across microbatches.
        loss_reduced = {}
        for key in losses_reduced[0]:
            losses_reduced_for_key = [x[key] for x in losses_reduced]
            loss_reduced[key] = sum(losses_reduced_for_key) / len(losses_reduced_for_key)
        return loss_reduced, skipped_iter, grad_norm, num_zeros_in_grad
    return {}, skipped_iter, grad_norm, num_zeros_in_grad


def evaluate(forward_step_func,
             data_iterator,
             model,
             config,
             verbose=False):
    """Evaluation."""
    args = get_args()

    # Turn on evaluation mode which disables dropout.
    model.eval()

    total_loss_dict = {}

    # make validation batch size independent from training batch size
    eval_batch_size = args.global_batch_size
    eval_num_microbatches = eval_batch_size // \
        (args.micro_batch_size * args.data_parallel_size)

    with torch.no_grad():
        iteration = 0
        if verbose:
            print_rank_0(
                f'Evaluating on {args.eval_iters * eval_batch_size} samples')
        while iteration < args.eval_iters:
            iteration += 1
            if verbose:
                print_rank_0(f'Evaluating iter {iteration}/{args.eval_iters}')

            forward_backward_func = get_forward_backward_func()
            # Don't care about timing during evaluation
            config.timers = None
            loss_dicts = forward_backward_func(
                forward_step_func=forward_step_func,
                data_iterator=data_iterator,
                model=model,
                num_microbatches=eval_num_microbatches,
                seq_length=args.seq_length,
                micro_batch_size=args.micro_batch_size,
                decoder_seq_length=args.decoder_seq_length,
                forward_only=True)

            # Empty unused memory
            if args.empty_unused_memory_level >= 1:
                torch.cuda.empty_cache()

            if mpu.is_pipeline_last_stage(ignore_virtual=True):
                # Reduce across processes.
                for loss_dict in loss_dicts:
                    for key in loss_dict:
                        total_loss_dict[key] = total_loss_dict.get(
                            key, torch.cuda.FloatTensor([0.0])) + loss_dict[key]

            args.consumed_valid_samples += eval_batch_size

    # Move model back to the train mode.
    model.train()

    for key in total_loss_dict:
        total_loss_dict[key] /= args.eval_iters * eval_num_microbatches

    return total_loss_dict
