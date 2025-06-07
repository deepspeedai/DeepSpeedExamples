"""Pretrain utilities."""

from datetime import datetime
import math
import sys
import time

# The earliest we can measure the start time.
_TRAIN_START_TIME = time.time()

import torch
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP

from megatron import get_args
from megatron import get_signal_handler
from megatron import get_timers
from megatron import get_tensorboard_writer
from megatron import get_num_microbatches
from megatron import is_last_rank
from megatron import print_rank_0
from megatron import print_rank_last
from megatron import update_num_microbatches
from megatron.checkpointing import load_checkpoint
from megatron.checkpointing import save_checkpoint
from megatron.core import mpu, tensor_parallel
from megatron.core.enums import ModelType
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.utils import get_model_config
from megatron.data.data_samplers import build_pretraining_data_loader
from megatron.model import DistributedDataParallel as LocalDDP
from megatron.model import Float16Module
from megatron.model.vision.knn_monitor import compute_feature_bank
from megatron.optimizer import get_megatron_optimizer
from megatron.optimizer_param_scheduler import OptimizerParamScheduler
from megatron.initialize import initialize_megatron
from megatron.initialize import write_args_to_tensorboard
from megatron.initialize import set_jit_fusion_options
from megatron.utils import calc_params_l2_norm
from megatron.utils import check_adlr_autoresume_termination
from megatron.utils import report_memory
from torch.profiler import profile, ProfilerActivity

def is_rank_0():
    # if torch.cuda.current_device() == 0:
    if torch.distributed.get_rank() == 0:
        return True

def pretrain(model_builder,
             dataset_builder,
             forward_step_func):
    """Main training program.

    This function will run the followings in the order provided:
        1) initialize Megatron.
        2) setup model, optimizer and lr schedule using the model_provider.
        3) call train_val_test_data_provider to get train/val/test datasets.
        4) train the modle using the forward_step_func.

    Arguments:
        dataset_builder: a function that takes the size of
            train/valid/test dataset and returns `train, valid, test` datasets.
        model_builder: a function that returns a vanilla version of the
            model. By vanilla we mean a simple model on cpu with no fp16 or ddp.
        forward_step_func: a function that takes a `data iterator` and `model`,
            and returns a `loss` scalar with a dictionary with key:values being
            the info we would like to monitor during training, for example
            `lm-loss: value`. We also require that this function add
            `batch generator` to the timers class.
    """
    # Initalize and get arguments, timers, and Tensorboard writer.
    initialize_megatron(extra_args_provider=None,
                        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})
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
    print_datetime('after megatron is initialized')

    args = get_args()
    timers = get_timers()

    # Model, optimizer, and learning rate.
    timers('model-and-optimizer-setup', log_level=0).start(barrier=True)
    # model length is 1
    model, optimizer, opt_param_scheduler = setup_model_and_optimizer(
        model_builder, ModelType.encoder_or_decoder)
    timers('model-and-optimizer-setup').stop()
    print_datetime('after model, optimizer, and learning rate '
                   'scheduler are built')
    config = get_model_config(model)

    # Data stuff.
    timers('train/valid/test-data-iterators-setup', log_level=0).start(
        barrier=True)
    # Do not use virtual pipeline parallelism for data parallel
    train_data_iterator, valid_data_iterator, test_data_iterator \
        = get_dataset_iterator(dataset_builder)
    timers('train/valid/test-data-iterators-setup').stop()
    print_datetime('after dataloaders are built')

    # Print setup timing.
    print_rank_0('done with setup ...')
    timers.log(['model-and-optimizer-setup',
                'train/valid/test-data-iterators-setup'], barrier=True)
    
    # Train and eval.
    if not args.skip_train:
        print_rank_0('training ...')

        iteration = 0
        if args.do_train and args.train_iters > 0:
            iteration = train(forward_step_func,
                              model, optimizer, opt_param_scheduler,
                              train_data_iterator, valid_data_iterator, config)

        print_datetime('after training is done')

        if args.save and iteration != 0:
            save_checkpoint(iteration, model, optimizer, opt_param_scheduler)

    if args.do_valid:
        prefix = f'iteration {iteration} on validation set'
        evaluate_and_print_results(prefix, forward_step_func,
                                   valid_data_iterator, model,
                                   iteration, config,
                                   verbose=True, write_to_tensorboard=True)

    if args.do_test:
        prefix = f'iteration {iteration} on test set'
        evaluate_and_print_results(prefix, forward_step_func,
                                   test_data_iterator, model,
                                   iteration, config,
                                   verbose=True, write_to_tensorboard=True)


def print_datetime(string):
    """Note that this call will sync across all ranks."""
    torch.distributed.barrier()
    time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print_rank_0('[' + string + '] datetime: {} '.format(time_str))


def setup_model_and_optimizer(model_builder,
                              model_type,
                              no_wd_decay_cond=None,
                              scale_lr_cond=None,
                              lr_mult=1.0):
    """Setup model and optimizer."""
    args = get_args()

    model = get_model(model_builder, model_type)
    # Remove unwrap model

    if isinstance(model, list):
        models = model
    else:
        models = [model]
    optimizer = get_megatron_optimizer(models, no_wd_decay_cond,
                                       scale_lr_cond, lr_mult)
    opt_param_scheduler = get_optimizer_param_scheduler(optimizer)

    if args.load is not None:
        timers = get_timers()
        timers('load-checkpoint', log_level=0).start(barrier=True)
        args.iteration = load_checkpoint(model, optimizer, opt_param_scheduler)
        timers('load-checkpoint').stop(barrier=True)
        timers.log(['load-checkpoint'])
    else:
        args.iteration = 0

    return model, optimizer, opt_param_scheduler


def get_model(model_builder, model_type=ModelType.encoder_or_decoder, wrap_with_ddp=True):
    """Build the model."""
    args = get_args()
    args.model_type = model_type

    # Build model.
    # Remove pipeline model initialization.
    pre_process = mpu.is_pipeline_first_stage()
    post_process = mpu.is_pipeline_last_stage()
    # Only consider encoder or decoder model type
    if model_type != ModelType.encoder_and_decoder:
        model = model_builder(
                pre_process=pre_process,
                post_process=post_process
            )
        
    model.model_type = model_type

    # Disallow training and inference with Transformer Engine
    # for non-GPT models
    args.allow_transformer_engine = True
    assert args.allow_transformer_engine or args.transformer_impl == 'local', \
        'Transformer Engine is only approved for GPT models'

    # Set tensor model parallel attributes if not set.
    # Only parameters that are already tensor model parallel have these
    # attributes set for them. We should make sure the default attributes
    # are set for all params so the optimizer can use them.
    for param in model.parameters():
        tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes(
            param)

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
    # Fp16 conversion.
    # if args.fp16 or args.bf16:
    #     model = [Float16Module(model_module, args) for model_module in model]


    # from .transformer import ParallelTransformerLayer
    # graphs = []
    # for m in model:
    #     for i, module in enumerate(m.modules()):
    #         if isinstance(module, ParallelTransformerLayer):      
    #             # print("Building CUDA graph for transformer layer", i)
    #             module.build_cuda_graph()
    #             # graphs.append(module.g_)

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
        print("train_iters: ", args.train_iters)
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


def get_dataset_iterator(dataset_builder):
    """Build pretraining data iterators."""

    args = get_args()

    # Build loaders.
    train_dataloader, valid_dataloader, test_dataloader = \
        get_data_loader(dataset_builder)

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


def get_data_loader(dataset_builder):
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

        # Build datasets.
        train_ds, valid_ds, test_ds = get_dataset(dataset_builder)

        # Build dataloders.
        train_dataloader = build_pretraining_data_loader(
            train_ds, args.consumed_train_samples)
        if args.skip_train:
            valid_dataloader = build_pretraining_data_loader(valid_ds, 0)
        else:
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


def get_dataset(dataset_builder):
    """Build pretraining datasets."""

    args = get_args()

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

    # Build the datasets.
    return dataset_builder(train_val_test_num_samples)


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

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # timers('ite-time', log_level=0).start(barrier=True)
    num_prof_iter = 0
    total_time = 0.0
    start_iter = 40
    time_list = []

    # def trace_handler(prof):
    #     # output = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
    #     # print(output)
    #     if is_rank_0():
    #         prof.export_chrome_trace(f"/home/code/profile/8MI300_GPT_13b_b16s1024_hw_gpt3_async_1stmlp.json")

    # with profile(
    #     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    #     record_shapes=True,
    #     schedule=torch.profiler.schedule(wait=40, warmup=2, active=3),
    #     on_trace_ready=trace_handler
    # ) as prof:
    if True:
        while iteration < args.train_iters:

            start.record()
            args.curr_iteration = iteration
            # prof.step()
            torch.cuda.nvtx.range_push("Training interation")
            loss_dict, skipped_iter, grad_norm, num_zeros_in_grad = \
                train_step(forward_step_func,
                            train_data_iterator,
                            model,
                            optimizer,
                            opt_param_scheduler,
                            config)

            end.record()
            torch.cuda.synchronize()
            
            torch.cuda.nvtx.range_pop()

            iteration += 1
            args.consumed_train_samples += mpu.get_data_parallel_world_size() * \
                args.micro_batch_size * get_num_microbatches()
            
            ite_time = start.elapsed_time(end)

            if iteration > start_iter:
                total_time += ite_time
                num_prof_iter += 1
                time_list.append(ite_time)


            # ite_time = timers('ite-time').elapsed(barrier=True)
            if iteration % args.log_interval == 0 and is_rank_0():
                loss = loss_dict['lm loss'].item()
                print( 'iteration: {} | loss: {:.3f} | iteration time (ms): {} '.format(iteration, loss, ite_time))
                # loss_scale = optimizer.cur_scale
                # lr = optimizer.param_groups[0]['lr']
                # print( 'lr: {} loss scale: {:.1f} |'.format(lr, loss_scale))'

    if is_rank_0() and num_prof_iter > 0:
        print("Average time of the last", args.train_iters - start_iter,  "iterations time(ms): ", total_time/num_prof_iter)
        print("Min iteration time: ", min(time_list))
    return iteration

# def train(forward_step_func, model, optimizer, opt_param_scheduler,
#           train_data_iterator, valid_data_iterator, config):
#     """Train the model function."""
#     args = get_args()
#     timers = get_timers()

#     # Write args to tensorboard
#     write_args_to_tensorboard()
#     # def count_params(model, is_human: bool = False):
#     #     params: int = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     #     return f"{params / 1e6:.2f}M" if is_human else params

#     # print("Total # of params:", count_params(model, is_human=True))
#     # Turn on training mode which enables dropout.
#     model.train()

#     # Tracking loss.
#     total_loss_dict = {}

#     # Iterations.
#     iteration = args.iteration

#     # Setup some training config params
#     config.grad_scale_func = optimizer.scale_loss
#     config.timers = timers

#     timers('interval-time', log_level=0).start(barrier=True)
#     print_datetime('before the start of training step')
#     report_memory_flag = False #True

#     # def trace_handler(prof):
#     #     # output = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
#     #     # print(output)
#     #     if is_rank_0():
#     #         prof.export_chrome_trace(f"./traces/1node_13b_mb8_domino_flash-attn.json")

#     # with profile(
#     #     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
#     #     record_shapes=True,
#     #     schedule=torch.profiler.schedule(wait=30, warmup=2, active=3),
#     #     on_trace_ready=trace_handler
#     # ) as prof:
#     if True:
#         while iteration < args.train_iters:
#             if args.profile and \
#                     iteration == args.profile_step_start and \
#                     torch.distributed.get_rank() in args.profile_ranks:
#                 torch.cuda.cudart().cudaProfilerStart()
#                 torch.autograd.profiler.emit_nvtx(
#                     record_shapes=True).__enter__()

#             update_num_microbatches(args.consumed_train_samples)
#             args.curr_iteration = iteration
#             loss_dict, skipped_iter, grad_norm, num_zeros_in_grad = \
#                 train_step(forward_step_func,
#                            train_data_iterator,
#                            model,
#                            optimizer,
#                            opt_param_scheduler,
#                            config)
#             iteration += 1
#             # prof.step()
#             args.consumed_train_samples += mpu.get_data_parallel_world_size() * \
#                 args.micro_batch_size * \
#                 get_num_microbatches()

#             # Logging.
#             loss_scale = optimizer.get_loss_scale().item()
#             params_norm = None
#             if args.log_params_norm:
#                 params_norm = calc_params_l2_norm(model)
#             report_memory_flag = training_log(loss_dict, total_loss_dict,
#                                               optimizer.param_groups[0]['lr'],
#                                               iteration, loss_scale,
#                                               report_memory_flag, skipped_iter,
#                                               grad_norm, params_norm, num_zeros_in_grad)#, model)

#             # Evaluation
#             if args.eval_interval and iteration % args.eval_interval == 0 and \
#                     args.do_valid:
#                 prefix = 'iteration {}'.format(iteration)
#                 evaluate_and_print_results(prefix, forward_step_func,
#                                            valid_data_iterator, model,
#                                            iteration, config, False)
            
#             # Checkpointing
#             saved_checkpoint = False
#             if args.exit_signal_handler:
#                 signal_handler = get_signal_handler()
#                 if any(signal_handler.signals_received()):
#                     save_checkpoint_and_time(iteration, model, optimizer,
#                                              opt_param_scheduler)
#                     print_datetime('exiting program after receiving SIGTERM.')
#                     sys.exit()

#             if args.save and args.save_interval and \
#                     iteration % args.save_interval == 0:
#                 save_checkpoint_and_time(iteration, model, optimizer,
#                                          opt_param_scheduler)
#                 saved_checkpoint = True

#             if args.profile and \
#                     iteration == args.profile_step_end and \
#                     torch.distributed.get_rank() in args.profile_ranks:
#                 torch.cuda.cudart().cudaProfilerStop()

#     return iteration


def is_rank_0():
    # if torch.cuda.current_device() == 0:
    if torch.distributed.get_rank() == 0:
        return True

# def get_parameters_in_billions(model):
#     gpus_per_model = torch.distributed.get_world_size(group=mpu.get_model_parallel_group())

#     approx_parameters_in_billions = sum([sum([p.ds_numel if hasattr(p,'ds_id') else  p.nelement() for p in model_module.parameters()])
#                                         for model_module in model])

#     return approx_parameters_in_billions*gpus_per_model/(1e9)

# def throughput_calculator(model, args, iteration_time, total_iterations):
#     batch_size = args.micro_batch_size * get_num_microbatches() * args.data_parallel_size
#     # approx_parameters_in_billions = None if (model is None) else get_parameters_in_billions(model)
#     elapsed_time_per_iter = iteration_time/total_iterations
#     samples_per_second = batch_size / elapsed_time_per_iter

#     #flops calculator
#     hidden_size = args.hidden_size
#     num_layers = args.num_layers
#     vocab_size = args.padded_vocab_size

#     # General TFLOPs formula (borrowed from Equation 3 in Section 5.1 of
#     # https://arxiv.org/pdf/2104.04473.pdf).
#     # The factor of 4 is when used with activation check-pointing,
#     # otherwise it will be 3.
#     checkpoint_activations_factor = 3
#     if hasattr(args, 'checkpoint_activations') and args.checkpoint_activations:
#         checkpoint_activations_factor = 4
#     if hasattr(args, 'recompute_granularity') and (args.recompute_granularity == 'selective' or args.recompute_granularity == 'full'):
#         checkpoint_activations_factor = 4
#     seq_len = args.seq_length
#     if hasattr(args, 'actual_seq_length'):
#         seq_len = args.actual_seq_length
#     flops_per_iteration = (24 * checkpoint_activations_factor * batch_size * seq_len * num_layers * (hidden_size**2)) * (1. + (seq_len / (6. * hidden_size)) + (vocab_size / (16. * num_layers * hidden_size)))
#     tflops = flops_per_iteration / (elapsed_time_per_iter * args.world_size * (1e12))
#     return samples_per_second, tflops


def training_log(loss_dict, total_loss_dict, learning_rate, iteration,
                 loss_scale, report_memory_flag, skipped_iter,
                 grad_norm, params_norm, num_zeros_in_grad): #, model):
    """Log training information such as losses, timing, ...."""
    args = get_args()
    timers = get_timers()
    writer = get_tensorboard_writer()

    # Advanced, skipped, and Nan iterations.
    advanced_iters_key = 'advanced iterations'
    skipped_iters_key = 'skipped iterations'
    nan_iters_key = 'nan iterations'
    # Advanced iterations.
    if not skipped_iter:
        total_loss_dict[advanced_iters_key] = total_loss_dict.get(
            advanced_iters_key, 0) + 1
    else:
        if advanced_iters_key not in total_loss_dict:
            total_loss_dict[advanced_iters_key] = 0
    # Skipped iterations.
    total_loss_dict[skipped_iters_key] = total_loss_dict.get(
        skipped_iters_key, 0) + skipped_iter
    # Update losses and set nan iterations
    got_nan = False
    for key in loss_dict:
        if not skipped_iter:
            total_loss_dict[key] = total_loss_dict.get(
                key, torch.cuda.FloatTensor([0.0])) + loss_dict[key]
        else:
            value = loss_dict[key].float().sum().item()
            is_nan = value == float('inf') or \
                     value == -float('inf') or \
                     value != value
            got_nan = got_nan or is_nan
    total_loss_dict[nan_iters_key] = total_loss_dict.get(
        nan_iters_key, 0) + int(got_nan)

    # Logging.
    timers_to_log = [
        'forward-backward',
        'forward-compute',
        'backward-compute',
        'batch-generator',
        'forward-recv',
        'forward-send',
        'backward-recv',
        'backward-send',
        'forward-send-forward-recv',
        'forward-send-backward-recv',
        'backward-send-forward-recv',
        'backward-send-backward-recv',
        'forward-backward-send-forward-backward-recv',
        'layernorm-grads-all-reduce',
        'embedding-grads-all-reduce',
        'grads-all-reduce',
        'grads-reduce-scatter',
        'params-all-gather',
        'optimizer-copy-to-main-grad',
        'optimizer-unscale-and-check-inf',
        'optimizer-clip-main-grad',
        'optimizer-count-zeros',
        'optimizer-inner-step',
        'optimizer-copy-main-to-model-params',
        'optimizer']

    # Calculate batch size.
    batch_size = args.micro_batch_size * args.data_parallel_size * \
        get_num_microbatches()

    total_iterations = total_loss_dict[advanced_iters_key] + \
                       total_loss_dict[skipped_iters_key]

    # Tensorboard values.
    # Timer requires all the ranks to call.
    if args.log_timers_to_tensorboard and \
       (iteration % args.tensorboard_log_interval == 0):
        timers.write(timers_to_log, writer, iteration,
                     normalizer=total_iterations)
    if writer and (iteration % args.tensorboard_log_interval == 0):
        if args.log_learning_rate_to_tensorboard:
            writer.add_scalar('learning-rate', learning_rate, iteration)
            writer.add_scalar('learning-rate vs samples', learning_rate,
                              args.consumed_train_samples)
        if args.log_batch_size_to_tensorboard:
            writer.add_scalar('batch-size', batch_size, iteration)
            writer.add_scalar('batch-size vs samples', batch_size,
                              args.consumed_train_samples)
        for key in loss_dict:
            writer.add_scalar(key , loss_dict[key], iteration)
            writer.add_scalar(key + ' vs samples', loss_dict[key],
                              args.consumed_train_samples)
        if args.log_loss_scale_to_tensorboard:
            writer.add_scalar('loss-scale', loss_scale, iteration)
            writer.add_scalar('loss-scale vs samples', loss_scale,
                              args.consumed_train_samples)
        if args.log_world_size_to_tensorboard:
            writer.add_scalar('world-size', args.world_size, iteration)
            writer.add_scalar('world-size vs samples', args.world_size,
                              args.consumed_train_samples)
        if grad_norm is not None:
            writer.add_scalar('grad-norm', grad_norm, iteration)
            writer.add_scalar('grad-norm vs samples', grad_norm,
                              args.consumed_train_samples)
        if num_zeros_in_grad is not None:
            writer.add_scalar('num-zeros', num_zeros_in_grad, iteration)
            writer.add_scalar('num-zeros vs samples', num_zeros_in_grad,
                              args.consumed_train_samples)
        if params_norm is not None:
            writer.add_scalar('params-norm', params_norm, iteration)
            writer.add_scalar('params-norm vs samples', params_norm,
                              args.consumed_train_samples)
        if args.log_memory_to_tensorboard:
            mem_stats = torch.cuda.memory_stats()
            writer.add_scalar(
                "mem-reserved-bytes",
                mem_stats["reserved_bytes.all.current"],
                iteration,
            )
            writer.add_scalar(
                "mem-allocated-bytes",
                mem_stats["allocated_bytes.all.current"],
                iteration,
            )
            writer.add_scalar(
                "mem-allocated-count",
                mem_stats["allocation.all.current"],
                iteration,
            )

    if iteration % args.log_interval == 0:
        elapsed_time = timers('interval-time').elapsed(barrier=True)
        # _, tflops = throughput_calculator(model, args, elapsed_time, total_iterations)
        elapsed_time_per_iteration = elapsed_time / total_iterations
        if writer:
            if args.log_timers_to_tensorboard:
                writer.add_scalar('iteration-time',
                                  elapsed_time_per_iteration, iteration)
        log_string = ' iteration {:8d}/{:8d} |'.format(
            iteration, args.train_iters)
        # log_string += ' consumed samples: {:12d} |'.format(
        #     args.consumed_train_samples)
        log_string += ' elapsed time per iteration (ms): {:.1f} |'.format(
            elapsed_time_per_iteration * 1000.0)
        # log_string += ' tflops: {:.1f} |'.format(tflops)
        log_string += ' learning rate: {:.3E} |'.format(learning_rate)
        # log_string += ' global batch size: {:5d} |'.format(batch_size)
        for key in total_loss_dict:
            if key not in [advanced_iters_key, skipped_iters_key,
                           nan_iters_key]:
                avg = total_loss_dict[key].item() / \
                      float(max(1, total_loss_dict[advanced_iters_key]))
                if avg > 0.0:
                    log_string += ' {}: {:.6E} |'.format(key, avg)
                total_loss_dict[key] = torch.cuda.FloatTensor([0.0])
        log_string += ' loss scale: {:.1f} |'.format(loss_scale)
        # if grad_norm is not None:
        #     log_string += ' grad norm: {:.3f} |'.format(grad_norm)
        # if num_zeros_in_grad is not None:
        #     log_string += ' num zeros: {:.1f} |'.format(num_zeros_in_grad)
        if params_norm is not None:
            log_string += ' params norm: {:.3f} |'.format(params_norm)
        log_string += ' number of skipped iterations: {:3d} |'.format(
            total_loss_dict[skipped_iters_key])
        log_string += ' number of nan iterations: {:3d} |'.format(
            total_loss_dict[nan_iters_key])
        total_loss_dict[advanced_iters_key] = 0
        total_loss_dict[skipped_iters_key] = 0
        total_loss_dict[nan_iters_key] = 0
        print_rank_last(log_string)
        if report_memory_flag and learning_rate > 0.:
            # Report memory after optimizer state has been initialized.
            report_memory('(after {} iterations)'.format(iteration))
            report_memory_flag = False
        timers.log(timers_to_log, normalizer=args.log_interval)

    return report_memory_flag


def train_step(forward_step_func, data_iterator,
               model, optimizer, opt_param_scheduler, config):
    """Single training step."""
    args = get_args()
    timers = get_timers()

    # Set grad to zero.
    if args.DDP_impl == 'local' and args.use_contiguous_buffers_in_local_ddp:
        model.zero_grad_buffer()
    optimizer.zero_grad()

    # Forward pass.
    timers('forward-backward', log_level=1).start(
        barrier=args.barrier_with_L1_time)
    forward_backward_func = get_forward_backward_func()

    # set timers to None if none of the timers in fwd_bwd are active, just to save the checks
    if args.timing_log_level < 2:
        config.timers = None

    losses_reduced = forward_backward_func(
        forward_step_func=forward_step_func,
        data_iterator=data_iterator,
        model=model,
        num_microbatches=get_num_microbatches(),
        seq_length=args.seq_length,
        micro_batch_size=args.micro_batch_size,
        decoder_seq_length=args.decoder_seq_length,
        forward_only=False)

    # reset timers if necessary
    if config.timers is None:
        config.timers = timers
    timers('forward-backward').stop()

    # Empty unused memory.
    if args.empty_unused_memory_level >= 1:
        torch.cuda.empty_cache()

    # Reduce gradients.
    optimizer.reduce_model_grads(args, timers)

    # Update parameters.
    timers('optimizer', log_level=1).start(barrier=args.barrier_with_L1_time)
    update_successful, grad_norm, num_zeros_in_grad = optimizer.step(args, timers)
    timers('optimizer').stop()

    # Gather params.
    if update_successful:
        optimizer.gather_model_params(args, timers)

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


def save_checkpoint_and_time(iteration, model, optimizer, opt_param_scheduler):
    timers = get_timers()
    # Extra barrier is added to make sure
    # all ranks report the max time.
    timers('save-checkpoint', log_level=0).start(barrier=True)
    save_checkpoint(iteration, model, optimizer, opt_param_scheduler)
    timers('save-checkpoint').stop(barrier=True)
    timers.log(['save-checkpoint'])


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
            config.timers = get_timers()

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


def evaluate_and_print_results(prefix, forward_step_func,
                               data_iterator, model,
                               iteration, config,
                               verbose=False, write_to_tensorboard=True):
    """Helper function to evaluate and dump results on screen."""
    args = get_args()
    if write_to_tensorboard:
        writer = get_tensorboard_writer()
    else:
        writer = None

    total_loss_dict = evaluate(
        forward_step_func, data_iterator, model, config, verbose)
    string = ' validation loss at {} | '.format(prefix)
    for key in total_loss_dict:
        string += '{} value: {:.6E} | '.format(key,
                                               total_loss_dict[key].item())
        ppl = math.exp(min(20, total_loss_dict[key].item()))
        string += '{} PPL: {:.6E} | '.format(key, ppl)
        if writer:
            writer.add_scalar('{} validation'.format(key),
                              total_loss_dict[key].item(),
                              iteration)
            writer.add_scalar('{} validation vs samples'.format(key),
                              total_loss_dict[key].item(),
                              args.consumed_train_samples)
            if args.log_validation_ppl_to_tensorboard:
                writer.add_scalar('{} validation ppl'.format(key), ppl,
                                  iteration)
                writer.add_scalar('{} validation ppl vs samples'.format(key),
                                  ppl, args.consumed_train_samples)

    length = len(string) + 1
    print_rank_last('-' * length)
    print_rank_last(string)
    print_rank_last('-' * length)