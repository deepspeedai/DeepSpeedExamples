
import sys
import argparse


def get_argument_parser():
    parser = argparse.ArgumentParser()

    # Required_parameter
    parser.add_argument("--config-file", "--cf",
                        help="pointer to the configuration file of the experiment", type=str, required=True)
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints will be written.")

    # Optional Params
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                        "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--max_predictions_per_seq", "--max_pred", default=80, type=int,
                        help="The maximum number of masked tokens in a sequence to be predicted.")
    parser.add_argument("--train_batch_size", default=32,
                        type=int, help="Total batch size for training.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--max_grad_norm',
                        type=float,
                        default=0.0,
                        help="The gradient clipping factor. Default: 0.0")
    parser.add_argument("--do_lower_case",
                        default=True,
                        action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--optimize_on_cpu',
                        default=False,
                        action='store_true',
                        help="Whether to perform optimization and keep the optimizer averages on CPU")
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--use_pretrain',
                        default=False,
                        action='store_true',
                        help="Whether to use Bert Pretrain Weights or not")

    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help='Loss scaling, positive power of 2 values can improve fp16 convergence.')

    parser.add_argument('--max_lamb',
                        type=float, default=2.0,
                        help='Max lamb Coeff.')

    parser.add_argument('--min_lamb',
                        type=float, default=0.01,
                        help='Min lamb Coeff.')

    parser.add_argument('--refresh_bucket_size',
                        type=int,
                        default=1,
                        help="This param makes sure that a certain task is repeated for this time steps to \
                            optimise on the back propogation speed with APEX's DistributedDataParallel")
    parser.add_argument('--finetune',
                        default=False,
                        action='store_true',
                        help="Whether to finetune only")

    parser.add_argument('--load_training_checkpoint', '--load_cp',
                        type=str,
                        default=None,
                        help="This is the path to the TAR file which contains model+opt state_dict() checkpointed.")
    parser.add_argument('--load_checkpoint_id', '--load_cp_id',
                        type=str,
                        default=None,
                        help='Checkpoint identifier to load from checkpoint path')
    parser.add_argument('--job_name',
                        type=str,
                        default=None,
                        help="This is the path to store the output and TensorBoard results.")

    parser.add_argument('--rewarmup',
                        default=False,
                        action='store_true',
                        help='Rewarmup learning rate after resuming from a checkpoint')

    parser.add_argument('--use_lamb',
                        default=False,
                        action='store_true',
                        help="Use deepspeed lamb")

    parser.add_argument('--delay_allreduce',
                        default=False,
                        action='store_true',
                        help='Delay all reduce to end of back propagation. Disable computation/communication overlap')

    parser.add_argument('--max_steps',
                        type=int,
                        default=sys.maxsize,
                        help='Maximum number of training steps of effective batch size to complete.')

    parser.add_argument('--max_steps_per_epoch',
                        type=int,
                        default=sys.maxsize,
                        help='Maximum number of training steps of effective batch size within an epoch to complete.')

    parser.add_argument('--print_steps',
                        type=int,
                        default=100,
                        help='Interval to print training details.')

    parser.add_argument('--wall_clock_breakdown',
                        default=False,
                        action='store_true',
                        help="Whether to display the breakdown of the wall-clock time for foraward, backward and step")

    return parser

def is_time_to_exit(args, epoch_steps=0, global_steps=0):
    return (epoch_steps >= args.max_steps_per_epoch) or \
            (global_steps >= args.max_steps)

