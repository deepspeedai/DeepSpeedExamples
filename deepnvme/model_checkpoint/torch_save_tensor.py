import time
import argparse
import torch
import os
from torch_save_utils import PINNED_BUFFER_MB
from torch_save_utils import test_save, test_ds_mock_save, test_ds_py_save, test_ds_aio_fast_save, test_ds_gds_fast_save
import deepspeed 
from deepspeed.accelerator import get_accelerator
import deepspeed.comm as dist
import os 

def run(args):
    device = get_accelerator().current_device_name() if args.gpu else 'cpu'
    buffer = torch.randint(high=128,
                           size=(args.mb_size * (1024**2), ),
                           dtype=torch.uint8,
                           device=device)

    fn_dict = {
        'test_save': test_save,
        'test_ds_mock_save': test_ds_mock_save,
        'test_ds_py_save': test_ds_py_save,
        'test_ds_gds_fast_save': test_ds_gds_fast_save,
        'test_ds_aio_fast_save': test_ds_aio_fast_save,
    }
    for tag, fn in fn_dict.items():
        if tag == 'test_ds_gds_fast_save' and not args.gpu:
            continue 
        file = os.path.join(args.folder, f'{tag}_{args.mb_size}MB.pt')
        print(f'checkpoint file = {file}')
        if os.path.isfile(file):
            os.remove(file)
        st = time.time()
        write_sec = fn(file, buffer, args)
        gb_per_sec = args.mb_size / (1024.0 * write_sec)
        gb_size = os.path.getsize(file) / (1024**3)
        print(
            f'{tag} -- {gb_size:5.2f} GB, {write_sec:5.2f} secs, {gb_per_sec:5.2f} GB/s'
        )
        print(f'*********************************************')



def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder',
                        default=None,
                        type=str,
                        required=True,
                        help='Folder to use for I/O.')
    parser.add_argument('--mb_size',
                        type=int,
                        default=None,
                        required=True,
                        help='Size of tensor to save in MB.')
    parser.add_argument('--legacy',
                        action='store_true',
                        help='Use torch legacy save format')

    parser.add_argument('--gpu', action='store_true', help='Use gpu tensors.')

    parser.add_argument('--io_buffer_mb',
                        type=int,
                        default=PINNED_BUFFER_MB,
                        help='Size of pinned i/o buffer in MB.')

    parser.add_argument('--no-statistics',
                        action='store_true',
                        help='Suppress low-level performance statistics.')

    parser.add_argument('--single_io_buffer',
                        action='store_true',
                        help='Disable double buffering of i/o buffer.')
    parser.add_argument('--local_rank',
                        type=int,
                        default=0,
                        help='Local rank' )

    args = parser.parse_args()
    print(f'args = {args}')
    return args


def main():
    print(
        f'Performance test of torch.save() integration of fast tensor checkpointing.'
    )
    args = parse_arguments()
    if not os.path.exists(args.folder):
        print(f'Invalid folder: {args.folder}')
        quit()
    deepspeed.init_distributed()
    run(args)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
