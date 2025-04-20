import torch
import os, timeit, functools
from deepspeed.ops.op_builder import AsyncIOBuilder
from utils import parse_read_arguments, GIGA_UNIT
from deepspeed.accelerator import get_accelerator

def file_read(inp_f, handle, bounce_buffer):
    handle.sync_pread(bounce_buffer, inp_f)
    return bounce_buffer.cpu()

def main():
    args = parse_read_arguments()
    input_file = args.input_file
    file_sz = os.path.getsize(input_file)
    cnt = args.loop

    aio_handle = AsyncIOBuilder().load().aio_handle()
    native_locked_tensor = get_accelerator()._name == 'cpu'

    if native_locked_tensor:
        bounce_buffer = aio_handle.new_cpu_locked_tensor(file_sz, torch.Tensor().to(torch.uint8))
    else:
        bounce_buffer = torch.empty(file_sz, dtype=torch.uint8).pin_memory()

    t = timeit.Timer(functools.partial(file_read, input_file, aio_handle, bounce_buffer))
    aio_t = t.timeit(cnt)
    aio_gbs = (cnt*file_sz)/GIGA_UNIT/aio_t
    print(f'aio load_cpu: {file_sz/GIGA_UNIT} GB, {aio_t/cnt} secs, {aio_gbs:5.2f} GB/sec')

    if args.validate: 
        from py_load_cpu_tensor import file_read as py_file_read 
        aio_tensor = file_read(input_file, aio_handle, bounce_buffer)
        py_tensor = py_file_read(input_file)
        print(f'Validation success = {aio_tensor.equal(py_tensor)}')

    if native_locked_tensor:
        aio_handle.free_cpu_locked_tensor(bounce_buffer)

if __name__ == "__main__":
    main()
