import torch
import os, timeit, functools
from utils import parse_read_arguments, GIGA_UNIT
from deepspeed.ops.op_builder import GDSBuilder

def file_read(inp_f, handle, gpu_buffer):
    handle.sync_pread(gpu_buffer, inp_f)
    return gpu_buffer.cuda()

def main():
    args = parse_read_arguments()
    input_file = args.input_file
    file_sz = os.path.getsize(input_file)
    cnt = args.loop

    gds_handle = GDSBuilder().load().gds_handle()
    gds_buffer = gds_handle.new_pinned_device_tensor(file_sz, torch.empty(0, dtype=torch.uint8, device='cuda', requires_grad=False))

    t = timeit.Timer(functools.partial(file_read, input_file, gds_handle, gds_buffer))
    gds_t = t.timeit(cnt)
    gds_gbs = (cnt*file_sz)/GIGA_UNIT/gds_t
    print(f'gds load_gpu: {file_sz/GIGA_UNIT} GB, {gds_t/cnt} secs, {gds_gbs:5.2f} GB/sec')

    if args.validate: 
        from py_load_cpu_tensor import file_read as py_file_read 
        aio_tensor = file_read(input_file, gds_handle, gds_buffer).cpu()
        py_tensor = py_file_read(input_file)
        print(f'Validation success = {aio_tensor.equal(py_tensor)}')

    gds_handle.free_pinned_device_tensor(gds_buffer)

if __name__ == "__main__":
    main()
