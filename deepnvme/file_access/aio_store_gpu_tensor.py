import torch
import os, timeit, functools, pathlib
from deepspeed.ops.op_builder import AsyncIOBuilder
from utils import parse_write_arguments, GIGA_UNIT

def file_write(out_f, tensor, handle, bounce_buffer):
    bounce_buffer.copy_(tensor)
    handle.sync_pwrite(bounce_buffer, out_f)

def main():
    args = parse_write_arguments()
    cnt = args.loop
    output_file = os.path.join(args.nvme_folder, f'test_ouput_{args.mb_size}MB.pt')
    pathlib.Path(output_file).unlink(missing_ok=True)
    file_sz = args.mb_size*(1024**2)
    app_tensor = torch.empty(file_sz, dtype=torch.uint8, device='cuda', requires_grad=False)

    aio_handle = AsyncIOBuilder().load().aio_handle()
    bounce_buffer = torch.empty(file_sz, dtype=torch.uint8, requires_grad=False).pin_memory()


    t = timeit.Timer(functools.partial(file_write, output_file, app_tensor, aio_handle, bounce_buffer))

    aio_t = t.timeit(cnt)
    aio_gbs = (cnt*file_sz)/GIGA_UNIT/aio_t
    print(f'aio store_gpu: {file_sz/GIGA_UNIT} GB, {aio_t/cnt} secs, {aio_gbs:5.2f} GB/sec')

    if args.validate: 
        import tempfile, filecmp
        from py_store_cpu_tensor import file_write as py_file_write 
        py_ref_file = os.path.join(tempfile.gettempdir(), os.path.basename(output_file))
        py_file_write(py_ref_file, app_tensor)
        filecmp.clear_cache()
        print(f'Validation success = {filecmp.cmp(py_ref_file, output_file, shallow=False) }')

    pathlib.Path(output_file).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
