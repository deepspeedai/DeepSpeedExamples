import torch
import os, timeit, functools, pathlib
from deepspeed.ops.op_builder import GDSBuilder
from utils import parse_write_arguments, GIGA_UNIT

def file_write(out_f, tensor, handle, gpu_buffer):
    gpu_buffer.copy_(tensor)
    handle.sync_pwrite(gpu_buffer, out_f)

def main():
    args = parse_write_arguments()
    cnt = args.loop
    output_file = os.path.join(args.nvme_folder, f'test_ouput_{args.mb_size}MB.pt')
    pathlib.Path(output_file).unlink(missing_ok=True)
    file_sz = args.mb_size*(1024**2)
    app_tensor = torch.empty(file_sz, dtype=torch.uint8, device='cuda', requires_grad=False)

    gds_handle = GDSBuilder().load().gds_handle()
    gds_buffer = gds_handle.new_pinned_device_tensor(file_sz, torch.empty(0, dtype=torch.uint8, device='cuda', requires_grad=False))

    t = timeit.Timer(functools.partial(file_write, output_file, app_tensor, gds_handle, gds_buffer))

    gds_t = t.timeit(cnt)
    gds_gbs = (cnt*file_sz)/GIGA_UNIT/gds_t
    print(f'gds store_gpu: {file_sz/GIGA_UNIT} GB, {gds_t/cnt} secs, {gds_gbs:5.2f} GB/sec')

    if args.validate: 
        import tempfile, filecmp
        from py_store_cpu_tensor import file_write as py_file_write 
        py_ref_file = os.path.join(tempfile.gettempdir(), os.path.basename(output_file))
        py_file_write(py_ref_file, app_tensor)
        filecmp.clear_cache()
        print(f'Validation success = {filecmp.cmp(py_ref_file, output_file, shallow=False) }')

    gds_handle.free_pinned_device_tensor(gds_buffer)
    pathlib.Path(output_file).unlink(missing_ok=True)

if __name__ == "__main__":
    main()
