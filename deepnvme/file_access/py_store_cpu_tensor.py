import torch
import os, timeit, functools
import pathlib
from utils import parse_write_arguments, GIGA_UNIT

def file_write(out_f, tensor):
    with open(out_f, 'wb') as f:
       f.write(tensor.numpy(force=True))

def main():
    args = parse_write_arguments()
    cnt = args.loop
    output_file = os.path.join(args.nvme_folder, f'test_ouput_{args.mb_size}MB.pt')
    pathlib.Path(output_file).unlink(missing_ok=True)
    file_sz = args.mb_size*(1024**2)
    cpu_tensor = torch.empty(file_sz, dtype=torch.uint8, device='cpu', requires_grad=False)

    t = timeit.Timer(functools.partial(file_write, output_file, cpu_tensor))

    py_t = t.timeit(cnt)
    py_gbs = (cnt*file_sz)/GIGA_UNIT/py_t
    print(f'py store_cpu: {file_sz/GIGA_UNIT} GB, {py_t/cnt} secs, {py_gbs:5.2f} GB/sec')
    pathlib.Path(output_file).unlink(missing_ok=True)

if __name__ == "__main__":
    main()
