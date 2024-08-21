import torch
import os, timeit, functools
from utils import parse_read_arguments, GIGA_UNIT

def file_read(inp_f):
    with open(inp_f, 'rb') as f:
       tensor = torch.frombuffer(f.read(), dtype=torch.uint8)
    return tensor

def main():
    args = parse_read_arguments()
    input_file = args.input_file
    file_sz = os.path.getsize(input_file)
    cnt = args.loop

    t = timeit.Timer(functools.partial(file_read, input_file))
    py_t = t.timeit(cnt)
    py_gbs = (cnt*file_sz)/GIGA_UNIT/py_t
    print(f'py load_cpu: {file_sz/GIGA_UNIT} GB, {py_t/cnt} secs, {py_gbs:5.2f} GB/sec')

if __name__ == "__main__":
    main()
