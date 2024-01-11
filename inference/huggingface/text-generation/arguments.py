from argparse import ArgumentParser
import os

parser = ArgumentParser()

parser.add_argument("--model", required=True, type=str, help="model_name")
parser.add_argument("--checkpoint_path", required=False, default=None, type=str, help="model checkpoint path")
parser.add_argument("--save_mp_checkpoint_path", required=False, default=None, type=str, help="save-path to store the new model checkpoint")
parser.add_argument("--batch_size", default=1, type=int, help="batch size")
parser.add_argument("--dtype", default="float16", type=str, choices=["float32", "float16", "int8", "bfloat16"], help="data-type")
parser.add_argument("--hf_baseline", action='store_true', help="disable DeepSpeed inference")
parser.add_argument("--use_kernel", action='store_true', help="enable kernel-injection")
parser.add_argument("--max_tokens", default=1024, type=int, help="maximum tokens used for the text-generation KV-cache")
parser.add_argument("--max_new_tokens", default=50, type=int, help="maximum new tokens to generate")
parser.add_argument("--greedy", action='store_true', help="greedy generation mode")
parser.add_argument("--use_meta_tensor", action='store_true', help="use the meta tensors to initialize model")
parser.add_argument("--test_performance", action='store_true', help="enable latency, bandwidth, and throughout testing")
parser.add_argument("--local_rank", type=int, default=int(os.getenv("LOCAL_RANK", "0")), help="local rank")
parser.add_argument("--world_size", type=int, default=int(os.getenv("WORLD_SIZE", "1")), help="world_size")
parser.add_argument("--test_hybrid_engine", action='store_true', help="enable hybrid engine testing")
parser.add_argument("--trust_remote_code", action='store_true', help="Trust remote code for hugging face models")