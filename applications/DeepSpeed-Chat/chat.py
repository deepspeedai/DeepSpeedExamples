# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import argparse
import subprocess

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path",
                        type=str,
                        help="Directory containing trained actor model")
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum new tokens to generate per response",
    )
    args = parser.parse_args()

    cmd = f"python3 ./inference/chatbot.py --path {args.path} --max_new_tokens {args.max_new_tokens}"
    p = subprocess.Popen(cmd, shell=True)
    p.wait()
