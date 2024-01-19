# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# Run benchmark
python ./run_benchmark.py \
        --model meta-llama/Llama-2-7b-hf \
        --tp_size 1 \
        --num_replicas 1 \
        --max_ragged_batch_size 768 \
        --mean_prompt_length 2600 \
        --mean_max_new_tokens 60 \
        --stream

### Gernerate the plots
python ./src/plot_th_lat.py

echo "Find figures in ./plots/ and log outputs in ./results/"