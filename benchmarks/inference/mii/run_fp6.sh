# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

MODELS=(NousResearch/Llama-2-70b-hf)

for MODEL in ${MODELS[@]}; do
    python ./run_benchmark.py --model ${MODEL} --num_requests 128 --stream --backend fastgen --fp6  --tp_size 1
done