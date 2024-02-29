# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

MODELS=(meta-llama/Llama-2-7b-hf meta-llama/Llama-2-13b-hf meta-llama/Llama-2-70b-hf tiiuae/falcon-40B tiiuae/falcon-180B microsoft/phi-2 mistralai/Mixtral-8x7B-v0.1)

for MODEL in ${MODELS[@]}; do
    python ./run_benchmark.py --model ${MODEL} --stream --backend fastgen
    python ./run_benchmark.py --model ${MODEL} --stream --backend vllm
done

# Extra runs for Mixtral with non-default settings
python ./run_benchmark.py --model mistralai/Mixtral-8x7B-v0.1 --stream --tp_size 4 --mean_prompt_length 500 --mean_max_new_tokens 150 500 1024 --backend fastgen
python ./run_benchmark.py --model mistralai/Mixtral-8x7B-v0.1 --stream --tp_size 4 --mean_prompt_length 500 --mean_max_new_tokens 150 500 1024 --backend vllm