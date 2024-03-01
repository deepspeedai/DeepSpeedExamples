# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# Run benchmark against AML endpoint
python ./run_benchmark.py \
        --model <model name> \
        --deployment_name <aml deployment name> \
        --aml_api_url <aml endpoint URL> \
        --aml_api_key <aml API key> \
        --mean_prompt_length 2600 \
        --mean_max_new_tokens 60 \
        --num_requests 256 \
        --backend aml

### Gernerate the plots
python ./src/plot_th_lat.py

echo "Find figures in ./plots/ and log outputs in ./results/"