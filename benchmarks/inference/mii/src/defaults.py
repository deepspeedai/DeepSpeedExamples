# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

ARG_DEFAULTS = {
    "tp_size": 1,
    "max_ragged_batch_size": 768,
    "num_replicas": 1,
    "max_prompt_length": 4000,
    "mean_prompt_length": 2600,
    "mean_max_new_tokens": 60,
}

MODEL_DEFAULTS = {
    "meta-llama/Llama-2-7b-hf": {
        "max_prompt_length": 4000,
        "mean_prompt_length": (1200, 2600),
        "mean_max_new_tokens": (60, 128),
        "tp_size": 1,
    },
    "meta-llama/Llama-13b-hf": {
        "max_prompt_length": 4000,
        "mean_prompt_length": (1200, 2600),
        "mean_max_new_tokens": (60, 128),
        "tp_size": (1, 2, 4),
    },
    "meta-llama/Llama-2-70b-hf": {
        "max_prompt_length": 4000,
        "mean_prompt_length": (1200, 2600),
        "mean_max_new_tokens": (60, 128),
        "tp_size": (4, 8),
    },
    "tiiuae/falcon-40B": {
        "max_prompt_length": 2000,
        "mean_prompt_length": (1200, 1900),
        "mean_max_new_tokens": (60, 128),
        "tp_size": (2, 4),
    },
    "tiiuae/falcon-180B": {
        "max_prompt_length": 2000,
        "mean_prompt_length": (1200, 1900),
        "mean_max_new_tokens": (60, 128),
        "tp_size": 8,
    },
    "microsoft/phi-2": {
        "max_prompt_length": 2000,
        "mean_prompt_length": (1200, 1900),
        "mean_max_new_tokens": (60, 128),
        "tp_size": 1,
    },
    "mistralai/Mixtral-8x7B-v0.1": {
        "max_prompt_length": 4000,
        "mean_prompt_length": (1200, 2600),
        "mean_max_new_tokens": (60, 128),
        "tp_size": 4,
    },
}
