#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# setup.py: install script for deepspeed_chat
"""
to install deepspeed_chat and its dependencies for development work,
run this cmd from the root directory:
    pip install -e .
"""
import setuptools

setuptools.setup(
    name="deepspeed-chat",
    version="0.1",
    url=
    "https://github.com/deepspeedai/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat",
    include_package_data=True,
    packages=setuptools.find_packages(include=['dschat']),
    install_requires=[
        "datasets>=2.8.0", "sentencepiece>=0.1.97", "protobuf==3.20.3",
        "accelerate>=0.15.0", "torch>=1.12.0", "deepspeed>=0.9.2",
        "transformers>=4.31.0,!=4.33.2", "tensorboard"
    ],
    extras_require={
        "azureml": [
            "azure-ml-component",
            "azureml-core",
        ],
    })
