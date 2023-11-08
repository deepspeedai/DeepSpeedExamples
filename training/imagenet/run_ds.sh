#!/bin/bash

deepspeed main.py -a resnet50 --deepspeed --deepspeed_config config/ds_config.json --multiprocessing_distributed /home/pagolnar/clones/clone_imagenet/imagenet/imagenet
