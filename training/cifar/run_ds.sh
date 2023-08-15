#!/bin/bash

deepspeed --bind_cores_to_rank cifar10_deepspeed.py --deepspeed $@
