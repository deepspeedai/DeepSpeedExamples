#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

############Note that The following script is *without* batch-norm############
## Step 1: obtain a checkpoint (to be compressed)
python -m torch.distributed.launch --nproc_per_node=1 \
  --master_port 6665 \
  train.py \
  --deepspeed_config config/ds_config.json \
  --deepspeed --batch-norm --epochs 10
### Step 2: compress: channel pruning 
### you may enbale other compression methods, see ds_config.json or our compression tutorial 
python -m torch.distributed.launch --nproc_per_node=1 \
  --master_port 66665 \
  train.py \
  --deepspeed_config config/ds_config_channel_prune.json \
  --deepspeed \
  --epochs 10 --batch-norm \
  --compression  --path-to-model ./checkpoints/net.pkl \
  --saving-folder ./checkpoints/
  


############Note that The following Script is **with** batch-norm
### Step 1: obtain a checkpoint (to be compressed)
# python -m torch.distributed.launch --nproc_per_node=1 \
#   --master_port 6665 \
#   train.py \
#   --deepspeed_config config/ds_config.json \
#   --deepspeed --epochs 3
##### the output here is 
#### Step 2: compress: channel pruning 
#### you may enbale other compression methods, see ds_config.json or our compression tutorial 
# python -m torch.distributed.launch --nproc_per_node=1 \
#   --master_port 66665 \
#   train.py \
#   --deepspeed_config config/ds_config_channel_prune.json \
#   --deepspeed \
#   --epochs 3 \
#   --compression  --path-to-model ./checkpoints/net.pkl \
#   --saving-folder ./checkpoints/
  

