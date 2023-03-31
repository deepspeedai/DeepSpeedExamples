
### use "deepspeed --help" to know more about deepspeed
#if you want to test with dummy dataset, use --dummy #<<=========================================================
#################################################################################
## Note that if you use 16GPU and set batch-size to be 256, then the micro-batch-size will be 256/16=16
## so you NEED to set train_batch_size=256 and train_micro_batch_size_per_gpu=16 in "config/ds_config_imagenet_random_ltd.json"
used_gpu=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
mkdir out/imagenet
deepspeed  --include worker-0:${used_gpu} --master_port 60000  main_imagenet.py /blob/data/ImageNetData/ --seed 2345    \
                --deepspeed_config config/ds_config_imagenet_random_ltd.json  \
                --deepspeed    --random_ltd          \
                -a vits16r224  \
                --img_size 224  --print-freq 100 --batch-size 256 \
                --lr 5e-05 --scheduler constant  \
                --epochs 14 \
                --out_dir out/imagenet | tee -a out/imagenet/training.log


# python main_imagenet.py /blob/data/ImageNetData/ --seed 2345    \
#                 --deepspeed_config config/ds_config.json  \
#                 --deepspeed              \
#                 --dist-url 'tcp://127.0.0.1:65402'  --dist-backend 'nccl' \
#                 --multiprocessing-distributed            \
#                 -a vits16r224 --world-size 1 --rank 0 \
#                 --img_size 224  --print-freq 100 --batch-size 256 \
#                 --lr 5e-05 --scheduler constant --dummy   \
#                 --epochs 14 \
#                 --out_dir onstant_tokenInterval0.8_tokenInit66_pass1-21layer | tee -a train_log.txt

