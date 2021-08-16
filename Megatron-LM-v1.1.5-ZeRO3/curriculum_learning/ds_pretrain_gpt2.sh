#! /bin/bash

CONFIG=$1
TAG=$2
MODEL_SIZE=$3
LR=$4
TOTAL_BATCHSIZE=$5
SEQ_LEN=$6
MP_SIZE=$7
SEED=$8
SAVE_INTERVAL=$9
NUM_ITER=${10}
NUM_TOKEN=${11}
LR_DECAY_ITER=${12}
LR_WARMUP_ITER=${13}
CONFIG_TEMPLATE=${14}
CURRICULUM_STEP=${15}
CURRICULUM_MIN=${16}

# 12-layer, 768-hidden, 12-heads, 117M parameters
# 24-layer, 1024-hidden, 16-heads, 345M parameters
# 36-layer, 1280-hidden, 20-heads, 774M parameters
# 48-layer, 1600-hidden, 25-heads, 1558M parameters
if [[ $MODEL_SIZE -eq 117 ]]; then
        NUM_LAYERS=12
        HIDDEN_SIZE=768
        NUM_ATTN_HEADS=12
elif [[ $MODEL_SIZE -eq 345 ]]; then
        NUM_LAYERS=24
        HIDDEN_SIZE=1024
        NUM_ATTN_HEADS=16
elif [[ $MODEL_SIZE -eq 774 ]]; then
        NUM_LAYERS=36
        HIDDEN_SIZE=1280
        NUM_ATTN_HEADS=20
elif [[ $MODEL_SIZE -eq 1558 ]]; then
        NUM_LAYERS=48
        HIDDEN_SIZE=1600
        NUM_ATTN_HEADS=25
else
        echo "Model size not supported."
        exit 1
fi

# Change for multinode config
NUM_WORKERS=16
NUM_GPUS_PER_WORKER=8
BATCHSIZE=$((MP_SIZE*TOTAL_BATCHSIZE/NUM_WORKERS/NUM_GPUS_PER_WORKER)) # per gpu batch size

DATA_PATH=/vc_data/Megatron-LM/data/indexed_datasets/megatron
VOCAB_PATH=/vc_data/Megatron-LM/data/gpt2-vocab.json
MERGE_PATH=/vc_data/Megatron-LM/data/gpt2-merges.txt

#ZeRO Configs
stage=2
reduce_scatter=true
contigious_gradients=true
rbs=50000000
agbs=5000000000

current_time=$(date "+%Y.%m.%d-%H.%M.%S")
script_path=$(realpath $0)
script_dir=$(dirname $script_path)
host="${HOSTNAME}"

if [ "${CONFIG_TEMPLATE}" = "true" ]; then
template_json="$script_dir/ds_zero_stage_${stage}_config_${CONFIG}.json"
config_json="$script_dir/ds_zero_stage_${stage}_config_${CONFIG}_min${CURRICULUM_MIN}_max${SEQ_LEN}_step${CURRICULUM_STEP}.json"
sed "s/CONFIG_CL_MIN/${CURRICULUM_MIN}/" ${template_json} \
    | sed "s/CONFIG_CL_MAX/${SEQ_LEN}/" \
    | sed "s/CONFIG_CL_DURATION/${CURRICULUM_STEP}/" \
	  > ${config_json}
else
config_json="$script_dir/ds_zero_stage_${stage}_config_${CONFIG}.json"
fi

JOB_NAME="gpt2_${MODEL_SIZE}M_bsz${TOTAL_BATCHSIZE}_seq${SEQ_LEN}_lr${LR}_warmup${LR_WARMUP_ITER}_decay${LR_DECAY_ITER}_seed${SEED}_${TAG}_stage${stage}_n${NUM_WORKERS}_g${NUM_GPUS_PER_WORKER}_mp${MP_SIZE}"
LOG_NAME="${JOB_NAME}_${host}_${current_time}"

#Actication Checkpointing and Contigious Memory
chkp_layers=1
PA=true
PA_CPU=false
CC=true
SYNCHRONIZE=true
PROFILE=false

OUTPUT_BASEPATH="/vc_data_blob/users/conglli"
mkdir -p "${OUTPUT_BASEPATH}/tensorboard/curriculum/"
mkdir -p "${OUTPUT_BASEPATH}/checkpoint/curriculum/"
mkdir -p "${OUTPUT_BASEPATH}/log/curriculum/"
LOGDIR="${OUTPUT_BASEPATH}/tensorboard/curriculum/${LOG_NAME}"
CHECKPOINT_PATH="${OUTPUT_BASEPATH}/checkpoint/curriculum/${JOB_NAME}"

gpt_options=" \
        --model-parallel-size ${MP_SIZE} \
        --num-layers $NUM_LAYERS \
        --hidden-size $HIDDEN_SIZE \
        --num-attention-heads $NUM_ATTN_HEADS \
        --seq-length $SEQ_LEN \
        --max-position-embeddings $SEQ_LEN \
        --batch-size $BATCHSIZE \
        --train-iters $NUM_ITER \
        --train-tokens $NUM_TOKEN \
        --lr-decay-iters $LR_DECAY_ITER \
        --save $CHECKPOINT_PATH \
        --load $CHECKPOINT_PATH \
        --data-path $DATA_PATH \
        --vocab-file $VOCAB_PATH \
        --merge-file $MERGE_PATH \
        --data-impl mmap \
        --split 949,50,1 \
        --distributed-backend nccl \
        --lr $LR \
        --lr-decay-style cosine \
        --min-lr 1.0e-5 \
        --weight-decay 1e-2 \
        --clip-grad 1.0 \
        --warmup-iters $LR_WARMUP_ITER \
        --checkpoint-activations \
        --log-interval 100 \
        --save-interval $SAVE_INTERVAL \
        --eval-interval 100 \
        --eval-iters 10 \
        --fp16 \
        --seed $SEED \
        --tensorboard-dir ${LOGDIR}
"

deepspeed_options=" \
        --deepspeed \
        --deepspeed_config ${config_json} \
        --zero-stage ${stage} \
        --zero-reduce-bucket-size ${rbs} \
        --zero-allgather-bucket-size ${agbs}
"

if [ "${contigious_gradients}" = "true" ]; then
deepspeed_options="${deepspeed_options} \
        --zero-contigious-gradients"
fi

if [ "${reduce_scatter}" = "true" ]; then
deepspeed_options="${deepspeed_options} \
        --zero-reduce-scatter"
fi

chkp_opt=" \
--deepspeed-activation-checkpointing \
--checkpoint-num-layers ${chkp_layers}"

if [ "${PA}" = "true" ]; then
chkp_opt="${chkp_opt} --partition-activations"
fi

if [ "${PA_CPU}" = "true" ]; then
chkp_opt="${chkp_opt} \
        --checkpoint-in-cpu"
fi

if [ "${SYNCHRONIZE}" = "true" ]; then
chkp_opt="${chkp_opt} \
        --synchronize-each-layer"
fi

if [ "${CC}" = "true" ]; then
chkp_opt="${chkp_opt} \
        --contigious-checkpointing"
fi

if [ "${PROFILE}" = "true" ]; then
chkp_opt="${chkp_opt} \
        --profile-backward"
fi


full_options="${gpt_options} ${deepspeed_options} ${chkp_opt}"

run_cmd="deepspeed --num_nodes ${NUM_WORKERS} --num_gpus ${NUM_GPUS_PER_WORKER}  ../pretrain_gpt2.py ${@:17} ${full_options} &>> ${OUTPUT_BASEPATH}/log/curriculum/${JOB_NAME}.log"
echo ${run_cmd}
eval ${run_cmd}

set +x
