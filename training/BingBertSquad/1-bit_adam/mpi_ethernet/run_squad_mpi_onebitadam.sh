# If you are able to install pytorch >= 1.8
# (and nccl >= 2.8.3 if you have 64 or more GPUs),
# we highly recommend you to use the NCCL-based 1-bit Adam
# which has better performance and ease of use
# (see scripts in DeepSpeedExamples/BingBertSquad/1-bit_adam/nccl
# and read the tutorial for more details:
# https://www.deepspeed.ai/tutorials/onebit-adam/)

NUM_NODES=4
NGPU_PER_NODE=8
MODEL_FILE="../../ckpt/bert-large-uncased-whole-word-masking-pytorch_model.bin"
ORIGIN_CONFIG_FILE="../../ckpt/bert-large-uncased-whole-word-masking-config.json"
SQUAD_DIR="../../data"
OUTPUT_DIR=$1
LR=3e-5
SEED=$RANDOM
MASTER_PORT=12345
DROPOUT=0.1

sudo rm -rf ${OUTPUT_DIR}

NGPU=$((NGPU_PER_NODE*NUM_NODES))
EFFECTIVE_BATCH_SIZE=96
MAX_GPU_BATCH_SIZE=3
PER_GPU_BATCH_SIZE=$((EFFECTIVE_BATCH_SIZE/NGPU))
if [[ $PER_GPU_BATCH_SIZE -lt $MAX_GPU_BATCH_SIZE ]]; then
       GRAD_ACCUM_STEPS=1
else
       GRAD_ACCUM_STEPS=$((PER_GPU_BATCH_SIZE/MAX_GPU_BATCH_SIZE))
fi
JOB_NAME="onebit_deepspeed_${NGPU}GPUs_${EFFECTIVE_BATCH_SIZE}batch_size"
config_json=deepspeed_onebitadam_bsz96_config.json

# NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=eth0 are used to disable infiniband. Remove it if needed.
mpirun -n $NGPU -npernode $NGPU_PER_NODE -hostfile /job/hostfile -x UCX_TLS=tcp --mca btl ^openib --mca btl_tcp_if_include eth0 -x NCCL_TREE_THRESHOLD=0 -x NCCL_IB_DISABLE=1 -x NCCL_SOCKET_IFNAME=eth0 python ../../nvidia_run_squad_deepspeed.py \
--bert_model bert-large-uncased \
--do_train \
--do_lower_case \
--predict_batch_size 3 \
--do_predict \
--train_file $SQUAD_DIR/train-v1.1.json \
--predict_file $SQUAD_DIR/dev-v1.1.json \
--train_batch_size $PER_GPU_BATCH_SIZE \
--learning_rate ${LR} \
--num_train_epochs 2.0 \
--max_seq_length 384 \
--doc_stride 128 \
--output_dir $OUTPUT_DIR \
--job_name ${JOB_NAME} \
--gradient_accumulation_steps ${GRAD_ACCUM_STEPS} \
--fp16 \
--deepspeed \
--deepspeed_mpi \
--deepspeed_transformer_kernel \
--deepspeed_config ${config_json} \
--dropout ${DROPOUT} \
--model_file $MODEL_FILE \
--seed ${SEED} \
--ckpt_type HF \
--origin_bert_config_file ${ORIGIN_CONFIG_FILE} \
