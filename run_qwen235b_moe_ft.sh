#!/bin/bash

# Load MoE FT configuration
source ~/.moe_ft_config

# Qwen2-MoE Model Configuration (Actual values from converted model)
export LAYER_NUM=28
export HEAD_NUM=12
export SIZE_PER_HEAD=128       # 1536 hidden / 12 heads
export INTER_SIZE=8960
export VOCAB_SIZE=151936
export MAX_SEQ_LEN=131072

# MoE specific
export EXPERT_NUM=6
export MOE_K=4
# All layers are MoE layers
export MOE_LAYER_INDEX="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27"

# Parallelism configuration (2-way tensor parallelism)
export TENSOR_PARA_SIZE=2
export PIPELINE_PARA_SIZE=1
export PROMPT_WORLD_SIZE=2   # Prompt-phase GPUs
export TOKEN_WORLD_SIZE=2    # Token-phase GPUs

# Batch configuration
export UBATCH_SIZE=4
export NUM_UBATCHES=8
export MAX_BATCH_SIZE=32

# Generation parameters
export INPUT_LEN=512
export OUTPUT_LEN=512
export TEMPERATURE=0.7
export TOP_K=50
export TOP_P=0.95

# Paths
export CKPT_PATH="/data/models/qwen235b-moe-ft/2-gpu"
export VOCAB_FILE="/data/models/qwen235b-moe/vocab.json"
export MERGES_FILE="/data/models/qwen235b-moe/merges.txt"
export LIB_PATH="/root/dejavu1/build/lib/libth_transformer.so"

# NCCL configuration for MIG devices
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=0
export NCCL_NET_GDR_LEVEL=0
export NCCL_SOCKET_IFNAME=lo

# CUDA debugging
export CUDA_LAUNCH_BLOCKING=1

# FasterTransformer logging
export FT_LOG_LEVEL=INFO

# Run the model
cd /root/dejavu1/build

# Use GPU device ordinals (MIG instances show as device 0)
# When MIG is enabled, each MIG instance gets a device ID starting from 0
export CUDA_VISIBLE_DEVICES="0,1"
echo "Using CUDA devices: ${CUDA_VISIBLE_DEVICES}"

mpirun -n $((TENSOR_PARA_SIZE * PIPELINE_PARA_SIZE)) \
    --allow-run-as-root \
    -x ENABLE_MOE_TOKEN_FT \
    -x MOE_CHECKPOINT_INTERVAL \
    -x MOE_MAX_CHECKPOINTS \
    -x MOE_CHECKPOINT_POLICY \
    --bind-to none \
    -x CUDA_VISIBLE_DEVICES \
    python3 ../examples/pytorch/gpt/run_qwen235b_moe_ft.py \
        --layer_num $LAYER_NUM \
        --head_num $HEAD_NUM \
        --size_per_head $SIZE_PER_HEAD \
        --inter_size $INTER_SIZE \
        --vocab_size $VOCAB_SIZE \
        --max_seq_len $MAX_SEQ_LEN \
        --expert_num $EXPERT_NUM \
        --moe_k $MOE_K \
        --moe_layer_index $MOE_LAYER_INDEX \
        --tensor_para_size $TENSOR_PARA_SIZE \
        --pipeline_para_size $PIPELINE_PARA_SIZE \
        --prompt_world_size $PROMPT_WORLD_SIZE \
        --token_world_size $TOKEN_WORLD_SIZE \
        --ckpt_path $CKPT_PATH \
        --vocab_file $VOCAB_FILE \
        --merges_file $MERGES_FILE \
        --lib_path $LIB_PATH \
        --input_len $INPUT_LEN \
        --output_len $OUTPUT_LEN \
        --ubatch_size $UBATCH_SIZE \
        --num_ubatches $NUM_UBATCHES \
        --temperature $TEMPERATURE \
        --top_k $TOP_K \
        --top_p $TOP_P \
        --inference_data_type fp16 \
        --time
