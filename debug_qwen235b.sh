#!/bin/bash

# Load MoE FT configuration
source ~/.moe_ft_config

# Qwen2-MoE Model Configuration
export LAYER_NUM=28
export HEAD_NUM=12
export SIZE_PER_HEAD=128
export INTER_SIZE=8960
export VOCAB_SIZE=151936
export MAX_SEQ_LEN=131072

# MoE specific
export EXPERT_NUM=6
export MOE_K=4
export MOE_LAYER_INDEX="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27"

# Parallelism configuration (2-way tensor parallelism)
export TENSOR_PARA_SIZE=2
export PIPELINE_PARA_SIZE=1
export PROMPT_WORLD_SIZE=2
export TOKEN_WORLD_SIZE=2

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
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=NVL

# Use regular GPU devices instead of MIG for testing
export CUDA_VISIBLE_DEVICES="0"

# Run the model with single GPU for debugging
cd /root/dejavu1/build

echo "Running with single GPU for debugging..."
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# Run single rank with gdb to get backtrace on crash
gdb -batch -ex "run" -ex "bt" -ex "quit" --args \
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
        --tensor_para_size 1 \
        --pipeline_para_size 1 \
        --prompt_world_size 1 \
        --token_world_size 1 \
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
