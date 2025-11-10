#!/bin/bash

# Load MoE FT configuration
source ~/.moe_ft_config

# Reduce memory overhead for MoE Token FT
export MOE_DEVICE_POOL_SIZE=128  # Reduce from 512MB to 128MB
export MOE_HOST_BUFFER_SIZE=64   # Reduce from 128MB to 64MB

# PyTorch memory management
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,expandable_segments:True
export CUDA_MODULE_LOADING=LAZY

# Qwen3-30B-A3B Model Configuration
export LAYER_NUM=48
export HEAD_NUM=32
export SIZE_PER_HEAD=64        # 2048 hidden / 32 heads
export INTER_SIZE=6144
export VOCAB_SIZE=151936
export MAX_SEQ_LEN=2048  # Reduce from 40960 to minimize KV cache memory

# MoE specific
export EXPERT_NUM=128
export MOE_K=8
# All layers are MoE layers
export MOE_LAYER_INDEX="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47"

# Parallelism configuration (1-GPU)
export TENSOR_PARA_SIZE=1
export PIPELINE_PARA_SIZE=1
export PROMPT_WORLD_SIZE=1   # Prompt-phase GPUs
export TOKEN_WORLD_SIZE=1    # Token-phase GPUs

# Batch configuration (REDUCED FOR DEBUGGING)
export UBATCH_SIZE=1
export NUM_UBATCHES=1
export MAX_BATCH_SIZE=1

# Generation parameters
export INPUT_LEN=32
export OUTPUT_LEN=32
export TEMPERATURE=0.7
export TOP_K=50
export TOP_P=0.95

# Paths - Using NF4 quantized model
export CKPT_PATH="/root/Qwen3-30B-A3B-FT/1-gpu-nf4"
export VOCAB_FILE="/root/Qwen3-30B-A3B/vocab.json"
export MERGES_FILE="/root/Qwen3-30B-A3B/merges.txt"
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

# Use single GPU
export CUDA_VISIBLE_DEVICES="0"
echo "Using CUDA devices: ${CUDA_VISIBLE_DEVICES}"
# which python3
mpirun -n $((TENSOR_PARA_SIZE * PIPELINE_PARA_SIZE)) \
    --allow-run-as-root \
    -x ENABLE_MOE_TOKEN_FT \
    -x MOE_CHECKPOINT_INTERVAL \
    -x MOE_MAX_CHECKPOINTS \
    -x MOE_CHECKPOINT_POLICY \
    --bind-to none \
    -x CUDA_VISIBLE_DEVICES \
    /opt/miniconda3/bin/python3 ../examples/pytorch/gpt/run_qwen235b_moe_ft.py \
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
