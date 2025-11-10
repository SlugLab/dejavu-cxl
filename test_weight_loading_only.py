#!/usr/bin/env python3
"""
Test weight loading only without creating C++ model
"""

import os
import sys
import torch

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, "examples/pytorch/gpt"))

import examples.pytorch.gpt.utils.gpt_token_encoder as encoder
from examples.pytorch.gpt.utils import comm
from examples.pytorch.gpt.utils.parallel_gpt_dv import ParallelGPT

# Configuration matching run_qwen235b_moe_ft.sh
config = {
    'layer_num': 48,
    'head_num': 32,
    'size_per_head': 64,
    'inter_size': 6144,
    'vocab_size': 151936,
    'max_seq_len': 2048,
    'expert_num': 128,
    'moe_k': 8,
    'moe_layer_index': list(range(48)),
    'tensor_para_size': 1,
    'pipeline_para_size': 1,
    'ckpt_path': '/root/Qwen3-30B-A3B-FT/1-gpu-nf4',
    'vocab_file': '/root/Qwen3-30B-A3B/vocab.json',
    'merges_file': '/root/Qwen3-30B-A3B/merges.txt',
}

print("Initializing model parallel...")
comm.initialize_model_parallel(config['tensor_para_size'], config['pipeline_para_size'])
rank = comm.get_rank()

print(f"\n[Rank {rank}] Loading GPT weights...")

# Try to instantiate GPT (which loads weights but doesn't call .cuda() yet)
try:
    gpt = ParallelGPT(
        head_num=config['head_num'],
        size_per_head=config['size_per_head'],
        vocab_size=config['vocab_size'],
        start_id=151643,  # Default from Qwen
        end_id=151643,
        layer_num=config['layer_num'],
        ckpt_path=config['ckpt_path'],
        max_seq_len=config['max_seq_len'],
        tensor_para_size=config['tensor_para_size'],
        pipeline_para_size=config['pipeline_para_size'],
        lib_path='/root/dejavu1/build/lib/libth_transformer.so',
        inference_data_type='fp16',
        int8_mode=0,
        weights_data_type='fp16',
        layernorm_eps=1e-6,
        layernorm_type='rmsnorm',
        activation_type='silu',
        has_positional_encoding=False,
        has_pre_decoder_layernorm=False,
        has_post_decoder_layernorm=True,
        has_adapters=False,
        adapter_inter_size=0,
        use_attention_linear_bias=False,
        inter_size=config['inter_size'],
        gpt_with_moe=True,
        expert_num=config['expert_num'],
        moe_k=config['moe_k'],
        moe_layer_index=config['moe_layer_index'],
        shared_contexts_ratio=1.0,
        prompt_world_size=1,
        token_world_size=1,
        restart=0
    )

    print(f"\n[Rank {rank}] ✓ GPT instance created successfully")
    print(f"  Number of weight tensors: {len(gpt.weights.w)}")
    print(f"  Total weight memory (approx): {sum([w.numel() * w.element_size() for w in gpt.weights.w if w.numel() > 0]) / 1e9:.2f} GB")

    # Don't call .cuda() - that's where the C++ model is created
    print(f"\n[Rank {rank}] Weights loaded successfully WITHOUT creating C++ model")
    print("SUCCESS: Weight loading works!")

except Exception as e:
    print(f"\n[Rank {rank}] ✗ Failed to create GPT instance")
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
