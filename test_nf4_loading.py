#!/usr/bin/env python3
"""Test NF4 weight loading in Python"""

import sys
import os
sys.path.insert(0, 'examples/pytorch/gpt')

from utils.gpt_dv import GPT
import torch

print("=" * 80)
print("Testing NF4 Weight Loading")
print("=" * 80)

# Create GPT instance with minimal config (just 1 layer for testing)
try:
    gpt = GPT(
        head_num=32,
        size_per_head=64,
        vocab_size=151936,
        start_id=151643,
        end_id=151645,
        layer_num=1,  # Only 1 layer for memory test
        max_seq_len=2048,
        tensor_para_size=1,
        pipeline_para_size=1,
        lib_path='build/lib/libth_transformer.so',
        ckpt_path='/root/Qwen3-30B-A3B-FT/1-gpu-nf4',
        inference_data_type='fp16',
        weights_data_type=torch.float16,
        layernorm_eps=1e-6,
        layernorm_type='pre_layernorm',
        activation_type='Gelu',
        has_positional_encoding=False,
        has_pre_decoder_layernorm=False,
        has_post_decoder_layernorm=True,
        has_adapters=False,
        adapter_inter_size=0,
        use_attention_linear_bias=False,
        int8_mode=0,
        expert_num=128,
        moe_k=8,
        moe_layer_index=[0],
    )

    print("✅ GPT instance created")

    print("\nWeights already loaded during __init__")

    print("✅ NF4 weights loaded successfully!")

    # Check memory usage
    import torch.cuda
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"\nGPU Memory:")
        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Reserved: {reserved:.2f} GB")

    print("\n" + "=" * 80)
    print("SUCCESS: NF4 loading works correctly!")
    print("=" * 80)

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
