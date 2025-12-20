#!/usr/bin/env python3
"""Simple test to identify where the code gets stuck"""

import os
import sys
import torch

print("Step 1: Imports starting...")
sys.stdout.flush()

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, "examples/pytorch/gpt"))

print("Step 2: Loading modules...")
sys.stdout.flush()

# Skip the comm module since it hangs without MPI
# from examples.pytorch.gpt.utils import comm
from examples.pytorch.gpt.utils.parallel_gpt_dv import ParallelGPT

print("Step 3: ParallelGPT loaded")
sys.stdout.flush()

config = {
    'layer_num': 48,
    'head_num': 32,
    'size_per_head': 128,
    'hidden_size': 2048,
    'inter_size': 6144,
    'vocab_size': 151936,
    'max_seq_len': 2048,
    'expert_num': 128,
    'moe_k': 8,
    'moe_layer_index': list(range(48)),
    'tensor_para_size': 1,
    'pipeline_para_size': 1,
    'ckpt_path': '/home/victoryang00/Qwen3-30B-A3B-FT/1-gpu',
    'lib_path': os.path.join(dir_path, 'build/lib/libth_transformer.so'),
}

print("Step 4: Creating model (skipping model parallel init)...")
sys.stdout.flush()

# Set up basic environment for single-GPU
torch.cuda.set_device(0)

gpt = ParallelGPT(
    head_num=config['head_num'],
    size_per_head=config['size_per_head'],
    vocab_size=config['vocab_size'],
    start_id=151643,
    end_id=151643,
    layer_num=config['layer_num'],
    ckpt_path=config['ckpt_path'],
    max_seq_len=config['max_seq_len'],
    tensor_para_size=1,
    pipeline_para_size=1,
    lib_path=config['lib_path'],
    inference_data_type='fp16',
    int8_mode=0,
    weights_data_type='fp16',
    layernorm_eps=1e-6,
    layernorm_type='pre_layernorm',
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
    torch_rank=0,
    restart=False,
    hidden_size=config['hidden_size'],
    num_kv_heads=4,
)

print("Step 5: Model created, checking weights...")
sys.stdout.flush()

print(f"  Weights count: {len(gpt.weights.w)}")
for i in [0, 96]:
    if i < len(gpt.weights.w):
        w = gpt.weights.w[i]
        print(f"  w[{i}]: numel={w.numel()}, device={w.device}")
sys.stdout.flush()

print("Step 6: Calling cuda()...")
sys.stdout.flush()

gpt.cuda()

print("Step 7: cuda() completed!")
sys.stdout.flush()

print("Step 8: Running inference test...")
sys.stdout.flush()

# Simple inference test
# The forward function expects lists of tensors for batched inference
input_ids = [torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.int32).cuda()]
input_lengths = [torch.tensor([6], dtype=torch.int32).cuda()]
output_len = torch.tensor([2], dtype=torch.int32)  # Should be a tensor

print(f"  input_ids[0] shape: {input_ids[0].shape}")
print(f"  input_lengths[0]: {input_lengths[0]}")
sys.stdout.flush()

with torch.no_grad():
    outputs = gpt(input_ids, input_lengths, output_len)

print(f"Step 9: Inference completed!")
print(f"  Output: {outputs}")
sys.stdout.flush()

print("SUCCESS!")
