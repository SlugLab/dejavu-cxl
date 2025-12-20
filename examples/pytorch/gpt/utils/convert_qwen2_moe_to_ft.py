#!/usr/bin/env python3
"""
Convert Qwen2-MoE HuggingFace model to FasterTransformer format
"""

import argparse
import configparser
import json
import os
from pathlib import Path
import numpy as np
from safetensors import safe_open
import torch
from tqdm import tqdm

def get_weight_data_type(data_type):
    if data_type == "fp32":
        return np.float32
    elif data_type == "fp16":
        return np.float16
    else:
        raise ValueError(f"Invalid weight data type {data_type}")

def load_hf_qwen2_moe_model(model_path):
    """Load Qwen2-MoE model weights from safetensors"""
    model_path = Path(model_path)

    # Load config
    with open(model_path / "config.json", 'r') as f:
        config = json.load(f)

    # Load weight index
    with open(model_path / "model.safetensors.index.json", 'r') as f:
        index = json.load(f)

    # Load all weights
    weights = {}
    weight_files = set(index['weight_map'].values())

    print(f"Loading weights from {len(weight_files)} files...")
    for weight_file in tqdm(sorted(weight_files)):
        file_path = model_path / weight_file
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                weights[key] = f.get_tensor(key)

    return config, weights

def split_qkv_weights(q_weight, k_weight, v_weight, tensor_para_size):
    """Combine and split Q, K, V weights for tensor parallelism"""
    # Qwen2-MoE uses Grouped Query Attention (GQA)
    # Q: [hidden_size, num_heads * head_dim]
    # K, V: [hidden_size, num_kv_heads * head_dim]

    hidden_size = q_weight.shape[0]

    # For FT, we need to interleave Q, K, V
    # FT expects: [hidden_size, (num_heads + 2*num_kv_heads) * head_dim]
    # Split pattern: [Q_head0, K_head0, V_head0, Q_head1, K_head1, V_head1, ...]

    # This is simplified - just concatenate for now
    # TODO: Proper QKV interleaving for GQA
    qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=1)

    # Split across tensor parallel dimension
    splits = torch.chunk(qkv_weight, tensor_para_size, dim=1)
    return splits

def convert_qwen2_moe_to_ft(args):
    """Main conversion function"""

    # Create output directory
    saved_dir = Path(args.saved_dir) / f"{args.infer_gpu_num}-gpu"
    saved_dir.mkdir(parents=True, exist_ok=True)

    print(f"Converting Qwen2-MoE model from {args.in_file} to {saved_dir}")

    # Load HuggingFace model
    config, hf_weights = load_hf_qwen2_moe_model(args.in_file)

    # Extract model configuration
    num_layers = config['num_hidden_layers']
    hidden_size = config['hidden_size']
    num_attention_heads = config['num_attention_heads']
    num_kv_heads = config.get('num_key_value_heads', num_attention_heads)
    # Use explicit head_dim from config if available, otherwise calculate
    head_dim = config.get('head_dim', hidden_size // num_attention_heads)
    vocab_size = config['vocab_size']
    intermediate_size = config.get('intermediate_size', hidden_size * 4)
    num_experts = config.get('num_experts', 0)
    num_experts_per_tok = config.get('num_experts_per_tok', 0)
    moe_intermediate_size = config.get('moe_intermediate_size', intermediate_size)

    # GQA ratio: how many Q heads share each KV head
    gqa_ratio = num_attention_heads // num_kv_heads
    use_gqa = (num_kv_heads != num_attention_heads)

    print(f"\nModel config:")
    print(f"  Layers: {num_layers}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Attention heads: {num_attention_heads}")
    print(f"  KV heads: {num_kv_heads}")
    print(f"  Head dim: {head_dim}")
    print(f"  GQA enabled: {use_gqa} (ratio: {gqa_ratio}:1)")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Num experts: {num_experts}")
    print(f"  Experts per token: {num_experts_per_tok}")
    print(f"  MoE intermediate size: {moe_intermediate_size}")
    print(f"  Tensor parallel size: {args.infer_gpu_num}")

    # Save config.ini for FasterTransformer
    # For MoE models, use moe_intermediate_size; for dense models, use intermediate_size
    ft_inter_size = moe_intermediate_size if num_experts > 0 else intermediate_size

    ft_config = configparser.ConfigParser()
    ft_config['gpt'] = {
        'model_name': 'qwen2_moe',
        'head_num': str(num_attention_heads),
        'size_per_head': str(head_dim),
        'hidden_size': str(hidden_size),
        'inter_size': str(ft_inter_size),
        'num_layer': str(num_layers),
        'vocab_size': str(vocab_size),
        'start_id': str(config.get('bos_token_id', 151643)),
        'end_id': str(config.get('eos_token_id', 151643)),
        'max_pos_seq_len': str(config.get('max_position_embeddings', 32768)),
        'weight_data_type': args.weight_data_type,
        'tensor_para_size': str(args.infer_gpu_num),
        'layernorm_eps': str(config.get('rms_norm_eps', 1e-6)),
        'layernorm_type': 'pre_layernorm',  # Qwen uses RMSNorm before attention (pre-layernorm)
        'activation_type': 'SiGLU',  # Qwen uses SwiGLU/SiGLU activation
        'num_kv_heads': str(num_kv_heads),
    }

    # Add MoE structure if present
    if num_experts > 0:
        ft_config['structure'] = {
            'gpt_with_moe': 'True',
            'expert_num': str(num_experts),
            'moe_k': str(num_experts_per_tok),
            'moe_layers': str(list(range(num_layers))),  # All layers are MoE in this model
        }

    with open(saved_dir / "config.ini", 'w') as f:
        ft_config.write(f)

    print(f"\nSaved config to {saved_dir / 'config.ini'}")

    # Get data type
    np_weight_dtype = get_weight_data_type(args.weight_data_type)

    # Convert embedding weights
    print("\nConverting embeddings...")
    if 'model.embed_tokens.weight' in hf_weights:
        embed_weight = hf_weights['model.embed_tokens.weight'].float().numpy().astype(np_weight_dtype)
        embed_weight.tofile(saved_dir / "model.wte.bin")
        print(f"  Saved word embeddings: {embed_weight.shape}")

    # Convert final layer norm
    if 'model.norm.weight' in hf_weights:
        ln_weight = hf_weights['model.norm.weight'].float().numpy().astype(np_weight_dtype)
        ln_weight.tofile(saved_dir / "model.final_layernorm.weight.bin")
        print(f"  Saved final layernorm: {ln_weight.shape}")

    # Convert LM head - no transpose needed
    # CUBLAS GemmEx with CUBLAS_OP_T handles the transpose automatically
    # HuggingFace lm_head.weight is (vocab_size, hidden_size), save as-is
    if 'lm_head.weight' in hf_weights:
        lm_head = hf_weights['lm_head.weight']
        lm_head_np = lm_head.float().numpy().astype(np_weight_dtype)
        lm_head_np.tofile(saved_dir / "model.lm_head.weight.bin")
        print(f"  Saved LM head: {lm_head.shape}")

    # Convert layer weights
    print(f"\nConverting {num_layers} transformer layers...")
    tensor_para_size = args.infer_gpu_num

    for layer_idx in tqdm(range(num_layers)):
        prefix = f"model.layers.{layer_idx}"

        # 1. Input LayerNorm
        if f"{prefix}.input_layernorm.weight" in hf_weights:
            weight = hf_weights[f"{prefix}.input_layernorm.weight"].float().numpy().astype(np_weight_dtype)
            weight.tofile(saved_dir / f"model.layers.{layer_idx}.input_layernorm.weight.bin")

        # 2. Self-Attention QKV weights (with tensor parallelism)
        if (f"{prefix}.self_attn.q_proj.weight" in hf_weights and
            f"{prefix}.self_attn.k_proj.weight" in hf_weights and
            f"{prefix}.self_attn.v_proj.weight" in hf_weights):

            q_weight = hf_weights[f"{prefix}.self_attn.q_proj.weight"]
            k_weight = hf_weights[f"{prefix}.self_attn.k_proj.weight"]
            v_weight = hf_weights[f"{prefix}.self_attn.v_proj.weight"]

            # Handle GQA: expand K and V weights to match Q heads
            # Q shape: [num_attention_heads * head_dim, hidden_size]
            # K/V shape: [num_kv_heads * head_dim, hidden_size]
            # We need to expand K/V to [num_attention_heads * head_dim, hidden_size]
            if use_gqa and gqa_ratio > 1:
                # Reshape K to [num_kv_heads, head_dim, hidden_size]
                k_reshaped = k_weight.view(num_kv_heads, head_dim, hidden_size)
                # Repeat each KV head gqa_ratio times: [num_attention_heads, head_dim, hidden_size]
                k_expanded = k_reshaped.repeat_interleave(gqa_ratio, dim=0)
                # Reshape back to [num_attention_heads * head_dim, hidden_size]
                k_weight = k_expanded.view(num_attention_heads * head_dim, hidden_size)

                # Same for V
                v_reshaped = v_weight.view(num_kv_heads, head_dim, hidden_size)
                v_expanded = v_reshaped.repeat_interleave(gqa_ratio, dim=0)
                v_weight = v_expanded.view(num_attention_heads * head_dim, hidden_size)

                if layer_idx == 0:
                    print(f"  GQA: Expanded K/V from {num_kv_heads} to {num_attention_heads} heads")

            # Concatenate Q, K, V
            qkv_combined = torch.cat([q_weight, k_weight, v_weight], dim=0)

            # Split for tensor parallelism
            split_size = qkv_combined.shape[0] // tensor_para_size
            for tp_rank in range(tensor_para_size):
                start_idx = tp_rank * split_size
                end_idx = (tp_rank + 1) * split_size
                # Transpose to match FasterTransformer's column-major GEMM format
                # PyTorch row-major [qkv_dim, hidden_size] -> save as [hidden_size, qkv_dim] row-major
                # which CUBLAS reads as column-major [qkv_dim, hidden_size]
                qkv_split = qkv_combined[start_idx:end_idx, :].T.contiguous().float().numpy().astype(np_weight_dtype)
                qkv_split.tofile(saved_dir / f"model.layers.{layer_idx}.attention.query_key_value.weight.{tp_rank}.bin")

            # QKV biases if present
            if f"{prefix}.self_attn.q_proj.bias" in hf_weights:
                q_bias = hf_weights[f"{prefix}.self_attn.q_proj.bias"]
                k_bias = hf_weights[f"{prefix}.self_attn.k_proj.bias"]
                v_bias = hf_weights[f"{prefix}.self_attn.v_proj.bias"]

                # Expand K/V biases for GQA
                if use_gqa and gqa_ratio > 1:
                    # Reshape to [num_kv_heads, head_dim]
                    k_bias_reshaped = k_bias.view(num_kv_heads, head_dim)
                    k_bias_expanded = k_bias_reshaped.repeat_interleave(gqa_ratio, dim=0)
                    k_bias = k_bias_expanded.view(num_attention_heads * head_dim)

                    v_bias_reshaped = v_bias.view(num_kv_heads, head_dim)
                    v_bias_expanded = v_bias_reshaped.repeat_interleave(gqa_ratio, dim=0)
                    v_bias = v_bias_expanded.view(num_attention_heads * head_dim)

                qkv_bias_combined = torch.cat([q_bias, k_bias, v_bias], dim=0)

                split_size = qkv_bias_combined.shape[0] // tensor_para_size
                for tp_rank in range(tensor_para_size):
                    start_idx = tp_rank * split_size
                    end_idx = (tp_rank + 1) * split_size
                    bias_split = qkv_bias_combined[start_idx:end_idx].float().numpy().astype(np_weight_dtype)
                    bias_split.tofile(saved_dir / f"model.layers.{layer_idx}.attention.query_key_value.bias.{tp_rank}.bin")

        # 3. Attention output projection (split input dimension)
        if f"{prefix}.self_attn.o_proj.weight" in hf_weights:
            o_proj_weight = hf_weights[f"{prefix}.self_attn.o_proj.weight"]

            # Split input dimension for tensor parallelism
            split_size = o_proj_weight.shape[1] // tensor_para_size
            for tp_rank in range(tensor_para_size):
                start_idx = tp_rank * split_size
                end_idx = (tp_rank + 1) * split_size
                # Transpose for CUBLAS column-major format
                o_proj_split = o_proj_weight[:, start_idx:end_idx].T.contiguous().float().numpy().astype(np_weight_dtype)
                o_proj_split.tofile(saved_dir / f"model.layers.{layer_idx}.attention.dense.weight.{tp_rank}.bin")

        # 4. Post-attention LayerNorm
        if f"{prefix}.post_attention_layernorm.weight" in hf_weights:
            weight = hf_weights[f"{prefix}.post_attention_layernorm.weight"].float().numpy().astype(np_weight_dtype)
            weight.tofile(saved_dir / f"model.layers.{layer_idx}.post_attention_layernorm.weight.bin")

        # 5. MoE expert weights
        if num_experts > 0:
            # Gate (router) weights
            if f"{prefix}.mlp.gate.weight" in hf_weights:
                gate_weight = hf_weights[f"{prefix}.mlp.gate.weight"].float().numpy().astype(np_weight_dtype)
                gate_weight.tofile(saved_dir / f"model.layers.{layer_idx}.mlp.gate.weight.bin")

            # Process each expert
            for expert_idx in range(num_experts):
                expert_prefix = f"{prefix}.mlp.experts.{expert_idx}"

                # Expert gate projection (up_proj)
                if f"{expert_prefix}.gate_proj.weight" in hf_weights:
                    gate_proj = hf_weights[f"{expert_prefix}.gate_proj.weight"]
                    # Split output dimension
                    split_size = gate_proj.shape[0] // tensor_para_size
                    for tp_rank in range(tensor_para_size):
                        start_idx = tp_rank * split_size
                        end_idx = (tp_rank + 1) * split_size
                        # Transpose for CUBLAS column-major format
                        weight_split = gate_proj[start_idx:end_idx, :].T.contiguous().float().numpy().astype(np_weight_dtype)
                        weight_split.tofile(saved_dir / f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.weight.{tp_rank}.bin")

                # Expert up projection
                if f"{expert_prefix}.up_proj.weight" in hf_weights:
                    up_proj = hf_weights[f"{expert_prefix}.up_proj.weight"]
                    split_size = up_proj.shape[0] // tensor_para_size
                    for tp_rank in range(tensor_para_size):
                        start_idx = tp_rank * split_size
                        end_idx = (tp_rank + 1) * split_size
                        # Transpose for CUBLAS column-major format
                        weight_split = up_proj[start_idx:end_idx, :].T.contiguous().float().numpy().astype(np_weight_dtype)
                        weight_split.tofile(saved_dir / f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.weight.{tp_rank}.bin")

                # Expert down projection
                if f"{expert_prefix}.down_proj.weight" in hf_weights:
                    down_proj = hf_weights[f"{expert_prefix}.down_proj.weight"]
                    # Split input dimension
                    split_size = down_proj.shape[1] // tensor_para_size
                    for tp_rank in range(tensor_para_size):
                        start_idx = tp_rank * split_size
                        end_idx = (tp_rank + 1) * split_size
                        # Transpose for CUBLAS column-major format
                        weight_split = down_proj[:, start_idx:end_idx].T.contiguous().float().numpy().astype(np_weight_dtype)
                        weight_split.tofile(saved_dir / f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.weight.{tp_rank}.bin")

            # Shared expert (if present)
            shared_prefix = f"{prefix}.mlp.shared_expert"
            if f"{shared_prefix}.gate_proj.weight" in hf_weights:
                # Shared expert gate projection
                gate_proj = hf_weights[f"{shared_prefix}.gate_proj.weight"]
                split_size = gate_proj.shape[0] // tensor_para_size
                for tp_rank in range(tensor_para_size):
                    start_idx = tp_rank * split_size
                    end_idx = (tp_rank + 1) * split_size
                    # Transpose for CUBLAS column-major format
                    weight_split = gate_proj[start_idx:end_idx, :].T.contiguous().float().numpy().astype(np_weight_dtype)
                    weight_split.tofile(saved_dir / f"model.layers.{layer_idx}.mlp.shared_expert.gate_proj.weight.{tp_rank}.bin")

                # Shared expert up projection
                up_proj = hf_weights[f"{shared_prefix}.up_proj.weight"]
                split_size = up_proj.shape[0] // tensor_para_size
                for tp_rank in range(tensor_para_size):
                    start_idx = tp_rank * split_size
                    end_idx = (tp_rank + 1) * split_size
                    # Transpose for CUBLAS column-major format
                    weight_split = up_proj[start_idx:end_idx, :].T.contiguous().float().numpy().astype(np_weight_dtype)
                    weight_split.tofile(saved_dir / f"model.layers.{layer_idx}.mlp.shared_expert.up_proj.weight.{tp_rank}.bin")

                # Shared expert down projection
                down_proj = hf_weights[f"{shared_prefix}.down_proj.weight"]
                split_size = down_proj.shape[1] // tensor_para_size
                for tp_rank in range(tensor_para_size):
                    start_idx = tp_rank * split_size
                    end_idx = (tp_rank + 1) * split_size
                    # Transpose for CUBLAS column-major format
                    weight_split = down_proj[:, start_idx:end_idx].T.contiguous().float().numpy().astype(np_weight_dtype)
                    weight_split.tofile(saved_dir / f"model.layers.{layer_idx}.mlp.shared_expert.down_proj.weight.{tp_rank}.bin")

    print(f"\n✅ Conversion complete! Weights saved to {saved_dir}")
    print(f"Total files created: {len(list(saved_dir.glob('*.bin')))}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-saved_dir', '-o', type=str, required=True,
                        help='Output directory for FasterTransformer weights')
    parser.add_argument('-in_file', '-i', type=str, required=True,
                        help='Path to HuggingFace model directory')
    parser.add_argument('-infer_gpu_num', '-i_g', type=int, required=True,
                        help='Number of GPUs for tensor parallelism')
    parser.add_argument("-weight_data_type", type=str, default="fp16",
                        choices=["fp32", "fp16"], help="Weight data type")

    args = parser.parse_args()

    print("\n" + "="*80)
    print("Qwen2-MoE to FasterTransformer Conversion")
    print("="*80)
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print("="*80 + "\n")

    convert_qwen2_moe_to_ft(args)
