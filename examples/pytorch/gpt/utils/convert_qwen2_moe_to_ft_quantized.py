#!/usr/bin/env python3
"""
Convert Qwen2-MoE HuggingFace model to FasterTransformer format with FP4/NF4 quantization
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
    """Map data type string to numpy dtype"""
    if data_type == "fp32":
        return np.float32
    elif data_type == "fp16":
        return np.float16
    elif data_type in ["fp4", "nf4"]:
        return np.uint8  # We'll store quantized as uint8
    else:
        raise ValueError(f"Invalid weight data type {data_type}")

def quantize_to_fp4(tensor, quant_type="nf4"):
    """
    Quantize tensor to 4-bit using symmetric quantization or NF4

    Args:
        tensor: torch.Tensor to quantize
        quant_type: "fp4" for symmetric or "nf4" for normal float 4

    Returns:
        quantized: uint8 tensor (2 values packed per byte)
        scale: per-channel scale factors
    """
    if quant_type == "nf4":
        # NF4 quantization levels (optimized for normal distribution)
        # These are the 16 levels for NF4
        nf4_levels = torch.tensor([
            -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
            -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
            0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
            0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0
        ], device=tensor.device, dtype=tensor.dtype)
    else:
        # Symmetric FP4: 16 evenly spaced levels from -1 to 1
        nf4_levels = torch.linspace(-1, 1, 16, device=tensor.device, dtype=tensor.dtype)

    # Flatten and get absolute max per output channel (first dimension)
    original_shape = tensor.shape
    if len(original_shape) > 1:
        # For 2D tensors (weight matrices), quantize per output channel
        tensor_2d = tensor.reshape(original_shape[0], -1)
        abs_max = tensor_2d.abs().max(dim=1, keepdim=True)[0]
        abs_max = abs_max.clamp(min=1e-5)  # Avoid division by zero

        # Normalize to [-1, 1]
        normalized = tensor_2d / abs_max

        # Find nearest quantization level for each element
        distances = (normalized.unsqueeze(-1) - nf4_levels.reshape(1, 1, -1)).abs()
        indices = distances.argmin(dim=-1)  # Shape: [out_channels, in_features]

        # Pack two 4-bit values into one uint8
        # Reshape to ensure we can pack pairs
        flat_indices = indices.flatten()
        if flat_indices.shape[0] % 2 != 0:
            # Pad with zeros if odd number of elements
            flat_indices = torch.cat([flat_indices, torch.zeros(1, dtype=flat_indices.dtype, device=flat_indices.device)])

        # Pack: high nibble = even index, low nibble = odd index
        packed = (flat_indices[::2] << 4) | flat_indices[1::2]

        # Store scale factors (per output channel)
        scales = abs_max.squeeze().cpu().numpy().astype(np.float16)

        return packed.cpu().numpy().astype(np.uint8), scales
    else:
        # For 1D tensors (biases, norms), use simple scalar quantization
        abs_max = tensor.abs().max()
        abs_max = abs_max.clamp(min=1e-5)
        normalized = tensor / abs_max

        distances = (normalized.unsqueeze(-1) - nf4_levels.reshape(1, -1)).abs()
        indices = distances.argmin(dim=-1)

        flat_indices = indices.flatten()
        if flat_indices.shape[0] % 2 != 0:
            flat_indices = torch.cat([flat_indices, torch.zeros(1, dtype=flat_indices.dtype, device=flat_indices.device)])

        packed = (flat_indices[::2] << 4) | flat_indices[1::2]
        scales = np.array([abs_max.cpu().item()], dtype=np.float16)

        return packed.cpu().numpy().astype(np.uint8), scales

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

def convert_qwen2_moe_to_ft_quantized(args):
    """Main conversion function with quantization"""

    # Create output directory
    saved_dir = Path(args.saved_dir) / f"{args.infer_gpu_num}-gpu-{args.quant_type}"
    saved_dir.mkdir(parents=True, exist_ok=True)

    print(f"Converting Qwen2-MoE model from {args.in_file} to {saved_dir}")
    print(f"Quantization: {args.quant_type}")

    # Load HuggingFace model
    config, hf_weights = load_hf_qwen2_moe_model(args.in_file)

    # Extract model configuration
    num_layers = config['num_hidden_layers']
    hidden_size = config['hidden_size']
    num_attention_heads = config['num_attention_heads']
    num_kv_heads = config['num_key_value_heads']
    head_dim = hidden_size // num_attention_heads
    vocab_size = config['vocab_size']
    intermediate_size = config.get('intermediate_size', hidden_size * 4)
    num_experts = config.get('num_experts', 0)
    num_experts_per_tok = config.get('num_experts_per_tok', 0)
    moe_intermediate_size = config.get('moe_intermediate_size', intermediate_size)

    print(f"\nModel config:")
    print(f"  Layers: {num_layers}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Attention heads: {num_attention_heads}")
    print(f"  KV heads: {num_kv_heads}")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Num experts: {num_experts}")
    print(f"  Experts per token: {num_experts_per_tok}")
    print(f"  Tensor parallel size: {args.infer_gpu_num}")

    # Save config.ini for FasterTransformer
    ft_config = configparser.ConfigParser()
    ft_config['gpt'] = {
        'model_name': 'qwen2_moe',
        'head_num': str(num_attention_heads),
        'size_per_head': str(head_dim),
        'inter_size': str(intermediate_size),
        'num_layer': str(num_layers),
        'vocab_size': str(vocab_size),
        'start_id': str(config.get('bos_token_id', 151643)),
        'end_id': str(config.get('eos_token_id', 151645)),
        'max_pos_seq_len': str(config.get('max_position_embeddings', 32768)),
        'weight_data_type': args.quant_type,
        'tensor_para_size': str(args.infer_gpu_num),
        'layernorm_eps': str(config.get('rms_norm_eps', 1e-6)),
    }

    # Add MoE structure if present
    if num_experts > 0:
        ft_config['structure'] = {
            'gpt_with_moe': 'True',
            'expert_num': str(num_experts),
            'moe_k': str(num_experts_per_tok),
            'moe_layers': str(list(range(num_layers))),
        }

    with open(saved_dir / "config.ini", 'w') as f:
        ft_config.write(f)

    print(f"\nSaved config to {saved_dir / 'config.ini'}")

    # Get data type
    np_weight_dtype = get_weight_data_type(args.quant_type)

    # Determine which weights to quantize
    quantize_weights = args.quant_type in ["fp4", "nf4"]

    # Convert embedding weights (don't quantize embeddings - keep fp16)
    print("\nConverting embeddings (keeping fp16)...")
    if 'model.embed_tokens.weight' in hf_weights:
        embed_weight = hf_weights['model.embed_tokens.weight'].float().numpy().astype(np.float16)
        embed_weight.tofile(saved_dir / "model.wte.bin")
        print(f"  Saved word embeddings: {embed_weight.shape}")

    # Convert final layer norm (don't quantize - keep fp16)
    if 'model.norm.weight' in hf_weights:
        ln_weight = hf_weights['model.norm.weight'].float().numpy().astype(np.float16)
        ln_weight.tofile(saved_dir / "model.final_layernorm.weight.bin")
        print(f"  Saved final layernorm: {ln_weight.shape}")

    # Convert LM head (don't quantize - keep fp16)
    if 'lm_head.weight' in hf_weights:
        lm_head = hf_weights['lm_head.weight'].float().numpy().astype(np.float16)
        lm_head.tofile(saved_dir / "model.lm_head.weight.bin")
        print(f"  Saved LM head: {lm_head.shape}")

    # Convert layer weights
    print(f"\nConverting {num_layers} transformer layers with {args.quant_type} quantization...")
    tensor_para_size = args.infer_gpu_num

    for layer_idx in tqdm(range(num_layers)):
        prefix = f"model.layers.{layer_idx}"

        # 1. Input LayerNorm (don't quantize)
        if f"{prefix}.input_layernorm.weight" in hf_weights:
            weight = hf_weights[f"{prefix}.input_layernorm.weight"].float().numpy().astype(np.float16)
            weight.tofile(saved_dir / f"model.layers.{layer_idx}.input_layernorm.weight.bin")

        # 2. Self-Attention QKV weights (quantize if enabled)
        if (f"{prefix}.self_attn.q_proj.weight" in hf_weights and
            f"{prefix}.self_attn.k_proj.weight" in hf_weights and
            f"{prefix}.self_attn.v_proj.weight" in hf_weights):

            q_weight = hf_weights[f"{prefix}.self_attn.q_proj.weight"].float()
            k_weight = hf_weights[f"{prefix}.self_attn.k_proj.weight"].float()
            v_weight = hf_weights[f"{prefix}.self_attn.v_proj.weight"].float()

            qkv_combined = torch.cat([q_weight, k_weight, v_weight], dim=0)

            # Split for tensor parallelism
            split_size = qkv_combined.shape[0] // tensor_para_size
            for tp_rank in range(tensor_para_size):
                start_idx = tp_rank * split_size
                end_idx = (tp_rank + 1) * split_size
                qkv_split = qkv_combined[start_idx:end_idx, :]

                if quantize_weights:
                    # Quantize to FP4/NF4
                    qkv_quant, qkv_scales = quantize_to_fp4(qkv_split, args.quant_type)
                    qkv_quant.tofile(saved_dir / f"model.layers.{layer_idx}.attention.query_key_value.weight.{tp_rank}.bin")
                    qkv_scales.tofile(saved_dir / f"model.layers.{layer_idx}.attention.query_key_value.weight.{tp_rank}.scales.bin")
                else:
                    qkv_split = qkv_split.numpy().astype(np.float16)
                    qkv_split.tofile(saved_dir / f"model.layers.{layer_idx}.attention.query_key_value.weight.{tp_rank}.bin")

        # 3. Attention output projection (quantize if enabled)
        if f"{prefix}.self_attn.o_proj.weight" in hf_weights:
            o_proj_weight = hf_weights[f"{prefix}.self_attn.o_proj.weight"].float()

            split_size = o_proj_weight.shape[1] // tensor_para_size
            for tp_rank in range(tensor_para_size):
                start_idx = tp_rank * split_size
                end_idx = (tp_rank + 1) * split_size
                o_proj_split = o_proj_weight[:, start_idx:end_idx]

                if quantize_weights:
                    o_proj_quant, o_proj_scales = quantize_to_fp4(o_proj_split, args.quant_type)
                    o_proj_quant.tofile(saved_dir / f"model.layers.{layer_idx}.attention.dense.weight.{tp_rank}.bin")
                    o_proj_scales.tofile(saved_dir / f"model.layers.{layer_idx}.attention.dense.weight.{tp_rank}.scales.bin")
                else:
                    o_proj_split = o_proj_split.numpy().astype(np.float16)
                    o_proj_split.tofile(saved_dir / f"model.layers.{layer_idx}.attention.dense.weight.{tp_rank}.bin")

        # 4. Post-attention LayerNorm (don't quantize)
        if f"{prefix}.post_attention_layernorm.weight" in hf_weights:
            weight = hf_weights[f"{prefix}.post_attention_layernorm.weight"].float().numpy().astype(np.float16)
            weight.tofile(saved_dir / f"model.layers.{layer_idx}.post_attention_layernorm.weight.bin")

        # 5. MoE expert weights (quantize if enabled - this is where we save the most memory!)
        if num_experts > 0:
            # Gate (router) weights (don't quantize - needs precision)
            if f"{prefix}.mlp.gate.weight" in hf_weights:
                gate_weight = hf_weights[f"{prefix}.mlp.gate.weight"].float().numpy().astype(np.float16)
                gate_weight.tofile(saved_dir / f"model.layers.{layer_idx}.mlp.gate.weight.bin")

            # Process each expert
            for expert_idx in range(num_experts):
                expert_prefix = f"{prefix}.mlp.experts.{expert_idx}"

                # Expert gate projection
                if f"{expert_prefix}.gate_proj.weight" in hf_weights:
                    gate_proj = hf_weights[f"{expert_prefix}.gate_proj.weight"].float()
                    split_size = gate_proj.shape[0] // tensor_para_size
                    for tp_rank in range(tensor_para_size):
                        start_idx = tp_rank * split_size
                        end_idx = (tp_rank + 1) * split_size
                        weight_split = gate_proj[start_idx:end_idx, :]

                        if quantize_weights:
                            weight_quant, weight_scales = quantize_to_fp4(weight_split, args.quant_type)
                            weight_quant.tofile(saved_dir / f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.weight.{tp_rank}.bin")
                            weight_scales.tofile(saved_dir / f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.weight.{tp_rank}.scales.bin")
                        else:
                            weight_split = weight_split.numpy().astype(np.float16)
                            weight_split.tofile(saved_dir / f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.weight.{tp_rank}.bin")

                # Expert up projection
                if f"{expert_prefix}.up_proj.weight" in hf_weights:
                    up_proj = hf_weights[f"{expert_prefix}.up_proj.weight"].float()
                    split_size = up_proj.shape[0] // tensor_para_size
                    for tp_rank in range(tensor_para_size):
                        start_idx = tp_rank * split_size
                        end_idx = (tp_rank + 1) * split_size
                        weight_split = up_proj[start_idx:end_idx, :]

                        if quantize_weights:
                            weight_quant, weight_scales = quantize_to_fp4(weight_split, args.quant_type)
                            weight_quant.tofile(saved_dir / f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.weight.{tp_rank}.bin")
                            weight_scales.tofile(saved_dir / f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.weight.{tp_rank}.scales.bin")
                        else:
                            weight_split = weight_split.numpy().astype(np.float16)
                            weight_split.tofile(saved_dir / f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.weight.{tp_rank}.bin")

                # Expert down projection
                if f"{expert_prefix}.down_proj.weight" in hf_weights:
                    down_proj = hf_weights[f"{expert_prefix}.down_proj.weight"].float()
                    split_size = down_proj.shape[1] // tensor_para_size
                    for tp_rank in range(tensor_para_size):
                        start_idx = tp_rank * split_size
                        end_idx = (tp_rank + 1) * split_size
                        weight_split = down_proj[:, start_idx:end_idx]

                        if quantize_weights:
                            weight_quant, weight_scales = quantize_to_fp4(weight_split, args.quant_type)
                            weight_quant.tofile(saved_dir / f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.weight.{tp_rank}.bin")
                            weight_scales.tofile(saved_dir / f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.weight.{tp_rank}.scales.bin")
                        else:
                            weight_split = weight_split.numpy().astype(np.float16)
                            weight_split.tofile(saved_dir / f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.weight.{tp_rank}.bin")

    print(f"\n✅ Conversion complete! Weights saved to {saved_dir}")
    all_files = list(saved_dir.glob('*.bin'))
    print(f"Total files created: {len(all_files)}")

    # Calculate size savings
    total_size = sum(f.stat().st_size for f in all_files)
    print(f"Total size: {total_size / 1e9:.2f} GB")
    if quantize_weights:
        print(f"Expected memory savings: ~4x compared to fp16")
        print(f"Estimated runtime memory: ~{total_size * 1.5 / 1e9:.2f} GB (including activations)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-saved_dir', '-o', type=str, required=True,
                        help='Output directory for FasterTransformer weights')
    parser.add_argument('-in_file', '-i', type=str, required=True,
                        help='Path to HuggingFace model directory')
    parser.add_argument('-infer_gpu_num', '-i_g', type=int, required=True,
                        help='Number of GPUs for tensor parallelism')
    parser.add_argument("-quant_type", type=str, default="nf4",
                        choices=["fp16", "fp4", "nf4"],
                        help="Weight quantization type: fp16 (no quant), fp4 (symmetric), nf4 (normal float 4)")

    args = parser.parse_args()

    print("\n" + "="*80)
    print("Qwen2-MoE to FasterTransformer Conversion with Quantization")
    print("="*80)
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print("="*80 + "\n")

    convert_qwen2_moe_to_ft_quantized(args)
