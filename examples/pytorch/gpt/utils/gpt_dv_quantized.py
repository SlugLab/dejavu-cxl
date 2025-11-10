"""
Modified GPT loader that supports loading NF4 quantized weights and dequantizing them
"""

import os
import numpy as np
import torch
from .gpt_dv import GPT


def dequantize_nf4(quantized_data, scales, original_shape):
    """
    Dequantize NF4 format back to FP16

    Args:
        quantized_data: uint8 numpy array with packed 4-bit values
        scales: FP16 scale factors
        original_shape: tuple of original tensor shape

    Returns:
        torch.Tensor in FP16
    """
    # NF4 quantization levels
    nf4_levels = np.array([
        -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
        -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
        0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
        0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0
    ], dtype=np.float16)

    # Unpack 4-bit values
    high_nibble = (quantized_data >> 4) & 0x0F
    low_nibble = quantized_data & 0x0F

    # Interleave to get original indices
    indices = np.empty(len(quantized_data) * 2, dtype=np.uint8)
    indices[::2] = high_nibble
    indices[1::2] = low_nibble

    # Remove padding if added
    total_elements = np.prod(original_shape)
    indices = indices[:total_elements]

    # Map indices to NF4 levels
    dequantized = nf4_levels[indices]

    # Reshape to 2D for scaling
    if len(original_shape) > 1:
        dequantized_2d = dequantized.reshape(original_shape[0], -1)
        # Apply per-channel scales
        if len(scales) == original_shape[0]:
            scales_expanded = scales.reshape(-1, 1)
            dequantized_2d = dequantized_2d * scales_expanded
        else:
            dequantized_2d = dequantized_2d * scales[0]
        result = dequantized_2d.reshape(original_shape)
    else:
        result = dequantized * scales[0]
        result = result.reshape(original_shape)

    return torch.from_numpy(result.astype(np.float16))


class QuantizedGPT(GPT):
    """GPT model loader with NF4 quantization support"""

    def __init__(self, *args, **kwargs):
        # Check if weights are quantized
        ckpt_path = kwargs.get('ckpt_path', None)
        if ckpt_path and os.path.exists(os.path.join(ckpt_path, 'config.ini')):
            import configparser
            config = configparser.ConfigParser()
            config.read(os.path.join(ckpt_path, 'config.ini'))
            self.is_quantized = config.get('gpt', 'weight_data_type', fallback='fp16') in ['nf4', 'fp4']
        else:
            self.is_quantized = False

        super().__init__(*args, **kwargs)

    def load(self, ckpt_path, tp_rank, pipeline_para_rank):
        """Override load to support quantized weights"""
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Failed to find {ckpt_path}")

        w = []
        self.start_layer = self.layers_per_device * pipeline_para_rank
        self.end_layer = self.layers_per_device * (pipeline_para_rank + 1)

        if (pipeline_para_rank == self.pipeline_para_size - 1):
            self.layers_per_device += self.layer_num % self.pipeline_para_size
            self.end_layer = self.layer_num

        type_map = {np.float32: torch.float32, np.float16: torch.float16}
        str_type_map = self.str_type_map

        def is_load(i): return i >= self.start_layer and i < self.end_layer

        def load_to_torch(file_path: str, is_load: bool, shape_hint=None):
            """Load weight, dequantizing if necessary"""
            if not is_load:
                return torch.empty(0).to(str_type_map[self.inference_data_type])

            if not os.path.isfile(file_path):
                return torch.empty(0).to(str_type_map[self.inference_data_type])

            # Check if this is a quantized weight (has .scales.bin file)
            scales_path = file_path.replace('.bin', '.scales.bin')
            if self.is_quantized and os.path.isfile(scales_path):
                # Load quantized data and scales
                quantized = np.fromfile(file_path, dtype=np.uint8)
                scales = np.fromfile(scales_path, dtype=np.float16)

                # Dequantize
                if shape_hint is not None:
                    tensor = dequantize_nf4(quantized, scales, shape_hint)
                else:
                    # Try to infer shape from scales
                    # For 2D weights, scales has length = out_channels
                    print(f"Warning: No shape hint for {file_path}, may not dequantize correctly")
                    tensor = torch.from_numpy(quantized.astype(np.float16))

                return tensor.to(str_type_map[self.inference_data_type])
            else:
                # Load FP16/FP32 directly
                return torch.from_numpy(np.fromfile(file_path, dtype=self.weights_data_type)).to(str_type_map[self.inference_data_type])

        # Calculate shapes for proper dequantization
        hidden = self.global_hidden_units
        local_inter = self.local_inter_size

        # Layer norms
        w.extend([load_to_torch(f"{ckpt_path}/model.layers.{i}.input_layernorm.weight.bin", is_load(i), (hidden,))
                 for i in range(self.layer_num)])
        w.extend([load_to_torch(f"{ckpt_path}/model.layers.{i}.input_layernorm.bias.bin", is_load(i), (hidden,))
                 for i in range(self.layer_num)])

        # Attention QKV - shape depends on model config
        qkv_out_dim = hidden + 2 * (hidden // self.head_num * self.num_kv_heads)  # Q + K + V
        qkv_split_dim = qkv_out_dim // self.tensor_para_size
        w.extend([load_to_torch(
            f"{ckpt_path}/model.layers.{i}.attention.query_key_value.weight.{tp_rank}.bin",
            is_load(i), (qkv_split_dim, hidden)) for i in range(self.layer_num)])
        w.extend([load_to_torch(
            f"{ckpt_path}/model.layers.{i}.attention.query_key_value.bias.{tp_rank}.bin",
            is_load(i), (qkv_split_dim,)) for i in range(self.layer_num)])

        # Attention output
        attn_out_split = hidden // self.tensor_para_size
        w.extend([load_to_torch(f"{ckpt_path}/model.layers.{i}.attention.dense.weight.{tp_rank}.bin",
                 is_load(i), (hidden, attn_out_split)) for i in range(self.layer_num)])
        w.extend([load_to_torch(f"{ckpt_path}/model.layers.{i}.attention.dense.bias.bin",
                 is_load(i), (hidden,)) for i in range(self.layer_num)])

        # Post attention norm
        w.extend([load_to_torch(f"{ckpt_path}/model.layers.{i}.post_attention_layernorm.weight.bin",
                 is_load(i), (hidden,)) for i in range(self.layer_num)])
        w.extend([load_to_torch(f"{ckpt_path}/model.layers.{i}.post_attention_layernorm.bias.bin",
                 is_load(i), (hidden,)) for i in range(self.layer_num)])

        # FFN/MLP weights - for MoE, create placeholders
        if not self.gpt_with_moe:
            inter_split = local_inter // self.tensor_para_size
            w.extend([load_to_torch(f"{ckpt_path}/model.layers.{i}.mlp.dense_h_to_4h.weight.{tp_rank}.bin",
                     is_load(i), (inter_split, hidden)) for i in range(self.layer_num)])
            w.extend([load_to_torch(f"{ckpt_path}/model.layers.{i}.mlp.dense_h_to_4h.bias.{tp_rank}.bin",
                     is_load(i), (inter_split,)) for i in range(self.layer_num)])
            w.extend([load_to_torch(f"{ckpt_path}/model.layers.{i}.mlp.dense_4h_to_h.weight.{tp_rank}.bin",
                     is_load(i), (hidden, inter_split)) for i in range(self.layer_num)])
            w.extend([load_to_torch(f"{ckpt_path}/model.layers.{i}.mlp.dense_4h_to_h.bias.bin",
                     is_load(i), (hidden,)) for i in range(self.layer_num)])
        else:
            # For MoE, create placeholders (C++ loads experts separately)
            dtype = str_type_map[self.inference_data_type]
            w.extend([torch.zeros(hidden, local_inter, dtype=dtype) if is_load(i)
                     else torch.empty(0, dtype=dtype) for i in range(self.layer_num)])
            w.extend([torch.zeros(local_inter, dtype=dtype) if is_load(i)
                     else torch.empty(0, dtype=dtype) for i in range(self.layer_num)])
            w.extend([torch.zeros(local_inter, hidden, dtype=dtype) if is_load(i)
                     else torch.empty(0, dtype=dtype) for i in range(self.layer_num)])
            w.extend([torch.zeros(hidden, dtype=dtype) if is_load(i)
                     else torch.empty(0, dtype=dtype) for i in range(self.layer_num)])

        # Pre/post decoder norms
        if self.has_pre_decoder_layernorm:
            w.append(load_to_torch(f"{ckpt_path}/model.pre_decoder_layernorm.weight.bin", True, (hidden,)))
            w.append(load_to_torch(f"{ckpt_path}/model.pre_decoder_layernorm.bias.bin", True, (hidden,)))

        if self.has_post_decoder_layernorm:
            w.append(load_to_torch(f"{ckpt_path}/model.final_layernorm.weight.bin", True, (hidden,)))
            w.append(load_to_torch(f"{ckpt_path}/model.final_layernorm.bias.bin", True, (hidden,)))

        # Positional encoding
        if self.has_positional_encoding:
            wpe = load_to_torch(f"{ckpt_path}/model.wpe.bin", True).reshape(-1, hidden)
            assert self.max_seq_len <= wpe.size(0)
            w.append(wpe)

        # Word embeddings
        w.append(load_to_torch(f"{ckpt_path}/model.wte.bin", True, (self.vocab_size, hidden)))

        # LM head
        if os.path.isfile(f"{ckpt_path}/model.lm_head.weight.bin"):
            self.share_embed = False
            w.append(load_to_torch(f"{ckpt_path}/model.lm_head.weight.bin", True, (self.vocab_size, hidden)))
        else:
            self.share_embed = True
            w.append(torch.empty(0).to(str_type_map[self.inference_data_type]))

        # MoE gates (FP16, not quantized)
        gate_list = []
        for i in range(self.layer_num):
            if os.path.isfile(f"{ckpt_path}/model.layers.{i}.mlp.gate.weight.bin"):
                gate_list.append(load_to_torch(f"{ckpt_path}/model.layers.{i}.mlp.gate.weight.bin", True))
            else:
                gate_list.append(torch.empty(0).to(str_type_map[self.inference_data_type]))
        w.extend(gate_list)

        # Adapters if present
        if self.has_adapters:
            # Similar to above but for adapters...
            # Skipping for brevity as this model doesn't use adapters
            pass

        assert len(self.w) == len(w), f"Weight count mismatch: expected {len(self.w)}, got {len(w)}"

        # Reshape weights
        try:
            for i in range(len(w)):
                if w[i].nelement() == self.w[i].nelement():
                    self.w[i] = w[i].reshape(self.w[i].shape)
                else:
                    self.w[i] = w[i]
        except RuntimeError as e:
            raise RuntimeError(
                f"Shape mismatch at index {i}: expected {self.w[i].shape}, got {w[i].shape}"
            ) from e

        # INT8 quantization (if enabled)
        if self.int8_mode != 0:
            # Skip int8 conversion as we're using NF4
            pass

        print(f"✅ Loaded {'quantized' if self.is_quantized else 'fp16'} weights from {ckpt_path}")
