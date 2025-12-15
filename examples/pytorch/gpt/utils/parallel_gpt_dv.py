# Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import time

from .gpt_dv import GPT


class ParallelGPT(GPT):

    def cuda(self):
        if self.build_model:
            del self.model
            self.build_model = False

        print(f"[DEBUG] ParallelGptOp parameters:")
        print(f"  expert_num={self.expert_num} (type: {type(self.expert_num)})")
        print(f"  moe_k={self.moe_k} (type: {type(self.moe_k)})")
        print(f"  moe_layer_index={self.moe_layer_index} (type: {type(self.moe_layer_index)})")
        print(f"  inter_size={self.inter_size} (type: {type(self.inter_size)})")
        print(f"  layernorm_type={self.layernorm_type} (type: {type(self.layernorm_type)})")
        print(f"  activation_type={self.activation_type} (type: {type(self.activation_type)})")

        # Debug weight tensors
        print(f"[DEBUG] Weight tensor validation:")
        print(f"  Total weights: {len(self.weights.w)}")
        print(f"  int8_weights: {len(self.weights.int8_w)}")
        print(f"  scales: {len(self.weights.scale)}")

        # Check for any problematic tensors
        problem_found = False
        for i, w in enumerate(self.weights.w):
            if not w.is_cuda or w.dtype not in [torch.float16, torch.float32, torch.bfloat16] or not w.is_contiguous():
                print(f"    WARNING w[{i}]: shape={w.shape}, dtype={w.dtype}, device={w.device}, is_cuda={w.is_cuda}, is_contiguous={w.is_contiguous()}")
                problem_found = True

        if not problem_found:
            print(f"  All {len(self.weights.w)} weights validated: fp16, CUDA, contiguous")
            # Show first/last few and some middle weights (QKV should be around index 96)
            for i in [0, 1, 2, 96, 97, 98, 144, 192, len(self.weights.w)-3, len(self.weights.w)-2, len(self.weights.w)-1]:
                if i < len(self.weights.w):
                    w = self.weights.w[i]
                    print(f"    w[{i}]: shape={w.shape}, numel={w.numel()}")

        self.model = torch.classes.FasterTransformer.ParallelGptOp(
            self.head_num, self.size_per_head, self.inter_size,
            self.layer_num,
            self.expert_num,
            self.moe_k,
            self.moe_layer_index,
            self.vocab_size, self.start_id, self.end_id,
            self.tensor_para_size, self.pipeline_para_size, self.int8_mode,
            # GPT variant parameters
            self.layernorm_eps,
            self.layernorm_type,
            self.activation_type,
            self.has_positional_encoding,
            self.has_pre_decoder_layernorm,
            self.has_post_decoder_layernorm,
            self.has_adapters,
            self.adapter_inter_size,
            self.use_attention_linear_bias,
            self.weights.w,
            self.weights.int8_w,
            self.weights.scale,
            self.shared_contexts_ratio,
            self.prompt_world_size,
            self.token_world_size,
            self.torch_rank,
            self.restart,
            self.hidden_size)  # hidden_size for GQA models where hidden != head_num * size_per_head
        self.build_model = True
