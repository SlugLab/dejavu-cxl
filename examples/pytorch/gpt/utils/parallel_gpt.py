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

from .gpt import GPT


class ParallelGPT(GPT):

    def cuda(self):
        self.weights._map(lambda w: w.cuda(self.device))
        if self.int8_mode != 0:
            self.weights._map_int8(lambda w: w.cuda(self.device))

        if self.build_model:
            del self.model
            self.build_model = False

        print(f"[ParallelGPT.cuda] Creating ParallelGptOp with hidden_size={self.hidden_size}")
        print(f"[ParallelGPT.cuda] head_num={self.head_num}, size_per_head={self.size_per_head}, head_num*size_per_head={self.head_num * self.size_per_head}")
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
            0,  # prompt_world_size (default for single GPU)
            0,  # token_world_size (default)
            0,  # torch_rank (default)
            False,  # is_restart
            self.hidden_size)  # hidden_size for GQA models where hidden != head_num * size_per_head
        self.build_model = True
