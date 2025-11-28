# Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
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

add_definitions(-DTORCH_CUDA=1)
if(STREAM_SYNC)
	add_definitions(-DSTREAM_SYNC=1)
endif()

add_library(th_utils STATIC th_utils.cu)
set_property(TARGET th_utils PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET th_utils PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(th_utils PUBLIC torch -lcublas -lcudart -lcurand tensor)

# GPT-only build - excludes protobuf/gRPC dependent components
add_subdirectory(common)
add_subdirectory(multi_gpu_gpt)

add_library(th_transformer SHARED
            $<TARGET_OBJECTS:th_common>
            $<TARGET_OBJECTS:th_parallel_gpt>
            $<TARGET_OBJECTS:th_utils>
            # Pull in FT model objects directly to resolve template symbols at link time
            $<TARGET_OBJECTS:ParallelGptContextDecoder>
            $<TARGET_OBJECTS:ParallelGptDecoder>
            $<TARGET_OBJECTS:ParallelGptDecoderLayerWeight>
            # Ensure cublas host symbols are present in this .so
            $<TARGET_OBJECTS:cublasAlgoMap>
            $<TARGET_OBJECTS:cublasMMWrapper>
)
target_link_libraries(th_transformer PUBLIC torch
                      # Link core FT GPT libs explicitly
                      ParallelGpt
                      ParallelGptContextDecoder
                      ParallelGptDecoder
                      ParallelGptDecoderLayerWeight
                      TensorParallelGptContextAttentionLayer
                      TensorParallelDecoderSelfAttentionLayer
                      -Wl,--whole-archive th_parallel_gpt -Wl,--no-whole-archive
                      th_common
                      th_utils
                      calibrate_quantize_weight_kernels
                      nccl_utils
                      -lmpi_cxx
)

# Add include directories for the directly compiled th_op sources
target_include_directories(th_transformer PRIVATE ${Protobuf_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS})

# Ensure linker does not garbage-collect static initializer sections for registrars
target_link_options(th_transformer PUBLIC "-Wl,--no-gc-sections" "-Wl,--no-as-needed" "-Wl,--allow-shlib-undefined")

if(ENABLE_FP8)
add_subdirectory(gpt_fp8)
target_link_libraries(th_transformer PUBLIC $<TARGET_OBJECTS:th_gpt_fp8> th_gpt_fp8)
endif()
