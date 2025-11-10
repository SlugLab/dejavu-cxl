/*
 * Copyright (c) 2024, DejaVu Authors
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "src/fastertransformer/utils/cuda_utils.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace fastertransformer {

/**
 * @brief Dequantize NF4 (Normal Float 4-bit) weights to FP16
 *
 * NF4 uses 16 levels optimized for normal distribution:
 * Provides better quality than symmetric quantization for neural network weights
 *
 * @param output Output FP16 tensor [rows, cols]
 * @param input Input uint8 tensor with packed 4-bit values [rows, (cols+1)/2]
 * @param scales Per-row scale factors in FP16 [rows]
 * @param rows Number of output rows
 * @param cols Number of output columns
 * @param stream CUDA stream
 */
template<typename T>
void invokeNF4Dequantize(T* output,
                        const uint8_t* input,
                        const half* scales,
                        const int rows,
                        const int cols,
                        cudaStream_t stream);

/**
 * @brief Dequantize FP4 (symmetric 4-bit) weights to FP16
 *
 * FP4 uses 16 evenly-spaced levels from -1 to 1
 * Simpler than NF4 but may have lower quality
 *
 * @param output Output FP16 tensor [rows, cols]
 * @param input Input uint8 tensor with packed 4-bit values [rows, (cols+1)/2]
 * @param scales Per-row scale factors in FP16 [rows]
 * @param rows Number of output rows
 * @param cols Number of output columns
 * @param stream CUDA stream
 */
template<typename T>
void invokeFP4Dequantize(T* output,
                        const uint8_t* input,
                        const half* scales,
                        const int rows,
                        const int cols,
                        cudaStream_t stream);

/**
 * @brief Load NF4 quantized weight file and dequantize to GPU memory
 *
 * @param weight_path Path to .bin file with quantized weights
 * @param scales_path Path to .scales.bin file with scale factors
 * @param rows Expected number of rows
 * @param cols Expected number of columns
 * @param output Pre-allocated GPU buffer for FP16 output
 * @param stream CUDA stream
 * @return true if successful
 */
bool loadAndDequantizeNF4(const char* weight_path,
                         const char* scales_path,
                         const int rows,
                         const int cols,
                         half* output,
                         cudaStream_t stream);

} // namespace fastertransformer
