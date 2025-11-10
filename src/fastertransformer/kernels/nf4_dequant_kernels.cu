/*
 * Copyright (c) 2024, DejaVu Authors
 * SPDX-License-Identifier: Apache-2.0
 */

#include "nf4_dequant_kernels.h"
#include "src/fastertransformer/utils/memory_utils.h"
#include <fstream>
#include <vector>

namespace fastertransformer {

// NF4 quantization levels stored as floats (converted to half in device code)
__device__ __constant__ float NF4_LEVELS[16] = {
    -1.0f,
    -0.6961928009986877f,
    -0.5250730514526367f,
    -0.39491748809814453f,
    -0.28444138169288635f,
    -0.18477343022823334f,
    -0.09105003625154495f,
    0.0f,
    0.07958029955625534f,
    0.16093020141124725f,
    0.24611230194568634f,
    0.33791524171829224f,
    0.44070982933044434f,
    0.5626170039176941f,
    0.7229568362236023f,
    1.0f
};

// FP4 quantization levels (16 evenly-spaced values from -1 to 1)
__device__ __constant__ half FP4_LEVELS[16];

// Initialize FP4 levels
void initFP4Levels() {
    static bool initialized = false;
    if (!initialized) {
        half fp4_levels_host[16];
        for (int i = 0; i < 16; i++) {
            float val = -1.0f + (2.0f * i / 15.0f);
            fp4_levels_host[i] = __float2half(val);
        }
        cudaMemcpyToSymbol(FP4_LEVELS, fp4_levels_host, 16 * sizeof(half));
        initialized = true;
    }
}

/**
 * @brief CUDA kernel for NF4 dequantization
 *
 * Each thread processes one element. Two 4-bit values are packed in each uint8.
 * High nibble = even index, low nibble = odd index
 */
template<typename T>
__global__ void nf4DequantizeKernel(T* __restrict__ output,
                                   const uint8_t* __restrict__ input,
                                   const half* __restrict__ scales,
                                   const int rows,
                                   const int cols)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= rows || col >= cols) {
        return;
    }

    // Calculate position in packed input
    // Two values per byte: even indices in high nibble, odd in low nibble
    const int flat_idx = row * cols + col;
    const int byte_idx = flat_idx / 2;
    const bool is_even = (flat_idx % 2) == 0;

    // Extract 4-bit index
    uint8_t packed_val = input[byte_idx];
    uint8_t quant_idx = is_even ? ((packed_val >> 4) & 0x0F) : (packed_val & 0x0F);

    // Lookup dequantized value from table and convert to half
    half dequant_val = __float2half(NF4_LEVELS[quant_idx]);

    // Apply per-row scale
    half scale = scales[row];
    half result = __hmul(dequant_val, scale);

    // Write output
    output[row * cols + col] = (T)result;
}

/**
 * @brief CUDA kernel for FP4 dequantization
 */
template<typename T>
__global__ void fp4DequantizeKernel(T* __restrict__ output,
                                   const uint8_t* __restrict__ input,
                                   const half* __restrict__ scales,
                                   const int rows,
                                   const int cols)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= rows || col >= cols) {
        return;
    }

    const int flat_idx = row * cols + col;
    const int byte_idx = flat_idx / 2;
    const bool is_even = (flat_idx % 2) == 0;

    uint8_t packed_val = input[byte_idx];
    uint8_t quant_idx = is_even ? ((packed_val >> 4) & 0x0F) : (packed_val & 0x0F);

    half dequant_val = FP4_LEVELS[quant_idx];
    half scale = scales[row];
    half result = __hmul(dequant_val, scale);

    output[row * cols + col] = (T)result;
}

// Explicit template instantiation
template<typename T>
void invokeNF4Dequantize(T* output,
                        const uint8_t* input,
                        const half* scales,
                        const int rows,
                        const int cols,
                        cudaStream_t stream)
{
    // Each block processes 16x16 = 256 elements
    dim3 block(16, 16);
    dim3 grid((cols + block.x - 1) / block.x,
              (rows + block.y - 1) / block.y);

    nf4DequantizeKernel<T><<<grid, block, 0, stream>>>(
        output, input, scales, rows, cols
    );
}

template<typename T>
void invokeFP4Dequantize(T* output,
                        const uint8_t* input,
                        const half* scales,
                        const int rows,
                        const int cols,
                        cudaStream_t stream)
{
    initFP4Levels();

    dim3 block(16, 16);
    dim3 grid((cols + block.x - 1) / block.x,
              (rows + block.y - 1) / block.y);

    fp4DequantizeKernel<T><<<grid, block, 0, stream>>>(
        output, input, scales, rows, cols
    );
}

// Explicit instantiations
template void invokeNF4Dequantize<half>(half*, const uint8_t*, const half*, const int, const int, cudaStream_t);
template void invokeNF4Dequantize<float>(float*, const uint8_t*, const half*, const int, const int, cudaStream_t);

template void invokeFP4Dequantize<half>(half*, const uint8_t*, const half*, const int, const int, cudaStream_t);
template void invokeFP4Dequantize<float>(float*, const uint8_t*, const half*, const int, const int, cudaStream_t);

/**
 * @brief Load quantized weights from disk and dequantize to GPU
 */
bool loadAndDequantizeNF4(const char* weight_path,
                         const char* scales_path,
                         const int rows,
                         const int cols,
                         half* output,
                         cudaStream_t stream)
{
    // Calculate sizes
    const size_t num_elements = rows * cols;
    const size_t packed_size = (num_elements + 1) / 2;  // Two 4-bit values per byte

    // Load quantized weights from disk
    std::ifstream weight_file(weight_path, std::ios::binary);
    if (!weight_file.is_open()) {
        printf("[ERROR] Failed to open weight file: %s\n", weight_path);
        return false;
    }

    std::vector<uint8_t> packed_weights(packed_size);
    weight_file.read(reinterpret_cast<char*>(packed_weights.data()), packed_size);
    weight_file.close();

    if (weight_file.gcount() != packed_size) {
        printf("[ERROR] Weight file size mismatch: expected %zu bytes, got %zu\n",
               packed_size, weight_file.gcount());
        return false;
    }

    // Load scales from disk
    std::ifstream scales_file(scales_path, std::ios::binary);
    if (!scales_file.is_open()) {
        printf("[ERROR] Failed to open scales file: %s\n", scales_path);
        return false;
    }

    std::vector<half> scales_host(rows);
    scales_file.read(reinterpret_cast<char*>(scales_host.data()), rows * sizeof(half));
    scales_file.close();

    if (scales_file.gcount() != rows * sizeof(half)) {
        printf("[ERROR] Scales file size mismatch: expected %zu bytes, got %zu\n",
               rows * sizeof(half), scales_file.gcount());
        return false;
    }

    // Allocate GPU memory for packed weights and scales
    uint8_t* packed_weights_gpu = nullptr;
    half* scales_gpu = nullptr;

    cudaMalloc(&packed_weights_gpu, packed_size);
    cudaMalloc(&scales_gpu, rows * sizeof(half));

    // Copy to GPU
    cudaMemcpyAsync(packed_weights_gpu, packed_weights.data(), packed_size,
                   cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(scales_gpu, scales_host.data(), rows * sizeof(half),
                   cudaMemcpyHostToDevice, stream);

    // Dequantize on GPU
    invokeNF4Dequantize<half>(output, packed_weights_gpu, scales_gpu, rows, cols, stream);

    // Wait for completion
    cudaStreamSynchronize(stream);

    // Free temporary GPU memory
    cudaFree(packed_weights_gpu);
    cudaFree(scales_gpu);

    return true;
}

} // namespace fastertransformer
