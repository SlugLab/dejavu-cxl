/*
 * Copyright (c) 2024, DejaVu Authors
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "src/fastertransformer/kernels/nf4_dequant_kernels.h"
#include "src/fastertransformer/utils/cuda_utils.h"
#include <string>
#include <fstream>

namespace fastertransformer {

enum class WeightQuantType {
    FP16,   // Standard FP16 weights
    FP32,   // Standard FP32 weights
    NF4,    // Normal Float 4-bit quantization
    FP4     // Symmetric FP4 quantization
};

/**
 * @brief Helper class for loading quantized MoE expert weights
 *
 * Loads quantized weights on-demand and dequantizes them to GPU memory.
 * This significantly reduces memory usage for large MoE models.
 */
class QuantizedWeightLoader {
public:
    QuantizedWeightLoader(WeightQuantType quant_type = WeightQuantType::FP16)
        : quant_type_(quant_type) {}

    /**
     * @brief Load a weight file, dequantizing if necessary
     *
     * @param weight_path Path to weight file (.bin)
     * @param rows Number of rows in weight matrix
     * @param cols Number of columns in weight matrix
     * @param output Pre-allocated GPU buffer for output
     * @param stream CUDA stream
     * @return true if successful
     */
    bool loadWeight(const std::string& weight_path,
                   int rows,
                   int cols,
                   half* output,
                   cudaStream_t stream);

    /**
     * @brief Check if a weight file is quantized
     *
     * @param weight_path Path to weight file
     * @return true if quantized (.scales.bin file exists)
     */
    static bool isQuantized(const std::string& weight_path);

    /**
     * @brief Get quantization type from config file
     *
     * @param config_path Path to config.ini
     * @return WeightQuantType
     */
    static WeightQuantType getQuantTypeFromConfig(const std::string& config_path);

private:
    WeightQuantType quant_type_;

    /**
     * @brief Load FP16/FP32 weight directly
     */
    bool loadFP16Weight(const std::string& weight_path,
                       int rows,
                       int cols,
                       half* output,
                       cudaStream_t stream);
};

/**
 * @brief Load expert weights with quantization support
 *
 * This function handles loading of MoE expert weights, automatically
 * detecting and dequantizing quantized weights.
 *
 * @param base_path Base path for weight files
 * @param layer_idx Layer index
 * @param expert_idx Expert index
 * @param weight_name Weight name (e.g., "gate_proj", "up_proj", "down_proj")
 * @param tp_rank Tensor parallel rank
 * @param rows Expected rows
 * @param cols Expected columns
 * @param output Pre-allocated GPU buffer
 * @param quant_type Quantization type
 * @param stream CUDA stream
 * @return true if successful
 */
bool loadExpertWeight(const std::string& base_path,
                     int layer_idx,
                     int expert_idx,
                     const std::string& weight_name,
                     int tp_rank,
                     int rows,
                     int cols,
                     half* output,
                     WeightQuantType quant_type,
                     cudaStream_t stream);

} // namespace fastertransformer
