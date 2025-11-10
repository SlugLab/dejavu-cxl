/*
 * Copyright (c) 2024, DejaVu Authors
 * SPDX-License-Identifier: Apache-2.0
 */

#include "nf4_weight_loader.h"
#include "src/fastertransformer/utils/logger.h"
#include <fstream>
#include <sys/stat.h>

namespace fastertransformer {

bool QuantizedWeightLoader::isQuantized(const std::string& weight_path) {
    std::string scales_path = weight_path;
    size_t pos = scales_path.rfind(".bin");
    if (pos != std::string::npos) {
        scales_path.replace(pos, 4, ".scales.bin");
    } else {
        scales_path += ".scales.bin";
    }

    struct stat buffer;
    return (stat(scales_path.c_str(), &buffer) == 0);
}

WeightQuantType QuantizedWeightLoader::getQuantTypeFromConfig(const std::string& config_path) {
    std::ifstream config_file(config_path);
    if (!config_file.is_open()) {
        FT_LOG_WARNING("Could not open config file: %s, assuming FP16", config_path.c_str());
        return WeightQuantType::FP16;
    }

    std::string line;
    while (std::getline(config_file, line)) {
        // Look for weight_data_type = nf4/fp4/fp16
        if (line.find("weight_data_type") != std::string::npos) {
            if (line.find("nf4") != std::string::npos) {
                return WeightQuantType::NF4;
            } else if (line.find("fp4") != std::string::npos) {
                return WeightQuantType::FP4;
            } else if (line.find("fp32") != std::string::npos) {
                return WeightQuantType::FP32;
            }
        }
    }

    return WeightQuantType::FP16;
}

bool QuantizedWeightLoader::loadFP16Weight(const std::string& weight_path,
                                          int rows,
                                          int cols,
                                          half* output,
                                          cudaStream_t stream) {
    std::ifstream weight_file(weight_path, std::ios::binary);
    if (!weight_file.is_open()) {
        FT_LOG_ERROR("Failed to open weight file: %s", weight_path.c_str());
        return false;
    }

    // Get file size
    weight_file.seekg(0, std::ios::end);
    size_t file_size = weight_file.tellg();
    weight_file.seekg(0, std::ios::beg);

    size_t expected_size = rows * cols * sizeof(half);
    if (file_size != expected_size) {
        FT_LOG_ERROR("Weight file size mismatch for %s: expected %zu bytes, got %zu",
                    weight_path.c_str(), expected_size, file_size);
        return false;
    }

    // Load to host buffer then copy to GPU
    std::vector<half> host_buffer(rows * cols);
    weight_file.read(reinterpret_cast<char*>(host_buffer.data()), file_size);
    weight_file.close();

    check_cuda_error(cudaMemcpyAsync(output, host_buffer.data(), file_size,
                                    cudaMemcpyHostToDevice, stream));

    return true;
}

bool QuantizedWeightLoader::loadWeight(const std::string& weight_path,
                                      int rows,
                                      int cols,
                                      half* output,
                                      cudaStream_t stream) {
    // Check if weight is quantized
    bool is_quant = isQuantized(weight_path);

    if (!is_quant || quant_type_ == WeightQuantType::FP16) {
        // Load FP16 directly
        return loadFP16Weight(weight_path, rows, cols, output, stream);
    }

    // Load and dequantize
    std::string scales_path = weight_path;
    size_t pos = scales_path.rfind(".bin");
    scales_path.replace(pos, 4, ".scales.bin");

    if (quant_type_ == WeightQuantType::NF4) {
        return loadAndDequantizeNF4(weight_path.c_str(), scales_path.c_str(),
                                   rows, cols, output, stream);
    } else if (quant_type_ == WeightQuantType::FP4) {
        // Similar to NF4 but use FP4 dequantization
        std::ifstream weight_file(weight_path, std::ios::binary);
        if (!weight_file.is_open()) {
            FT_LOG_ERROR("Failed to open weight file: %s", weight_path.c_str());
            return false;
        }

        const size_t num_elements = rows * cols;
        const size_t packed_size = (num_elements + 1) / 2;

        std::vector<uint8_t> packed_weights(packed_size);
        weight_file.read(reinterpret_cast<char*>(packed_weights.data()), packed_size);
        weight_file.close();

        std::ifstream scales_file(scales_path, std::ios::binary);
        if (!scales_file.is_open()) {
            FT_LOG_ERROR("Failed to open scales file: %s", scales_path.c_str());
            return false;
        }

        std::vector<half> scales_host(rows);
        scales_file.read(reinterpret_cast<char*>(scales_host.data()), rows * sizeof(half));
        scales_file.close();

        // Allocate GPU memory
        uint8_t* packed_weights_gpu = nullptr;
        half* scales_gpu = nullptr;

        check_cuda_error(cudaMalloc(&packed_weights_gpu, packed_size));
        check_cuda_error(cudaMalloc(&scales_gpu, rows * sizeof(half)));

        check_cuda_error(cudaMemcpyAsync(packed_weights_gpu, packed_weights.data(), packed_size,
                                        cudaMemcpyHostToDevice, stream));
        check_cuda_error(cudaMemcpyAsync(scales_gpu, scales_host.data(), rows * sizeof(half),
                                        cudaMemcpyHostToDevice, stream));

        invokeFP4Dequantize<half>(output, packed_weights_gpu, scales_gpu, rows, cols, stream);

        check_cuda_error(cudaStreamSynchronize(stream));

        check_cuda_error(cudaFree(packed_weights_gpu));
        check_cuda_error(cudaFree(scales_gpu));

        return true;
    }

    FT_LOG_ERROR("Unsupported quantization type");
    return false;
}

bool loadExpertWeight(const std::string& base_path,
                     int layer_idx,
                     int expert_idx,
                     const std::string& weight_name,
                     int tp_rank,
                     int rows,
                     int cols,
                     half* output,
                     WeightQuantType quant_type,
                     cudaStream_t stream) {
    // Construct weight path:
    // base_path/model.layers.{layer_idx}.mlp.experts.{expert_idx}.{weight_name}.weight.{tp_rank}.bin
    char path_buffer[1024];
    snprintf(path_buffer, sizeof(path_buffer),
            "%s/model.layers.%d.mlp.experts.%d.%s.weight.%d.bin",
            base_path.c_str(), layer_idx, expert_idx, weight_name.c_str(), tp_rank);

    std::string weight_path(path_buffer);

    QuantizedWeightLoader loader(quant_type);
    return loader.loadWeight(weight_path, rows, cols, output, stream);
}

} // namespace fastertransformer
