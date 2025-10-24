/*
 * Simple test for MoE Token FT Manager
 */

#include "src/fastertransformer/utils/moe_token_ft_manager.h"
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

using namespace fastertransformer;

int main() {
    std::cout << "=== MoE Token FT Manager Test ===" << std::endl;

    // Configuration for Qwen235B MoE
    const size_t num_experts = 256;
    const size_t moe_k = 8;
    const size_t hidden_units = 8192;
    const size_t checkpoint_interval = 1;
    const size_t max_checkpoints = 10;

    try {
        // Initialize MoE Token FT Manager
        std::cout << "Initializing MoE Token FT Manager..." << std::endl;
        MoETokenFTManager ft_manager(
            num_experts,
            moe_k,
            hidden_units,
            checkpoint_interval,
            max_checkpoints
        );

        std::cout << "✓ MoE Token FT Manager initialized successfully" << std::endl;

        // Create dummy data for testing
        std::vector<int> expert_indices(moe_k);
        std::vector<float> expert_weights(moe_k);

        // Simulate expert routing
        for (size_t i = 0; i < moe_k; i++) {
            expert_indices[i] = i * 10;  // Expert IDs
            expert_weights[i] = 1.0f / moe_k;  // Equal weights
        }

        // Allocate GPU memory for activations
        void* expert_activations_gpu;
        size_t activation_size = moe_k * hidden_units * sizeof(float);
        cudaMalloc(&expert_activations_gpu, activation_size);

        // Create a CUDA stream
        cudaStream_t stream;
        cudaStreamCreate(&stream);

        std::cout << "\nCreating checkpoints..." << std::endl;

        // Create multiple checkpoints
        for (int step = 0; step < 5; step++) {
            for (int token_id = 0; token_id < 3; token_id++) {
                int ubatch_id = 0;

                ft_manager.createCheckpoint(
                    token_id,
                    step,
                    ubatch_id,
                    expert_indices.data(),
                    expert_weights.data(),
                    expert_activations_gpu,
                    activation_size,
                    stream
                );
            }
        }

        cudaStreamSynchronize(stream);
        std::cout << "✓ Created 15 checkpoints (5 steps × 3 tokens)" << std::endl;

        // Print statistics
        std::cout << "\nCheckpoint Statistics:" << std::endl;
        ft_manager.printCheckpointStats();

        // Test checkpoint retrieval
        std::cout << "\nTesting checkpoint retrieval..." << std::endl;
        bool has_checkpoint = ft_manager.hasCheckpoint(2, 0);
        std::cout << "Has checkpoint at step 2, ubatch 0: "
                  << (has_checkpoint ? "YES" : "NO") << std::endl;

        size_t count = ft_manager.getCheckpointCount(0);
        std::cout << "Total checkpoints for ubatch 0: " << count << std::endl;

        // Test recovery
        std::cout << "\nTesting recovery..." << std::endl;
        std::vector<int> recovered_indices(moe_k);
        std::vector<float> recovered_weights(moe_k);
        void* recovered_activations_gpu;
        cudaMalloc(&recovered_activations_gpu, activation_size);

        bool recovery_success = ft_manager.restoreFromCheckpoint(
            2,  // step
            0,  // ubatch_id
            1,  // token_id
            recovered_indices.data(),
            recovered_weights.data(),
            recovered_activations_gpu,
            stream
        );

        cudaStreamSynchronize(stream);

        if (recovery_success) {
            std::cout << "✓ Recovery successful!" << std::endl;
            std::cout << "Recovered expert indices: [";
            for (size_t i = 0; i < moe_k; i++) {
                std::cout << recovered_indices[i];
                if (i < moe_k - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        } else {
            std::cout << "✗ Recovery failed" << std::endl;
        }

        // Test adaptive checkpoint policy
        std::cout << "\nTesting adaptive checkpoint policy..." << std::endl;
        AdaptiveCheckpointPolicy policy(num_experts, moe_k);

        bool should_checkpoint = policy.shouldCheckpoint(
            expert_indices.data(),
            expert_weights.data(),
            10  // num_tokens
        );

        float entropy = policy.calculateEntropy();
        float imbalance = policy.calculateImbalance();

        std::cout << "Adaptive policy decision: "
                  << (should_checkpoint ? "CHECKPOINT" : "SKIP") << std::endl;
        std::cout << "Entropy: " << entropy << std::endl;
        std::cout << "Imbalance: " << imbalance << std::endl;

        // Cleanup
        cudaFree(expert_activations_gpu);
        cudaFree(recovered_activations_gpu);
        cudaStreamDestroy(stream);

        std::cout << "\n✓ All tests passed!" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "✗ Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}
