/*
 * Example: Integrating MoE Token FT into existing DejaVu application
 *
 * This example shows how to add MoE token stream fault tolerance
 * to an existing Qwen235B MoE inference application.
 */

#include "src/fastertransformer/models/multi_gpu_gpt/ParallelGptDVFT.h"
#include "src/fastertransformer/utils/moe_token_ft_manager.h"
#include <iostream>
#include <exception>

using namespace fastertransformer;

// Example: Initialize model with MoE Token FT
template<typename T>
ParallelGptDVFT<T>* initializeQwen235BMoEWithFT(
    size_t max_batch_size,
    size_t max_seq_len,
    NcclParam tensor_para,
    NcclParam pipeline_para,
    NcclParam cache_stream_para,
    cudaStream_t stream,
    cublasMMWrapper* cublas_wrapper,
    IAllocator* allocator)
{
    // Qwen235B MoE configuration
    const size_t num_layers = 80;
    const size_t hidden_size = 8192;
    const size_t num_heads = 64;
    const size_t size_per_head = hidden_size / num_heads;
    const size_t inter_size = 29568;
    const size_t vocab_size = 152064;
    const size_t num_experts = 256;
    const size_t moe_k = 8;

    // MoE layer indices (example: every other layer)
    std::vector<int64_t> moe_layer_index;
    for (int i = 1; i < num_layers; i += 2) {
        moe_layer_index.push_back(i);
    }

    // Create model
    auto* model = new ParallelGptDVFT<T>(
        max_batch_size,
        max_seq_len,
        2048,                    // max_input_len
        1,                       // beam_width
        num_heads,
        size_per_head,
        inter_size,
        num_layers,
        num_experts,
        moe_k,
        moe_layer_index,
        vocab_size,
        1,                       // start_id
        2,                       // end_id
        0,                       // prompt_learning_start_id
        PromptLearningType::no_prompt,
        gptVariantParams{},      // default variant params
        0.0f,                    // beam_search_diversity_rate
        1,                       // top_k
        0.0f,                    // top_p
        0,                       // random_seed
        1.0f,                    // temperature
        1.0f,                    // len_penalty
        1.0f,                    // repetition_penalty
        tensor_para,
        pipeline_para,
        cache_stream_para,
        8,                       // prompt_world_size
        8,                       // token_world_size
        stream,
        cublas_wrapper,
        allocator,
        true,                    // is_free_buffer_after_forward
        nullptr,                 // cuda_device_prop
        AttentionType::UNFUSED_MHA,
        false,                   // sparse
        0,                       // int8_mode
        nullptr,                 // custom_all_reduce_comm
        0,                       // enable_custom_all_reduce
        1.0f                     // shared_contexts_ratio
    );

    // Initialize MoE Token FT
    size_t checkpoint_interval = 1;  // Checkpoint every token

    // Read from environment if set
    char* env_interval = std::getenv("MOE_CHECKPOINT_INTERVAL");
    if (env_interval != nullptr) {
        checkpoint_interval = std::stoul(env_interval);
    }

    bool enable_ft = true;
    char* env_enable = std::getenv("ENABLE_MOE_TOKEN_FT");
    if (env_enable != nullptr && std::string(env_enable) == "0") {
        enable_ft = false;
    }

    if (enable_ft) {
        std::cout << "Initializing MoE Token FT with interval=" << checkpoint_interval << std::endl;
        model->initializeMoETokenFT(true, checkpoint_interval);

        if (model->isMoETokenFTEnabled()) {
            std::cout << "MoE Token FT successfully enabled!" << std::endl;
            model->printMoETokenFTStats();
        }
    } else {
        std::cout << "MoE Token FT disabled" << std::endl;
    }

    return model;
}

// Example: Token generation with checkpointing
template<typename T>
void generateTokensWithFT(
    ParallelGptDVFT<T>* model,
    int* output_ids,
    const int* input_ids,
    int batch_size,
    int max_gen_len,
    const int* expert_indices_device,      // From MoE layer
    const float* expert_weights_device,    // From MoE layer
    const void* expert_activations_device, // From MoE layer
    size_t activation_size)
{
    for (int step = 0; step < max_gen_len; step++) {
        try {
            // Normal token generation happens here
            // ... (forward pass, sampling, etc.)

            // After MoE layer forward pass, checkpoint if enabled
            if (model->isMoETokenFTEnabled()) {
                for (int b = 0; b < batch_size; b++) {
                    int token_id = step;
                    int ubatch_id = b;

                    // Create checkpoint for this token
                    model->checkpointMoEToken(
                        token_id,
                        step,
                        ubatch_id,
                        expert_indices_device + b * model->getHiddenUnits(),
                        expert_weights_device + b * model->getHiddenUnits(),
                        expert_activations_device,
                        activation_size
                    );
                }
            }

        } catch (const std::exception& e) {
            std::cerr << "Error during token generation at step " << step
                      << ": " << e.what() << std::endl;

            if (model->isMoETokenFTEnabled()) {
                std::cout << "Initiating MoE recovery..." << std::endl;

                // Handle failure and recover
                for (int b = 0; b < batch_size; b++) {
                    model->handleMoEFailure(step, b);
                }

                std::cout << "Recovery initiated. Resuming from checkpoint..." << std::endl;

                // Recovery happens automatically
                // The system will restore from last valid checkpoint

                model->completeMoERecovery();
            } else {
                // Without FT, we have to fail
                throw;
            }
        }
    }
}

// Example: Manual recovery
template<typename T>
bool manualRecoveryExample(
    ParallelGptDVFT<T>* model,
    int failed_step,
    int ubatch_id)
{
    if (!model->isMoETokenFTEnabled()) {
        std::cerr << "MoE Token FT not enabled, cannot recover" << std::endl;
        return false;
    }

    std::cout << "Manually recovering from step " << failed_step << std::endl;

    // Allocate buffers for recovered data
    const size_t moe_k = 8;
    const size_t hidden_size = 8192;

    int* expert_indices = new int[moe_k];
    float* expert_weights = new float[moe_k];
    void* expert_activations;
    cudaMalloc(&expert_activations, moe_k * hidden_size * sizeof(float));

    // Try to recover each token up to the failed step
    int recovery_step = failed_step - 1;
    bool all_recovered = true;

    for (int token_id = 0; token_id <= recovery_step; token_id++) {
        bool success = model->recoverMoEToken(
            recovery_step,
            ubatch_id,
            token_id,
            expert_indices,
            expert_weights,
            expert_activations
        );

        if (!success) {
            std::cerr << "Failed to recover token " << token_id << std::endl;
            all_recovered = false;
            break;
        }

        std::cout << "Recovered token " << token_id << " - experts: [";
        for (size_t i = 0; i < moe_k; i++) {
            std::cout << expert_indices[i];
            if (i < moe_k - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }

    // Cleanup
    delete[] expert_indices;
    delete[] expert_weights;
    cudaFree(expert_activations);

    if (all_recovered) {
        model->completeMoERecovery();
        std::cout << "Manual recovery completed successfully" << std::endl;
    }

    return all_recovered;
}

// Example: Adaptive checkpointing
void adaptiveCheckpointExample() {
    const size_t num_experts = 256;
    const size_t moe_k = 8;

    AdaptiveCheckpointPolicy policy(num_experts, moe_k);

    // Simulate expert routing for 100 tokens
    std::vector<int> expert_indices(100 * moe_k);
    std::vector<float> expert_weights(100 * moe_k);

    // Fill with example data (random routing)
    for (size_t i = 0; i < expert_indices.size(); i++) {
        expert_indices[i] = rand() % num_experts;
        expert_weights[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Check if we should checkpoint based on expert distribution
    bool should_checkpoint = policy.shouldCheckpoint(
        expert_indices.data(),
        expert_weights.data(),
        100  // num_tokens
    );

    float entropy = policy.calculateEntropy();
    float imbalance = policy.calculateImbalance();

    std::cout << "Adaptive checkpoint decision: " << (should_checkpoint ? "YES" : "NO") << std::endl;
    std::cout << "  Entropy: " << entropy << std::endl;
    std::cout << "  Imbalance: " << imbalance << std::endl;
}

// Example: Monitoring and stats
template<typename T>
void monitoringExample(ParallelGptDVFT<T>* model) {
    if (!model->isMoETokenFTEnabled()) {
        std::cout << "MoE Token FT is not enabled" << std::endl;
        return;
    }

    std::cout << "\n=== MoE Token FT Monitoring ===" << std::endl;

    // Print comprehensive statistics
    model->printMoETokenFTStats();

    // Additional custom monitoring could be added here
    // For example, tracking checkpoint rates, memory pressure, etc.
}

// Main example
int main(int argc, char** argv) {
    std::cout << "MoE Token FT Integration Example for Qwen235B" << std::endl;
    std::cout << "===============================================\n" << std::endl;

    // Example 1: Adaptive checkpointing
    std::cout << "Example 1: Adaptive Checkpointing\n" << std::endl;
    adaptiveCheckpointExample();

    std::cout << "\n---\n" << std::endl;

    // Example 2: Model initialization (would require actual MPI/NCCL setup)
    std::cout << "Example 2: Model Initialization" << std::endl;
    std::cout << "Note: This is a skeleton - requires actual MPI/NCCL setup" << std::endl;

    // In a real application, you would:
    // 1. Initialize MPI/NCCL
    // 2. Set up CUDA devices
    // 3. Create allocators and CUBLAS wrappers
    // 4. Call initializeQwen235BMoEWithFT()

    std::cout << "\nFor a complete example, see:" << std::endl;
    std::cout << "  - examples/pytorch/gpt/multi_gpu_gpt_example.py" << std::endl;
    std::cout << "  - examples/pytorch/gpt/utils/gpt_dv.py" << std::endl;

    std::cout << "\nIntegration complete!" << std::endl;

    return 0;
}
