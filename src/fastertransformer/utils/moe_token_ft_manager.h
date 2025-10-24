/*
 * Copyright (c) 2025, MSR-Fiddle
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <unordered_map>
#include <memory>
#include <mutex>
#include "src/fastertransformer/utils/cache_utils.h"

namespace fastertransformer {

// Structure to hold MoE token-level checkpoint data
struct MoETokenCheckpoint {
    int token_id;                           // Token position in sequence
    int step;                               // Generation step
    int ubatch_id;                          // Microbatch ID

    // Expert routing information
    std::vector<int> expert_indices;        // Which experts were selected (k experts per token)
    std::vector<float> expert_weights;      // Gating weights for selected experts

    // Expert activations (intermediate states)
    void* expert_activations_ptr;           // GPU pointer to expert activations
    size_t activation_size;                 // Size in bytes

    // Timestamp for ordering
    uint64_t timestamp;

    // Validation flag
    bool is_valid;

    MoETokenCheckpoint()
        : token_id(-1), step(-1), ubatch_id(-1),
          expert_activations_ptr(nullptr), activation_size(0),
          timestamp(0), is_valid(false) {}
};

// Structure to track token stream state for recovery
struct TokenStreamState {
    int current_step;                       // Current generation step
    int checkpoint_step;                    // Last checkpointed step
    std::vector<int> active_ubatch_ids;     // Active microbatch IDs

    // Per-microbatch token checkpoints
    std::unordered_map<int, std::vector<MoETokenCheckpoint>> ubatch_checkpoints;

    // Recovery metadata
    bool in_recovery;                       // Whether we're in recovery mode
    int recovery_target_step;               // Step to recover to

    TokenStreamState()
        : current_step(0), checkpoint_step(0),
          in_recovery(false), recovery_target_step(-1) {}
};

// MoE Token Stream Fault Tolerance Manager
class MoETokenFTManager {
private:
    // Configuration
    size_t num_experts_;
    size_t moe_k_;                          // Top-k experts
    size_t hidden_units_;
    size_t checkpoint_interval_;            // Checkpoint every N tokens
    size_t max_checkpoints_;                // Maximum checkpoints to keep

    // State tracking
    TokenStreamState stream_state_;
    std::mutex state_mutex_;

    // Memory management
    cudaStream_t checkpoint_stream_;
    cudaStream_t recovery_stream_;

    // Host memory buffer for checkpoints
    void* host_checkpoint_buffer_;
    size_t host_buffer_size_;

    // Device memory pool for activations
    void* device_checkpoint_pool_;
    size_t device_pool_size_;
    size_t device_pool_offset_;

    // Checkpoint storage policy
    enum CheckpointPolicy {
        CHECKPOINT_ALL,                     // Checkpoint every token
        CHECKPOINT_INTERVAL,                // Checkpoint at intervals
        CHECKPOINT_ADAPTIVE                 // Adaptive based on expert distribution
    };
    CheckpointPolicy policy_;

    // Helper methods
    void allocateCheckpointMemory();
    void freeCheckpointMemory();
    void* getCheckpointSlot(size_t size);
    void pruneOldCheckpoints(int ubatch_id, int current_step);
    uint64_t getCurrentTimestamp();

public:
    MoETokenFTManager(size_t num_experts,
                      size_t moe_k,
                      size_t hidden_units,
                      size_t checkpoint_interval = 1,
                      size_t max_checkpoints = 100);

    ~MoETokenFTManager();

    // Checkpoint management
    void createCheckpoint(int token_id,
                         int step,
                         int ubatch_id,
                         const int* expert_indices,
                         const float* expert_weights,
                         const void* expert_activations,
                         size_t activation_size,
                         cudaStream_t stream);

    void saveCheckpointAsync(const MoETokenCheckpoint& checkpoint,
                            cudaStream_t stream);

    // Recovery operations
    bool hasCheckpoint(int step, int ubatch_id);

    MoETokenCheckpoint* getCheckpoint(int step, int ubatch_id, int token_id);

    void initiateRecovery(int target_step, int ubatch_id);

    bool restoreFromCheckpoint(int step,
                              int ubatch_id,
                              int token_id,
                              int* expert_indices,
                              float* expert_weights,
                              void* expert_activations,
                              cudaStream_t stream);

    void completeRecovery();

    // State management
    void updateStreamState(int current_step, const std::vector<int>& active_ubatches);

    void markStepComplete(int step, int ubatch_id);

    void clearCheckpoints(int ubatch_id);

    void clearAllCheckpoints();

    // Query methods
    int getCheckpointStep(int ubatch_id) const;

    bool isInRecovery() const { return stream_state_.in_recovery; }

    size_t getCheckpointCount(int ubatch_id) const;

    // Configuration
    void setCheckpointPolicy(CheckpointPolicy policy) { policy_ = policy; }

    void setCheckpointInterval(size_t interval) { checkpoint_interval_ = interval; }

    // Debug and monitoring
    void printCheckpointStats() const;

    size_t getMemoryUsage() const;
};

// Token Stream Recovery Helper
class TokenStreamRecoveryHelper {
private:
    MoETokenFTManager* ft_manager_;

    // Recovery state
    struct RecoveryContext {
        int ubatch_id;
        int target_step;
        int current_recovery_step;
        std::vector<MoETokenCheckpoint> recovery_checkpoints;
    };

    std::vector<RecoveryContext> active_recoveries_;

public:
    TokenStreamRecoveryHelper(MoETokenFTManager* ft_manager);

    // Recovery workflow
    bool startRecovery(int ubatch_id, int target_step);

    bool recoverNextToken(int ubatch_id,
                         int* expert_indices,
                         float* expert_weights,
                         void* expert_activations,
                         cudaStream_t stream);

    bool isRecoveryComplete(int ubatch_id);

    void finalizeRecovery(int ubatch_id);

    void abortRecovery(int ubatch_id);
};

// Adaptive checkpoint decision based on expert activation patterns
class AdaptiveCheckpointPolicy {
private:
    size_t num_experts_;
    size_t moe_k_;

    // Track expert usage distribution
    std::vector<int> expert_usage_count_;
    std::vector<float> expert_usage_entropy_;

    // Thresholds for checkpointing
    float entropy_threshold_;
    float imbalance_threshold_;

public:
    AdaptiveCheckpointPolicy(size_t num_experts, size_t moe_k);

    // Decide whether to checkpoint based on expert distribution
    bool shouldCheckpoint(const int* expert_indices,
                         const float* expert_weights,
                         int num_tokens);

    void updateStatistics(const int* expert_indices, int num_tokens);

    void resetStatistics();

    float calculateEntropy() const;

    float calculateImbalance() const;
};

}  // namespace fastertransformer
