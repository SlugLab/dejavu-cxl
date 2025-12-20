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
#include <atomic>
#include <queue>
#include "src/fastertransformer/utils/cuda_utils.h"

namespace fastertransformer {

// Delta checkpoint entry - stores only the changes since last checkpoint
struct MoEDeltaCheckpoint {
    int step;                               // Generation step
    int layer_id;                           // Layer index
    int ubatch_id;                          // Microbatch ID

    // Expert routing delta
    std::vector<int> expert_indices;        // Changed expert indices
    std::vector<float> expert_weights;      // Changed gating weights

    // KV cache delta (only new tokens)
    void* key_cache_delta;                  // Pointer to delta key cache
    void* value_cache_delta;                // Pointer to delta value cache
    size_t kv_delta_size;                   // Size of KV delta in bytes

    // Expert activation delta
    void* activation_delta;                 // Pointer to delta activations
    size_t activation_delta_size;           // Size in bytes

    // Metadata
    int num_tokens;                         // Number of tokens in this delta
    int seq_offset;                         // Sequence position offset
    uint64_t timestamp;                     // Creation timestamp
    bool is_valid;

    // For linked list of deltas
    MoEDeltaCheckpoint* next;
    MoEDeltaCheckpoint* prev;

    MoEDeltaCheckpoint()
        : step(-1), layer_id(-1), ubatch_id(-1),
          key_cache_delta(nullptr), value_cache_delta(nullptr), kv_delta_size(0),
          activation_delta(nullptr), activation_delta_size(0),
          num_tokens(0), seq_offset(0), timestamp(0), is_valid(false),
          next(nullptr), prev(nullptr) {}
};

// Configuration for delta checkpointing
struct DeltaCheckpointConfig {
    size_t num_layers;
    size_t num_experts;
    size_t moe_k;
    size_t hidden_units;
    size_t size_per_head;
    size_t num_heads;
    size_t max_seq_len;
    size_t max_batch_size;

    // Checkpoint policy settings
    int checkpoint_interval;                // Checkpoint every N steps
    int max_checkpoints_per_ubatch;         // Max checkpoints to keep per microbatch
    size_t device_pool_size_mb;             // Device memory pool size in MB
    size_t host_buffer_size_mb;             // Host buffer size in MB

    // Delta settings
    bool enable_kv_delta;                   // Enable KV cache delta checkpointing
    bool enable_activation_delta;           // Enable activation delta checkpointing
    bool enable_compression;                // Enable delta compression

    DeltaCheckpointConfig()
        : num_layers(48), num_experts(128), moe_k(8),
          hidden_units(2048), size_per_head(128), num_heads(32),
          max_seq_len(4096), max_batch_size(32),
          checkpoint_interval(1), max_checkpoints_per_ubatch(100),
          device_pool_size_mb(512), host_buffer_size_mb(128),
          enable_kv_delta(true), enable_activation_delta(true),
          enable_compression(false) {}
};

// Ring buffer for efficient delta storage
class DeltaRingBuffer {
private:
    void* buffer_;
    size_t capacity_;
    size_t head_;
    size_t tail_;
    std::mutex mutex_;

public:
    DeltaRingBuffer(size_t capacity);
    ~DeltaRingBuffer();

    void* allocate(size_t size);
    void free(void* ptr, size_t size);
    size_t getUsedSize() const { return (head_ >= tail_) ? (head_ - tail_) : (capacity_ - tail_ + head_); }
    size_t getFreeSize() const { return capacity_ - getUsedSize(); }
    void reset();
};

// MoE Delta Checkpoint Manager
class MoEDeltaCheckpointManager {
private:
    DeltaCheckpointConfig config_;

    // Memory pools
    DeltaRingBuffer* device_pool_;
    DeltaRingBuffer* host_pool_;

    // Checkpoint storage per layer per ubatch
    // ubatch_id -> layer_id -> list of delta checkpoints
    std::unordered_map<int, std::unordered_map<int, MoEDeltaCheckpoint*>> checkpoints_;
    std::mutex checkpoint_mutex_;

    // CUDA streams for async operations
    cudaStream_t checkpoint_stream_;
    cudaStream_t recovery_stream_;

    // State tracking
    std::unordered_map<int, int> last_checkpoint_step_;  // ubatch_id -> last step
    std::atomic<bool> in_recovery_;
    int recovery_target_step_;

    // Statistics
    std::atomic<size_t> total_checkpoints_;
    std::atomic<size_t> total_bytes_checkpointed_;
    std::atomic<size_t> total_recoveries_;

    // Helper methods
    uint64_t getCurrentTimestamp();
    void pruneOldCheckpoints(int ubatch_id, int layer_id);
    MoEDeltaCheckpoint* allocateCheckpoint();
    void freeCheckpoint(MoEDeltaCheckpoint* checkpoint);

public:
    MoEDeltaCheckpointManager(const DeltaCheckpointConfig& config);
    ~MoEDeltaCheckpointManager();

    // Initialize/cleanup
    void initialize();
    void cleanup();

    // Delta checkpoint creation
    // Only saves the delta (new tokens) since last checkpoint
    void createKVCacheDelta(
        int step,
        int layer_id,
        int ubatch_id,
        const void* key_cache,           // Full key cache
        const void* value_cache,          // Full value cache
        int seq_start,                    // Start position of new tokens
        int seq_end,                      // End position (exclusive)
        int batch_size,
        cudaStream_t stream);

    void createActivationDelta(
        int step,
        int layer_id,
        int ubatch_id,
        const int* expert_indices,        // Expert routing for new tokens
        const float* expert_weights,      // Gating weights for new tokens
        const void* activations,          // Expert activations
        int num_tokens,
        cudaStream_t stream);

    // Combined delta checkpoint (KV + activations)
    void createDeltaCheckpoint(
        int step,
        int layer_id,
        int ubatch_id,
        const void* key_cache,
        const void* value_cache,
        int seq_start,
        int seq_end,
        const int* expert_indices,
        const float* expert_weights,
        const void* activations,
        int num_tokens,
        int batch_size,
        cudaStream_t stream);

    // Recovery operations
    bool initiateRecovery(int target_step, int ubatch_id);

    // Reconstruct full state from deltas
    bool reconstructKVCache(
        int target_step,
        int layer_id,
        int ubatch_id,
        void* key_cache,                  // Output: reconstructed key cache
        void* value_cache,                // Output: reconstructed value cache
        int* seq_len,                     // Output: sequence length
        cudaStream_t stream);

    bool reconstructActivations(
        int target_step,
        int layer_id,
        int ubatch_id,
        int* expert_indices,              // Output: expert routing
        float* expert_weights,            // Output: gating weights
        void* activations,                // Output: reconstructed activations
        cudaStream_t stream);

    void completeRecovery();
    void abortRecovery();

    // Checkpoint management
    bool hasCheckpoint(int step, int layer_id, int ubatch_id);
    int getLatestCheckpointStep(int ubatch_id);
    void clearCheckpoints(int ubatch_id);
    void clearLayerCheckpoints(int ubatch_id, int layer_id);
    void clearAllCheckpoints();

    // Step notification (for knowing when to checkpoint)
    void onStepComplete(int step, int ubatch_id);
    bool shouldCheckpoint(int step) const;

    // Query methods
    bool isInRecovery() const { return in_recovery_.load(); }
    size_t getTotalCheckpoints() const { return total_checkpoints_.load(); }
    size_t getTotalBytesCheckpointed() const { return total_bytes_checkpointed_.load(); }
    size_t getMemoryUsage() const;

    // Debug and monitoring
    void printStats() const;
    void dumpCheckpointChain(int ubatch_id, int layer_id) const;
};

// Integration helper for FFN layer
class MoECheckpointHelper {
private:
    MoEDeltaCheckpointManager* manager_;
    int current_layer_;
    bool enabled_;

public:
    MoECheckpointHelper(MoEDeltaCheckpointManager* manager = nullptr);

    void setLayer(int layer_id) { current_layer_ = layer_id; }
    void setEnabled(bool enabled) { enabled_ = enabled; }
    bool isEnabled() const { return enabled_ && manager_ != nullptr; }

    // Called after MoE computation to checkpoint results
    void checkpointMoEOutput(
        int step,
        int ubatch_id,
        const void* key_cache,
        const void* value_cache,
        int seq_start,
        int seq_end,
        const int* expert_indices,
        const float* expert_weights,
        const void* moe_output,
        int num_tokens,
        int batch_size,
        cudaStream_t stream);

    // Called during recovery to restore MoE state
    bool restoreMoEState(
        int target_step,
        int ubatch_id,
        void* key_cache,
        void* value_cache,
        int* expert_indices,
        float* expert_weights,
        void* moe_output,
        cudaStream_t stream);
};

}  // namespace fastertransformer
