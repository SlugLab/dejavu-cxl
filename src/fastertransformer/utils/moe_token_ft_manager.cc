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

#include "src/fastertransformer/utils/moe_token_ft_manager.h"
#include "src/fastertransformer/utils/logger.h"
#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstring>

namespace fastertransformer {

// MoETokenFTManager Implementation

MoETokenFTManager::MoETokenFTManager(size_t num_experts,
                                     size_t moe_k,
                                     size_t hidden_units,
                                     size_t checkpoint_interval,
                                     size_t max_checkpoints)
    : num_experts_(num_experts),
      moe_k_(moe_k),
      hidden_units_(hidden_units),
      checkpoint_interval_(checkpoint_interval),
      max_checkpoints_(max_checkpoints),
      host_checkpoint_buffer_(nullptr),
      host_buffer_size_(0),
      device_checkpoint_pool_(nullptr),
      device_pool_size_(0),
      device_pool_offset_(0),
      policy_(CHECKPOINT_INTERVAL)
{
    FT_LOG_INFO("Initializing MoE Token FT Manager: experts=%zu, k=%zu, hidden=%zu",
                num_experts_, moe_k_, hidden_units_);

    // Create CUDA streams for asynchronous operations
    cudaStreamCreate(&checkpoint_stream_);
    cudaStreamCreate(&recovery_stream_);

    allocateCheckpointMemory();
}

MoETokenFTManager::~MoETokenFTManager()
{
    freeCheckpointMemory();

    if (checkpoint_stream_ != nullptr) {
        cudaStreamDestroy(checkpoint_stream_);
    }
    if (recovery_stream_ != nullptr) {
        cudaStreamDestroy(recovery_stream_);
    }
}

void MoETokenFTManager::allocateCheckpointMemory()
{
    // Calculate memory requirements
    // Each checkpoint stores: expert indices (k ints) + weights (k floats) + activations
    size_t per_checkpoint_metadata = (moe_k_ * sizeof(int)) + (moe_k_ * sizeof(float));
    size_t per_checkpoint_activation = hidden_units_ * moe_k_ * sizeof(float);

    // Allocate host buffer for metadata
    host_buffer_size_ = max_checkpoints_ * per_checkpoint_metadata * 10; // 10 microbatches
    cudaMallocHost(&host_checkpoint_buffer_, host_buffer_size_);

    // Allocate device pool for activations
    device_pool_size_ = max_checkpoints_ * per_checkpoint_activation * 2; // 2x buffer
    cudaMalloc(&device_checkpoint_pool_, device_pool_size_);

    FT_LOG_INFO("Allocated checkpoint memory: host=%zu MB, device=%zu MB",
                host_buffer_size_ / (1024 * 1024),
                device_pool_size_ / (1024 * 1024));
}

void MoETokenFTManager::freeCheckpointMemory()
{
    if (host_checkpoint_buffer_ != nullptr) {
        cudaFreeHost(host_checkpoint_buffer_);
        host_checkpoint_buffer_ = nullptr;
    }

    if (device_checkpoint_pool_ != nullptr) {
        cudaFree(device_checkpoint_pool_);
        device_checkpoint_pool_ = nullptr;
    }
}

void* MoETokenFTManager::getCheckpointSlot(size_t size)
{
    std::lock_guard<std::mutex> lock(state_mutex_);

    if (device_pool_offset_ + size > device_pool_size_) {
        // Pool is full, wrap around (simple circular buffer)
        device_pool_offset_ = 0;
        FT_LOG_WARNING("Checkpoint pool wrapped around, old checkpoints may be overwritten");
    }

    void* slot = static_cast<char*>(device_checkpoint_pool_) + device_pool_offset_;
    device_pool_offset_ += size;

    return slot;
}

uint64_t MoETokenFTManager::getCurrentTimestamp()
{
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
}

void MoETokenFTManager::createCheckpoint(int token_id,
                                        int step,
                                        int ubatch_id,
                                        const int* expert_indices,
                                        const float* expert_weights,
                                        const void* expert_activations,
                                        size_t activation_size,
                                        cudaStream_t stream)
{
    std::lock_guard<std::mutex> lock(state_mutex_);

    // Check if we should checkpoint based on policy
    if (policy_ == CHECKPOINT_INTERVAL && (step % checkpoint_interval_) != 0) {
        return;
    }

    // Create checkpoint structure
    MoETokenCheckpoint checkpoint;
    checkpoint.token_id = token_id;
    checkpoint.step = step;
    checkpoint.ubatch_id = ubatch_id;
    checkpoint.timestamp = getCurrentTimestamp();
    checkpoint.is_valid = true;

    // Copy expert indices and weights
    checkpoint.expert_indices.resize(moe_k_);
    checkpoint.expert_weights.resize(moe_k_);
    std::memcpy(checkpoint.expert_indices.data(), expert_indices, moe_k_ * sizeof(int));
    std::memcpy(checkpoint.expert_weights.data(), expert_weights, moe_k_ * sizeof(float));

    // Allocate device memory for activations
    checkpoint.activation_size = activation_size;
    checkpoint.expert_activations_ptr = getCheckpointSlot(activation_size);

    // Asynchronously copy activations to checkpoint slot
    cudaMemcpyAsync(checkpoint.expert_activations_ptr,
                   expert_activations,
                   activation_size,
                   cudaMemcpyDeviceToDevice,
                   stream);

    // Store checkpoint
    stream_state_.ubatch_checkpoints[ubatch_id].push_back(checkpoint);

    // Prune old checkpoints if needed
    pruneOldCheckpoints(ubatch_id, step);

    FT_LOG_DEBUG("Created checkpoint: step=%d, ubatch=%d, token=%d, size=%zu",
                step, ubatch_id, token_id, activation_size);
}

void MoETokenFTManager::saveCheckpointAsync(const MoETokenCheckpoint& checkpoint,
                                           cudaStream_t stream)
{
    // Additional async save to persistent storage could be implemented here
    // For now, we keep checkpoints in memory only
    cudaStreamSynchronize(stream);
}

void MoETokenFTManager::pruneOldCheckpoints(int ubatch_id, int current_step)
{
    auto& checkpoints = stream_state_.ubatch_checkpoints[ubatch_id];

    if (checkpoints.size() <= max_checkpoints_) {
        return;
    }

    // Sort by timestamp (oldest first)
    std::sort(checkpoints.begin(), checkpoints.end(),
              [](const MoETokenCheckpoint& a, const MoETokenCheckpoint& b) {
                  return a.timestamp < b.timestamp;
              });

    // Remove oldest checkpoints
    size_t to_remove = checkpoints.size() - max_checkpoints_;
    checkpoints.erase(checkpoints.begin(), checkpoints.begin() + to_remove);

    FT_LOG_DEBUG("Pruned %zu old checkpoints for ubatch %d", to_remove, ubatch_id);
}

bool MoETokenFTManager::hasCheckpoint(int step, int ubatch_id)
{
    std::lock_guard<std::mutex> lock(state_mutex_);

    auto it = stream_state_.ubatch_checkpoints.find(ubatch_id);
    if (it == stream_state_.ubatch_checkpoints.end()) {
        return false;
    }

    for (const auto& cp : it->second) {
        if (cp.step == step && cp.is_valid) {
            return true;
        }
    }

    return false;
}

MoETokenCheckpoint* MoETokenFTManager::getCheckpoint(int step, int ubatch_id, int token_id)
{
    std::lock_guard<std::mutex> lock(state_mutex_);

    auto it = stream_state_.ubatch_checkpoints.find(ubatch_id);
    if (it == stream_state_.ubatch_checkpoints.end()) {
        return nullptr;
    }

    for (auto& cp : it->second) {
        if (cp.step == step && cp.token_id == token_id && cp.is_valid) {
            return &cp;
        }
    }

    return nullptr;
}

void MoETokenFTManager::initiateRecovery(int target_step, int ubatch_id)
{
    std::lock_guard<std::mutex> lock(state_mutex_);

    stream_state_.in_recovery = true;
    stream_state_.recovery_target_step = target_step;

    FT_LOG_INFO("Initiated recovery: target_step=%d, ubatch=%d", target_step, ubatch_id);
}

bool MoETokenFTManager::restoreFromCheckpoint(int step,
                                              int ubatch_id,
                                              int token_id,
                                              int* expert_indices,
                                              float* expert_weights,
                                              void* expert_activations,
                                              cudaStream_t stream)
{
    MoETokenCheckpoint* checkpoint = getCheckpoint(step, ubatch_id, token_id);

    if (checkpoint == nullptr || !checkpoint->is_valid) {
        FT_LOG_ERROR("Cannot restore: checkpoint not found for step=%d, ubatch=%d, token=%d",
                    step, ubatch_id, token_id);
        return false;
    }

    // Restore expert indices and weights
    std::memcpy(expert_indices, checkpoint->expert_indices.data(), moe_k_ * sizeof(int));
    std::memcpy(expert_weights, checkpoint->expert_weights.data(), moe_k_ * sizeof(float));

    // Restore activations
    cudaMemcpyAsync(expert_activations,
                   checkpoint->expert_activations_ptr,
                   checkpoint->activation_size,
                   cudaMemcpyDeviceToDevice,
                   stream);

    cudaStreamSynchronize(stream);

    FT_LOG_INFO("Restored checkpoint: step=%d, ubatch=%d, token=%d", step, ubatch_id, token_id);
    return true;
}

void MoETokenFTManager::completeRecovery()
{
    std::lock_guard<std::mutex> lock(state_mutex_);

    stream_state_.in_recovery = false;
    stream_state_.recovery_target_step = -1;

    FT_LOG_INFO("Recovery completed");
}

void MoETokenFTManager::updateStreamState(int current_step, const std::vector<int>& active_ubatches)
{
    std::lock_guard<std::mutex> lock(state_mutex_);

    stream_state_.current_step = current_step;
    stream_state_.active_ubatch_ids = active_ubatches;
}

void MoETokenFTManager::markStepComplete(int step, int ubatch_id)
{
    std::lock_guard<std::mutex> lock(state_mutex_);

    if (step > stream_state_.checkpoint_step) {
        stream_state_.checkpoint_step = step;
    }
}

void MoETokenFTManager::clearCheckpoints(int ubatch_id)
{
    std::lock_guard<std::mutex> lock(state_mutex_);

    stream_state_.ubatch_checkpoints.erase(ubatch_id);

    FT_LOG_DEBUG("Cleared checkpoints for ubatch %d", ubatch_id);
}

void MoETokenFTManager::clearAllCheckpoints()
{
    std::lock_guard<std::mutex> lock(state_mutex_);

    stream_state_.ubatch_checkpoints.clear();
    device_pool_offset_ = 0;

    FT_LOG_DEBUG("Cleared all checkpoints");
}

int MoETokenFTManager::getCheckpointStep(int ubatch_id) const
{
    return stream_state_.checkpoint_step;
}

size_t MoETokenFTManager::getCheckpointCount(int ubatch_id) const
{
    // Note: Can't lock mutex in const method, so this is not thread-safe
    // Consider making this method non-const or using mutable mutex

    auto it = stream_state_.ubatch_checkpoints.find(ubatch_id);
    if (it == stream_state_.ubatch_checkpoints.end()) {
        return 0;
    }

    return it->second.size();
}

void MoETokenFTManager::printCheckpointStats() const
{
    // Note: Can't lock mutex in const method, so this is not thread-safe

    FT_LOG_INFO("=== MoE Token FT Manager Statistics ===");
    FT_LOG_INFO("Current step: %d", stream_state_.current_step);
    FT_LOG_INFO("Checkpoint step: %d", stream_state_.checkpoint_step);
    FT_LOG_INFO("Active ubatches: %zu", stream_state_.active_ubatch_ids.size());

    size_t total_checkpoints = 0;
    for (const auto& kv : stream_state_.ubatch_checkpoints) {
        FT_LOG_INFO("  Ubatch %d: %zu checkpoints", kv.first, kv.second.size());
        total_checkpoints += kv.second.size();
    }

    FT_LOG_INFO("Total checkpoints: %zu", total_checkpoints);
    FT_LOG_INFO("Device pool usage: %zu / %zu MB",
                device_pool_offset_ / (1024 * 1024),
                device_pool_size_ / (1024 * 1024));
}

size_t MoETokenFTManager::getMemoryUsage() const
{
    return host_buffer_size_ + device_pool_offset_;
}

// TokenStreamRecoveryHelper Implementation

TokenStreamRecoveryHelper::TokenStreamRecoveryHelper(MoETokenFTManager* ft_manager)
    : ft_manager_(ft_manager)
{
}

bool TokenStreamRecoveryHelper::startRecovery(int ubatch_id, int target_step)
{
    if (!ft_manager_->hasCheckpoint(target_step, ubatch_id)) {
        FT_LOG_ERROR("Cannot start recovery: no checkpoint for step=%d, ubatch=%d",
                    target_step, ubatch_id);
        return false;
    }

    RecoveryContext ctx;
    ctx.ubatch_id = ubatch_id;
    ctx.target_step = target_step;
    ctx.current_recovery_step = 0;

    active_recoveries_.push_back(ctx);

    ft_manager_->initiateRecovery(target_step, ubatch_id);

    FT_LOG_INFO("Started recovery for ubatch %d to step %d", ubatch_id, target_step);
    return true;
}

bool TokenStreamRecoveryHelper::recoverNextToken(int ubatch_id,
                                                int* expert_indices,
                                                float* expert_weights,
                                                void* expert_activations,
                                                cudaStream_t stream)
{
    // Find active recovery for this ubatch
    auto it = std::find_if(active_recoveries_.begin(), active_recoveries_.end(),
                          [ubatch_id](const RecoveryContext& ctx) {
                              return ctx.ubatch_id == ubatch_id;
                          });

    if (it == active_recoveries_.end()) {
        return false;
    }

    int token_id = it->current_recovery_step;
    bool success = ft_manager_->restoreFromCheckpoint(it->target_step,
                                                      ubatch_id,
                                                      token_id,
                                                      expert_indices,
                                                      expert_weights,
                                                      expert_activations,
                                                      stream);

    if (success) {
        it->current_recovery_step++;
    }

    return success;
}

bool TokenStreamRecoveryHelper::isRecoveryComplete(int ubatch_id)
{
    auto it = std::find_if(active_recoveries_.begin(), active_recoveries_.end(),
                          [ubatch_id](const RecoveryContext& ctx) {
                              return ctx.ubatch_id == ubatch_id;
                          });

    if (it == active_recoveries_.end()) {
        return true;
    }

    // Recovery is complete when we've recovered all checkpoints up to target step
    return it->current_recovery_step >= it->target_step;
}

void TokenStreamRecoveryHelper::finalizeRecovery(int ubatch_id)
{
    auto it = std::find_if(active_recoveries_.begin(), active_recoveries_.end(),
                          [ubatch_id](const RecoveryContext& ctx) {
                              return ctx.ubatch_id == ubatch_id;
                          });

    if (it != active_recoveries_.end()) {
        active_recoveries_.erase(it);
    }

    ft_manager_->completeRecovery();

    FT_LOG_INFO("Finalized recovery for ubatch %d", ubatch_id);
}

void TokenStreamRecoveryHelper::abortRecovery(int ubatch_id)
{
    auto it = std::find_if(active_recoveries_.begin(), active_recoveries_.end(),
                          [ubatch_id](const RecoveryContext& ctx) {
                              return ctx.ubatch_id == ubatch_id;
                          });

    if (it != active_recoveries_.end()) {
        active_recoveries_.erase(it);
    }

    FT_LOG_WARNING("Aborted recovery for ubatch %d", ubatch_id);
}

// AdaptiveCheckpointPolicy Implementation

AdaptiveCheckpointPolicy::AdaptiveCheckpointPolicy(size_t num_experts, size_t moe_k)
    : num_experts_(num_experts),
      moe_k_(moe_k),
      entropy_threshold_(0.8f),
      imbalance_threshold_(0.3f)
{
    expert_usage_count_.resize(num_experts_, 0);
    expert_usage_entropy_.resize(num_experts_, 0.0f);
}

bool AdaptiveCheckpointPolicy::shouldCheckpoint(const int* expert_indices,
                                               const float* expert_weights,
                                               int num_tokens)
{
    updateStatistics(expert_indices, num_tokens);

    float entropy = calculateEntropy();
    float imbalance = calculateImbalance();

    // Checkpoint if:
    // 1. Entropy is high (diverse expert usage) - important state
    // 2. Imbalance is high (skewed expert usage) - potential failure point
    return (entropy > entropy_threshold_) || (imbalance > imbalance_threshold_);
}

void AdaptiveCheckpointPolicy::updateStatistics(const int* expert_indices, int num_tokens)
{
    for (int i = 0; i < num_tokens * moe_k_; ++i) {
        int expert_idx = expert_indices[i];
        if (expert_idx >= 0 && expert_idx < num_experts_) {
            expert_usage_count_[expert_idx]++;
        }
    }
}

void AdaptiveCheckpointPolicy::resetStatistics()
{
    std::fill(expert_usage_count_.begin(), expert_usage_count_.end(), 0);
    std::fill(expert_usage_entropy_.begin(), expert_usage_entropy_.end(), 0.0f);
}

float AdaptiveCheckpointPolicy::calculateEntropy() const
{
    int total_usage = 0;
    for (int count : expert_usage_count_) {
        total_usage += count;
    }

    if (total_usage == 0) {
        return 0.0f;
    }

    float entropy = 0.0f;
    for (int count : expert_usage_count_) {
        if (count > 0) {
            float p = static_cast<float>(count) / total_usage;
            entropy -= p * std::log2(p);
        }
    }

    // Normalize by max entropy
    float max_entropy = std::log2(static_cast<float>(num_experts_));
    return entropy / max_entropy;
}

float AdaptiveCheckpointPolicy::calculateImbalance() const
{
    int total_usage = 0;
    int max_usage = 0;

    for (int count : expert_usage_count_) {
        total_usage += count;
        max_usage = std::max(max_usage, count);
    }

    if (total_usage == 0) {
        return 0.0f;
    }

    float avg_usage = static_cast<float>(total_usage) / num_experts_;
    float imbalance = (max_usage - avg_usage) / total_usage;

    return imbalance;
}

}  // namespace fastertransformer
