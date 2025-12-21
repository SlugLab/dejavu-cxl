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

#include "src/fastertransformer/utils/moe_delta_checkpoint.h"
#include "src/fastertransformer/utils/logger.h"
#include <chrono>
#include <algorithm>
#include <cstring>
#include <climits>
#include <sys/stat.h>

namespace fastertransformer {

// DeltaRingBuffer Implementation

DeltaRingBuffer::DeltaRingBuffer(size_t capacity)
    : buffer_(nullptr), capacity_(capacity), head_(0), tail_(0)
{
    cudaMalloc(&buffer_, capacity);
    FT_LOG_INFO("DeltaRingBuffer: allocated %zu MB", capacity / (1024 * 1024));
}

DeltaRingBuffer::~DeltaRingBuffer()
{
    if (buffer_ != nullptr) {
        cudaFree(buffer_);
        buffer_ = nullptr;
    }
}

void* DeltaRingBuffer::allocate(size_t size)
{
    std::lock_guard<std::mutex> lock(mutex_);

    // Align size to 256 bytes for efficient GPU access
    size = (size + 255) & ~255;

    if (size > getFreeSize()) {
        FT_LOG_WARNING("DeltaRingBuffer: not enough space, wrapping around");
        // Reset buffer (losing old data)
        head_ = 0;
        tail_ = 0;
    }

    void* ptr = static_cast<char*>(buffer_) + head_;
    head_ = (head_ + size) % capacity_;

    return ptr;
}

void DeltaRingBuffer::free(void* ptr, size_t size)
{
    std::lock_guard<std::mutex> lock(mutex_);

    // Simple ring buffer - we don't actually free individual allocations
    // Just advance tail when safe
    size = (size + 255) & ~255;

    // Only advance tail if this is the oldest allocation
    char* base = static_cast<char*>(buffer_);
    char* p = static_cast<char*>(ptr);
    size_t offset = p - base;

    if (offset == tail_) {
        tail_ = (tail_ + size) % capacity_;
    }
}

void DeltaRingBuffer::reset()
{
    std::lock_guard<std::mutex> lock(mutex_);
    head_ = 0;
    tail_ = 0;
}

// MoEDeltaCheckpointManager Implementation

MoEDeltaCheckpointManager::MoEDeltaCheckpointManager(const DeltaCheckpointConfig& config)
    : config_(config),
      device_pool_(nullptr),
      host_pool_(nullptr),
      checkpoint_stream_(nullptr),
      recovery_stream_(nullptr),
      in_recovery_(false),
      recovery_target_step_(-1),
      total_checkpoints_(0),
      total_bytes_checkpointed_(0),
      total_recoveries_(0)
{
    initialize();
}

MoEDeltaCheckpointManager::~MoEDeltaCheckpointManager()
{
    cleanup();
}

void MoEDeltaCheckpointManager::initialize()
{
    FT_LOG_INFO("Initializing MoE Delta Checkpoint Manager");

    // Create memory pools
    size_t device_pool_bytes = config_.device_pool_size_mb * 1024 * 1024;
    size_t host_pool_bytes = config_.host_buffer_size_mb * 1024 * 1024;

    device_pool_ = new DeltaRingBuffer(device_pool_bytes);

    // Host pool uses pinned memory
    void* host_buffer = nullptr;
    cudaMallocHost(&host_buffer, host_pool_bytes);
    // Note: host_pool_ would need different implementation for host memory
    // For now, we use device pool only

    // Create CUDA streams
    cudaStreamCreate(&checkpoint_stream_);
    cudaStreamCreate(&recovery_stream_);

    FT_LOG_INFO("MoE Delta Checkpoint Manager initialized: device_pool=%zu MB",
                device_pool_bytes / (1024 * 1024));
}

void MoEDeltaCheckpointManager::cleanup()
{
    FT_LOG_INFO("Cleaning up MoE Delta Checkpoint Manager");

    clearAllCheckpoints();

    if (device_pool_ != nullptr) {
        delete device_pool_;
        device_pool_ = nullptr;
    }

    if (checkpoint_stream_ != nullptr) {
        cudaStreamDestroy(checkpoint_stream_);
        checkpoint_stream_ = nullptr;
    }

    if (recovery_stream_ != nullptr) {
        cudaStreamDestroy(recovery_stream_);
        recovery_stream_ = nullptr;
    }
}

uint64_t MoEDeltaCheckpointManager::getCurrentTimestamp()
{
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
}

MoEDeltaCheckpoint* MoEDeltaCheckpointManager::allocateCheckpoint()
{
    return new MoEDeltaCheckpoint();
}

void MoEDeltaCheckpointManager::freeCheckpoint(MoEDeltaCheckpoint* checkpoint)
{
    if (checkpoint == nullptr) return;

    // Free device memory allocations
    if (checkpoint->key_cache_delta != nullptr) {
        device_pool_->free(checkpoint->key_cache_delta, checkpoint->kv_delta_size);
    }
    if (checkpoint->value_cache_delta != nullptr) {
        device_pool_->free(checkpoint->value_cache_delta, checkpoint->kv_delta_size);
    }
    if (checkpoint->activation_delta != nullptr) {
        device_pool_->free(checkpoint->activation_delta, checkpoint->activation_delta_size);
    }

    delete checkpoint;
}

void MoEDeltaCheckpointManager::pruneOldCheckpoints(int ubatch_id, int layer_id)
{
    auto& layer_checkpoints = checkpoints_[ubatch_id][layer_id];

    if (layer_checkpoints == nullptr) return;

    // Count checkpoints
    int count = 0;
    MoEDeltaCheckpoint* curr = layer_checkpoints;
    while (curr != nullptr) {
        count++;
        curr = curr->next;
    }

    // Remove oldest if over limit
    while (count > config_.max_checkpoints_per_ubatch && layer_checkpoints != nullptr) {
        MoEDeltaCheckpoint* oldest = layer_checkpoints;

        // Find tail (oldest)
        while (oldest->next != nullptr) {
            oldest = oldest->next;
        }

        // Remove from list
        if (oldest->prev != nullptr) {
            oldest->prev->next = nullptr;
        } else {
            layer_checkpoints = nullptr;
        }

        freeCheckpoint(oldest);
        count--;
    }
}

void MoEDeltaCheckpointManager::createKVCacheDelta(
    int step,
    int layer_id,
    int ubatch_id,
    const void* key_cache,
    const void* value_cache,
    int seq_start,
    int seq_end,
    int batch_size,
    cudaStream_t stream)
{
    if (!config_.enable_kv_delta) return;

    // Validate inputs
    if (key_cache == nullptr || value_cache == nullptr) {
        printf("[FT][Checkpoint] KV cache is null, skipping checkpoint\n");
        return;
    }
    if (seq_start < 0) {
        printf("[FT][Checkpoint] Invalid seq_start=%d, skipping checkpoint\n", seq_start);
        return;
    }

    std::lock_guard<std::mutex> lock(checkpoint_mutex_);

    int num_new_tokens = seq_end - seq_start;
    if (num_new_tokens <= 0) return;

    // Calculate delta size
    // KV cache format: [batch, heads, seq_len, size_per_head]
    size_t per_token_size = batch_size * config_.num_heads * config_.size_per_head * sizeof(half);
    size_t delta_size = num_new_tokens * per_token_size;

    // Allocate checkpoint
    MoEDeltaCheckpoint* checkpoint = allocateCheckpoint();
    checkpoint->step = step;
    checkpoint->layer_id = layer_id;
    checkpoint->ubatch_id = ubatch_id;
    checkpoint->num_tokens = num_new_tokens;
    checkpoint->seq_offset = seq_start;
    checkpoint->timestamp = getCurrentTimestamp();
    checkpoint->is_valid = true;
    checkpoint->kv_delta_size = delta_size;

    // Allocate device memory for delta
    checkpoint->key_cache_delta = device_pool_->allocate(delta_size);
    checkpoint->value_cache_delta = device_pool_->allocate(delta_size);

    if (checkpoint->key_cache_delta == nullptr || checkpoint->value_cache_delta == nullptr) {
        printf("[FT][Checkpoint] Failed to allocate memory for checkpoint, skipping\n");
        freeCheckpoint(checkpoint);
        return;
    }

    // For token decoder, the KV cache pointer already points to the current position
    // so we don't need to add an offset - just copy from the start
    // Note: The caller passes getPtrWithOffset which already includes the cache_offset
    printf("[FT][Checkpoint] KV delta: step=%d, layer=%d, seq_start=%d, seq_end=%d, tokens=%d, size=%zu\n",
           step, layer_id, seq_start, seq_end, num_new_tokens, delta_size);

    // Copy delta from cache to checkpoint (no offset needed, pointer is already positioned)
    cudaMemcpyAsync(checkpoint->key_cache_delta,
                   key_cache,
                   delta_size,
                   cudaMemcpyDeviceToDevice,
                   stream);

    cudaMemcpyAsync(checkpoint->value_cache_delta,
                   value_cache,
                   delta_size,
                   cudaMemcpyDeviceToDevice,
                   stream);

    // Add to checkpoint list (prepend - newest first)
    MoEDeltaCheckpoint*& head = checkpoints_[ubatch_id][layer_id];
    checkpoint->next = head;
    if (head != nullptr) {
        head->prev = checkpoint;
    }
    head = checkpoint;

    // Update stats
    total_checkpoints_++;
    total_bytes_checkpointed_ += 2 * delta_size;

    // Update last checkpoint step for this ubatch
    if (last_checkpoint_step_.find(ubatch_id) == last_checkpoint_step_.end() ||
        last_checkpoint_step_[ubatch_id] < step) {
        last_checkpoint_step_[ubatch_id] = step;
    }

    // Prune old checkpoints
    pruneOldCheckpoints(ubatch_id, layer_id);

    printf("[FT][Checkpoint] KV checkpoint created: step=%d, layer=%d, ubatch=%d, last_step=%d\n",
           step, layer_id, ubatch_id, last_checkpoint_step_[ubatch_id]);
}

void MoEDeltaCheckpointManager::createActivationDelta(
    int step,
    int layer_id,
    int ubatch_id,
    const int* expert_indices,
    const float* expert_weights,
    const void* activations,
    int num_tokens,
    cudaStream_t stream)
{
    if (!config_.enable_activation_delta) return;

    // For now, skip activation delta to simplify debugging
    // The KV cache delta is the most important for recovery
    printf("[FT][Checkpoint] Activation delta: step=%d, layer=%d, tokens=%d (skipped for now)\n",
           step, layer_id, num_tokens);
    return;

    std::lock_guard<std::mutex> lock(checkpoint_mutex_);

    if (num_tokens <= 0) return;

    // Calculate activation delta size
    // Activations: [num_tokens, moe_k, hidden_units]
    size_t activation_size = num_tokens * config_.moe_k * config_.hidden_units * sizeof(half);

    // Find or create checkpoint for this step
    MoEDeltaCheckpoint* checkpoint = nullptr;
    MoEDeltaCheckpoint* curr = checkpoints_[ubatch_id][layer_id];
    while (curr != nullptr) {
        if (curr->step == step) {
            checkpoint = curr;
            break;
        }
        curr = curr->next;
    }

    if (checkpoint == nullptr) {
        // Create new checkpoint
        checkpoint = allocateCheckpoint();
        checkpoint->step = step;
        checkpoint->layer_id = layer_id;
        checkpoint->ubatch_id = ubatch_id;
        checkpoint->num_tokens = num_tokens;
        checkpoint->timestamp = getCurrentTimestamp();
        checkpoint->is_valid = true;

        // Add to list
        MoEDeltaCheckpoint*& head = checkpoints_[ubatch_id][layer_id];
        checkpoint->next = head;
        if (head != nullptr) {
            head->prev = checkpoint;
        }
        head = checkpoint;
    }

    // Store expert routing info
    checkpoint->expert_indices.resize(num_tokens * config_.moe_k);
    checkpoint->expert_weights.resize(num_tokens * config_.moe_k);
    std::memcpy(checkpoint->expert_indices.data(), expert_indices,
                num_tokens * config_.moe_k * sizeof(int));
    std::memcpy(checkpoint->expert_weights.data(), expert_weights,
                num_tokens * config_.moe_k * sizeof(float));

    // Allocate and copy activations
    checkpoint->activation_delta = device_pool_->allocate(activation_size);
    checkpoint->activation_delta_size = activation_size;

    cudaMemcpyAsync(checkpoint->activation_delta,
                   activations,
                   activation_size,
                   cudaMemcpyDeviceToDevice,
                   stream);

    total_bytes_checkpointed_ += activation_size;

    FT_LOG_DEBUG("Created activation delta: step=%d, layer=%d, ubatch=%d, tokens=%d",
                step, layer_id, ubatch_id, num_tokens);
}

void MoEDeltaCheckpointManager::createDeltaCheckpoint(
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
    cudaStream_t stream)
{
    createKVCacheDelta(step, layer_id, ubatch_id, key_cache, value_cache,
                      seq_start, seq_end, batch_size, stream);
    createActivationDelta(step, layer_id, ubatch_id, expert_indices, expert_weights,
                         activations, num_tokens, stream);
}

bool MoEDeltaCheckpointManager::initiateRecovery(int target_step, int ubatch_id)
{
    std::lock_guard<std::mutex> lock(checkpoint_mutex_);

    // Check if we have checkpoints for this ubatch
    auto it = checkpoints_.find(ubatch_id);
    if (it == checkpoints_.end()) {
        FT_LOG_ERROR("No checkpoints found for ubatch %d", ubatch_id);
        return false;
    }

    // Check if target_step is reachable
    int latest_step = getLatestCheckpointStep(ubatch_id);
    if (target_step > latest_step) {
        FT_LOG_ERROR("Target step %d is beyond latest checkpoint %d", target_step, latest_step);
        return false;
    }

    in_recovery_ = true;
    recovery_target_step_ = target_step;

    FT_LOG_INFO("Initiated recovery: target_step=%d, ubatch=%d", target_step, ubatch_id);
    return true;
}

bool MoEDeltaCheckpointManager::reconstructKVCache(
    int target_step,
    int layer_id,
    int ubatch_id,
    void* key_cache,
    void* value_cache,
    int* seq_len,
    cudaStream_t stream)
{
    std::lock_guard<std::mutex> lock(checkpoint_mutex_);

    auto it = checkpoints_.find(ubatch_id);
    if (it == checkpoints_.end()) {
        return false;
    }

    auto layer_it = it->second.find(layer_id);
    if (layer_it == it->second.end()) {
        return false;
    }

    // Collect all checkpoints up to target_step, sorted by step
    std::vector<MoEDeltaCheckpoint*> deltas;
    MoEDeltaCheckpoint* curr = layer_it->second;
    while (curr != nullptr) {
        if (curr->step <= target_step && curr->is_valid && curr->key_cache_delta != nullptr) {
            deltas.push_back(curr);
        }
        curr = curr->next;
    }

    // Sort by step (ascending)
    std::sort(deltas.begin(), deltas.end(),
              [](MoEDeltaCheckpoint* a, MoEDeltaCheckpoint* b) {
                  return a->step < b->step;
              });

    // Apply deltas sequentially to reconstruct KV cache
    int total_seq_len = 0;
    for (auto* delta : deltas) {
        // Calculate destination offset
        size_t dst_offset = delta->seq_offset * delta->kv_delta_size / delta->num_tokens;

        // Copy key cache delta
        cudaMemcpyAsync(static_cast<char*>(key_cache) + dst_offset,
                       delta->key_cache_delta,
                       delta->kv_delta_size,
                       cudaMemcpyDeviceToDevice,
                       stream);

        // Copy value cache delta
        cudaMemcpyAsync(static_cast<char*>(value_cache) + dst_offset,
                       delta->value_cache_delta,
                       delta->kv_delta_size,
                       cudaMemcpyDeviceToDevice,
                       stream);

        total_seq_len = std::max(total_seq_len, delta->seq_offset + delta->num_tokens);
    }

    cudaStreamSynchronize(stream);

    if (seq_len != nullptr) {
        *seq_len = total_seq_len;
    }

    total_recoveries_++;
    FT_LOG_INFO("Reconstructed KV cache: layer=%d, ubatch=%d, seq_len=%d, deltas=%zu",
               layer_id, ubatch_id, total_seq_len, deltas.size());

    return true;
}

bool MoEDeltaCheckpointManager::reconstructActivations(
    int target_step,
    int layer_id,
    int ubatch_id,
    int* expert_indices,
    float* expert_weights,
    void* activations,
    cudaStream_t stream)
{
    std::lock_guard<std::mutex> lock(checkpoint_mutex_);

    // Find the exact checkpoint for target_step
    MoEDeltaCheckpoint* checkpoint = nullptr;
    MoEDeltaCheckpoint* curr = checkpoints_[ubatch_id][layer_id];
    while (curr != nullptr) {
        if (curr->step == target_step && curr->is_valid) {
            checkpoint = curr;
            break;
        }
        curr = curr->next;
    }

    if (checkpoint == nullptr) {
        FT_LOG_ERROR("Checkpoint not found for step=%d, layer=%d", target_step, layer_id);
        return false;
    }

    // Restore expert routing
    if (!checkpoint->expert_indices.empty() && expert_indices != nullptr) {
        std::memcpy(expert_indices, checkpoint->expert_indices.data(),
                   checkpoint->expert_indices.size() * sizeof(int));
    }

    if (!checkpoint->expert_weights.empty() && expert_weights != nullptr) {
        std::memcpy(expert_weights, checkpoint->expert_weights.data(),
                   checkpoint->expert_weights.size() * sizeof(float));
    }

    // Restore activations
    if (checkpoint->activation_delta != nullptr && activations != nullptr) {
        cudaMemcpyAsync(activations,
                       checkpoint->activation_delta,
                       checkpoint->activation_delta_size,
                       cudaMemcpyDeviceToDevice,
                       stream);
        cudaStreamSynchronize(stream);
    }

    FT_LOG_INFO("Restored activations: step=%d, layer=%d, ubatch=%d",
               target_step, layer_id, ubatch_id);

    return true;
}

void MoEDeltaCheckpointManager::completeRecovery()
{
    in_recovery_ = false;
    recovery_target_step_ = -1;
    FT_LOG_INFO("Recovery completed");
}

void MoEDeltaCheckpointManager::abortRecovery()
{
    in_recovery_ = false;
    recovery_target_step_ = -1;
    FT_LOG_WARNING("Recovery aborted");
}

bool MoEDeltaCheckpointManager::hasCheckpoint(int step, int layer_id, int ubatch_id)
{
    std::lock_guard<std::mutex> lock(checkpoint_mutex_);

    auto it = checkpoints_.find(ubatch_id);
    if (it == checkpoints_.end()) return false;

    auto layer_it = it->second.find(layer_id);
    if (layer_it == it->second.end()) return false;

    MoEDeltaCheckpoint* curr = layer_it->second;
    while (curr != nullptr) {
        if (curr->step == step && curr->is_valid) {
            return true;
        }
        curr = curr->next;
    }

    return false;
}

int MoEDeltaCheckpointManager::getLatestCheckpointStep(int ubatch_id)
{
    auto it = last_checkpoint_step_.find(ubatch_id);
    if (it != last_checkpoint_step_.end()) {
        return it->second;
    }

    // Search through checkpoints
    int latest = -1;
    auto ub_it = checkpoints_.find(ubatch_id);
    if (ub_it != checkpoints_.end()) {
        for (auto& layer_pair : ub_it->second) {
            MoEDeltaCheckpoint* curr = layer_pair.second;
            while (curr != nullptr) {
                if (curr->step > latest && curr->is_valid) {
                    latest = curr->step;
                }
                curr = curr->next;
            }
        }
    }

    return latest;
}

void MoEDeltaCheckpointManager::clearCheckpoints(int ubatch_id)
{
    std::lock_guard<std::mutex> lock(checkpoint_mutex_);

    auto it = checkpoints_.find(ubatch_id);
    if (it != checkpoints_.end()) {
        for (auto& layer_pair : it->second) {
            MoEDeltaCheckpoint* curr = layer_pair.second;
            while (curr != nullptr) {
                MoEDeltaCheckpoint* next = curr->next;
                freeCheckpoint(curr);
                curr = next;
            }
        }
        checkpoints_.erase(it);
    }

    last_checkpoint_step_.erase(ubatch_id);
}

void MoEDeltaCheckpointManager::clearLayerCheckpoints(int ubatch_id, int layer_id)
{
    std::lock_guard<std::mutex> lock(checkpoint_mutex_);

    auto it = checkpoints_.find(ubatch_id);
    if (it != checkpoints_.end()) {
        auto layer_it = it->second.find(layer_id);
        if (layer_it != it->second.end()) {
            MoEDeltaCheckpoint* curr = layer_it->second;
            while (curr != nullptr) {
                MoEDeltaCheckpoint* next = curr->next;
                freeCheckpoint(curr);
                curr = next;
            }
            it->second.erase(layer_it);
        }
    }
}

void MoEDeltaCheckpointManager::clearAllCheckpoints()
{
    std::lock_guard<std::mutex> lock(checkpoint_mutex_);

    for (auto& ubatch_pair : checkpoints_) {
        for (auto& layer_pair : ubatch_pair.second) {
            MoEDeltaCheckpoint* curr = layer_pair.second;
            while (curr != nullptr) {
                MoEDeltaCheckpoint* next = curr->next;
                freeCheckpoint(curr);
                curr = next;
            }
        }
    }

    checkpoints_.clear();
    last_checkpoint_step_.clear();
    device_pool_->reset();
}

void MoEDeltaCheckpointManager::onStepComplete(int step, int ubatch_id)
{
    last_checkpoint_step_[ubatch_id] = step;
}

bool MoEDeltaCheckpointManager::shouldCheckpoint(int step) const
{
    return (step % config_.checkpoint_interval) == 0;
}

size_t MoEDeltaCheckpointManager::getMemoryUsage() const
{
    return device_pool_->getUsedSize();
}

void MoEDeltaCheckpointManager::printStats() const
{
    FT_LOG_INFO("=== MoE Delta Checkpoint Statistics ===");
    FT_LOG_INFO("Total checkpoints: %zu", total_checkpoints_.load());
    FT_LOG_INFO("Total bytes checkpointed: %zu MB", total_bytes_checkpointed_.load() / (1024 * 1024));
    FT_LOG_INFO("Total recoveries: %zu", total_recoveries_.load());
    FT_LOG_INFO("Memory usage: %zu MB", getMemoryUsage() / (1024 * 1024));
    FT_LOG_INFO("In recovery: %s", in_recovery_.load() ? "yes" : "no");
}

void MoEDeltaCheckpointManager::dumpCheckpointChain(int ubatch_id, int layer_id) const
{
    auto ub_it = checkpoints_.find(ubatch_id);
    if (ub_it == checkpoints_.end()) {
        FT_LOG_INFO("No checkpoints for ubatch %d", ubatch_id);
        return;
    }

    auto layer_it = ub_it->second.find(layer_id);
    if (layer_it == ub_it->second.end()) {
        FT_LOG_INFO("No checkpoints for ubatch %d, layer %d", ubatch_id, layer_id);
        return;
    }

    FT_LOG_INFO("Checkpoint chain for ubatch %d, layer %d:", ubatch_id, layer_id);
    MoEDeltaCheckpoint* curr = layer_it->second;
    int idx = 0;
    while (curr != nullptr) {
        FT_LOG_INFO("  [%d] step=%d, tokens=%d, seq_offset=%d, kv_size=%zu, act_size=%zu",
                   idx, curr->step, curr->num_tokens, curr->seq_offset,
                   curr->kv_delta_size, curr->activation_delta_size);
        curr = curr->next;
        idx++;
    }
}

// MoECheckpointHelper Implementation

MoECheckpointHelper::MoECheckpointHelper(MoEDeltaCheckpointManager* manager)
    : manager_(manager), current_layer_(-1), enabled_(false)
{
}

void MoECheckpointHelper::checkpointMoEOutput(
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
    cudaStream_t stream)
{
    if (!isEnabled() || current_layer_ < 0) {
        return;
    }

    if (!manager_->shouldCheckpoint(step)) {
        return;
    }

    printf("[FT][Checkpoint] Creating MoE checkpoint: step=%d, layer=%d, ubatch=%d, tokens=%d\n",
           step, current_layer_, ubatch_id, num_tokens);

    manager_->createDeltaCheckpoint(
        step,
        current_layer_,
        ubatch_id,
        key_cache,
        value_cache,
        seq_start,
        seq_end,
        expert_indices,
        expert_weights,
        moe_output,
        num_tokens,
        batch_size,
        stream);
}

bool MoECheckpointHelper::restoreMoEState(
    int target_step,
    int ubatch_id,
    void* key_cache,
    void* value_cache,
    int* expert_indices,
    float* expert_weights,
    void* moe_output,
    cudaStream_t stream)
{
    if (!isEnabled() || current_layer_ < 0) return false;

    // First reconstruct KV cache
    int seq_len = 0;
    bool kv_success = manager_->reconstructKVCache(
        target_step, current_layer_, ubatch_id,
        key_cache, value_cache, &seq_len, stream);

    if (!kv_success) {
        FT_LOG_ERROR("Failed to reconstruct KV cache");
        return false;
    }

    // Then reconstruct activations
    bool act_success = manager_->reconstructActivations(
        target_step, current_layer_, ubatch_id,
        expert_indices, expert_weights, moe_output, stream);

    return kv_success && act_success;
}

// ============== File-based Checkpoint I/O ==============

void MoEDeltaCheckpointManager::ensureCheckpointDir()
{
    // Create checkpoint directory if it doesn't exist
    std::string cmd = "mkdir -p " + config_.checkpoint_dir;
    int ret = system(cmd.c_str());
    if (ret != 0) {
        printf("[FT][Checkpoint] Warning: Failed to create checkpoint directory: %s\n",
               config_.checkpoint_dir.c_str());
    }
}

std::string MoEDeltaCheckpointManager::getCheckpointFilepath(int step) const
{
    return config_.checkpoint_dir + "/" + config_.checkpoint_prefix + "_step" + std::to_string(step) + ".ckpt";
}

bool MoEDeltaCheckpointManager::checkpointFileExists(const std::string& filepath) const
{
    std::ifstream file(filepath);
    return file.good();
}

bool MoEDeltaCheckpointManager::getCheckpointFileInfo(const std::string& filepath, CheckpointFileHeader& header) const
{
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    file.read(reinterpret_cast<char*>(&header), sizeof(CheckpointFileHeader));
    if (!file.good()) {
        return false;
    }

    // Validate magic number
    if (header.magic != CheckpointFileHeader::MAGIC) {
        printf("[FT][Checkpoint] Invalid checkpoint file: bad magic number\n");
        return false;
    }

    // Check version compatibility
    if (header.version > CheckpointFileHeader::CURRENT_VERSION) {
        printf("[FT][Checkpoint] Checkpoint file version %u is newer than supported version %u\n",
               header.version, CheckpointFileHeader::CURRENT_VERSION);
        return false;
    }

    return true;
}

bool MoEDeltaCheckpointManager::writeCheckpointEntry(std::ofstream& file, const MoEDeltaCheckpoint* checkpoint)
{
    if (checkpoint == nullptr || !checkpoint->is_valid) {
        return false;
    }

    // Write entry header
    CheckpointEntryHeader entry_header;
    entry_header.step = checkpoint->step;
    entry_header.layer_id = checkpoint->layer_id;
    entry_header.ubatch_id = checkpoint->ubatch_id;
    entry_header.num_tokens = checkpoint->num_tokens;
    entry_header.seq_offset = checkpoint->seq_offset;
    entry_header.kv_delta_size = checkpoint->kv_delta_size;
    entry_header.activation_delta_size = checkpoint->activation_delta_size;
    entry_header.num_expert_indices = checkpoint->expert_indices.size();
    entry_header.num_expert_weights = checkpoint->expert_weights.size();
    entry_header.timestamp = checkpoint->timestamp;
    entry_header.reserved = 0;

    file.write(reinterpret_cast<const char*>(&entry_header), sizeof(CheckpointEntryHeader));

    // Allocate host buffer for device->host copy
    std::vector<char> host_buffer;

    // Write KV cache deltas (copy from device to host first)
    if (checkpoint->kv_delta_size > 0 && checkpoint->key_cache_delta != nullptr) {
        host_buffer.resize(checkpoint->kv_delta_size);

        cudaMemcpy(host_buffer.data(), checkpoint->key_cache_delta,
                   checkpoint->kv_delta_size, cudaMemcpyDeviceToHost);
        file.write(host_buffer.data(), checkpoint->kv_delta_size);

        cudaMemcpy(host_buffer.data(), checkpoint->value_cache_delta,
                   checkpoint->kv_delta_size, cudaMemcpyDeviceToHost);
        file.write(host_buffer.data(), checkpoint->kv_delta_size);
    }

    // Write expert routing info
    if (!checkpoint->expert_indices.empty()) {
        file.write(reinterpret_cast<const char*>(checkpoint->expert_indices.data()),
                   checkpoint->expert_indices.size() * sizeof(int));
    }
    if (!checkpoint->expert_weights.empty()) {
        file.write(reinterpret_cast<const char*>(checkpoint->expert_weights.data()),
                   checkpoint->expert_weights.size() * sizeof(float));
    }

    // Write activation deltas
    if (checkpoint->activation_delta_size > 0 && checkpoint->activation_delta != nullptr) {
        host_buffer.resize(checkpoint->activation_delta_size);
        cudaMemcpy(host_buffer.data(), checkpoint->activation_delta,
                   checkpoint->activation_delta_size, cudaMemcpyDeviceToHost);
        file.write(host_buffer.data(), checkpoint->activation_delta_size);
    }

    return file.good();
}

bool MoEDeltaCheckpointManager::readCheckpointEntry(std::ifstream& file, MoEDeltaCheckpoint* checkpoint)
{
    if (checkpoint == nullptr) {
        return false;
    }

    // Read entry header
    CheckpointEntryHeader entry_header;
    file.read(reinterpret_cast<char*>(&entry_header), sizeof(CheckpointEntryHeader));
    if (!file.good()) {
        return false;
    }

    checkpoint->step = entry_header.step;
    checkpoint->layer_id = entry_header.layer_id;
    checkpoint->ubatch_id = entry_header.ubatch_id;
    checkpoint->num_tokens = entry_header.num_tokens;
    checkpoint->seq_offset = entry_header.seq_offset;
    checkpoint->kv_delta_size = entry_header.kv_delta_size;
    checkpoint->activation_delta_size = entry_header.activation_delta_size;
    checkpoint->timestamp = entry_header.timestamp;
    checkpoint->is_valid = true;

    std::vector<char> host_buffer;

    // Read KV cache deltas
    if (entry_header.kv_delta_size > 0) {
        host_buffer.resize(entry_header.kv_delta_size);

        // Allocate device memory
        checkpoint->key_cache_delta = device_pool_->allocate(entry_header.kv_delta_size);
        checkpoint->value_cache_delta = device_pool_->allocate(entry_header.kv_delta_size);

        if (checkpoint->key_cache_delta == nullptr || checkpoint->value_cache_delta == nullptr) {
            printf("[FT][Checkpoint] Failed to allocate device memory for checkpoint\n");
            return false;
        }

        // Read key cache and copy to device
        file.read(host_buffer.data(), entry_header.kv_delta_size);
        cudaMemcpy(checkpoint->key_cache_delta, host_buffer.data(),
                   entry_header.kv_delta_size, cudaMemcpyHostToDevice);

        // Read value cache and copy to device
        file.read(host_buffer.data(), entry_header.kv_delta_size);
        cudaMemcpy(checkpoint->value_cache_delta, host_buffer.data(),
                   entry_header.kv_delta_size, cudaMemcpyHostToDevice);
    }

    // Read expert routing info
    if (entry_header.num_expert_indices > 0) {
        checkpoint->expert_indices.resize(entry_header.num_expert_indices);
        file.read(reinterpret_cast<char*>(checkpoint->expert_indices.data()),
                  entry_header.num_expert_indices * sizeof(int));
    }
    if (entry_header.num_expert_weights > 0) {
        checkpoint->expert_weights.resize(entry_header.num_expert_weights);
        file.read(reinterpret_cast<char*>(checkpoint->expert_weights.data()),
                  entry_header.num_expert_weights * sizeof(float));
    }

    // Read activation deltas
    if (entry_header.activation_delta_size > 0) {
        host_buffer.resize(entry_header.activation_delta_size);

        checkpoint->activation_delta = device_pool_->allocate(entry_header.activation_delta_size);
        if (checkpoint->activation_delta == nullptr) {
            printf("[FT][Checkpoint] Failed to allocate device memory for activation\n");
            return false;
        }

        file.read(host_buffer.data(), entry_header.activation_delta_size);
        cudaMemcpy(checkpoint->activation_delta, host_buffer.data(),
                   entry_header.activation_delta_size, cudaMemcpyHostToDevice);
    }

    return file.good();
}

bool MoEDeltaCheckpointManager::saveCheckpointsToFile(const std::string& filepath)
{
    return saveCheckpointsToFile(filepath, INT_MAX);
}

bool MoEDeltaCheckpointManager::saveCheckpointsToFile(const std::string& filepath, int max_step)
{
    std::lock_guard<std::mutex> lock(checkpoint_mutex_);

    ensureCheckpointDir();

    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        printf("[FT][Checkpoint] Failed to open file for writing: %s\n", filepath.c_str());
        return false;
    }

    printf("[FT][Checkpoint] Saving checkpoints to file: %s (max_step=%d)\n", filepath.c_str(), max_step);

    // Collect all checkpoints to save
    std::vector<MoEDeltaCheckpoint*> all_checkpoints;
    int min_step = INT_MAX;
    int actual_max_step = -1;

    for (auto& ubatch_pair : checkpoints_) {
        for (auto& layer_pair : ubatch_pair.second) {
            MoEDeltaCheckpoint* curr = layer_pair.second;
            while (curr != nullptr) {
                if (curr->is_valid && curr->step <= max_step) {
                    all_checkpoints.push_back(curr);
                    min_step = std::min(min_step, curr->step);
                    actual_max_step = std::max(actual_max_step, curr->step);
                }
                curr = curr->next;
            }
        }
    }

    if (all_checkpoints.empty()) {
        printf("[FT][Checkpoint] No checkpoints to save\n");
        file.close();
        return false;
    }

    // Sort by step, then layer, then ubatch for consistent ordering
    std::sort(all_checkpoints.begin(), all_checkpoints.end(),
              [](MoEDeltaCheckpoint* a, MoEDeltaCheckpoint* b) {
                  if (a->step != b->step) return a->step < b->step;
                  if (a->layer_id != b->layer_id) return a->layer_id < b->layer_id;
                  return a->ubatch_id < b->ubatch_id;
              });

    // Calculate total data size
    uint64_t total_data_size = 0;
    for (auto* ckpt : all_checkpoints) {
        total_data_size += sizeof(CheckpointEntryHeader);
        total_data_size += 2 * ckpt->kv_delta_size;  // key + value
        total_data_size += ckpt->expert_indices.size() * sizeof(int);
        total_data_size += ckpt->expert_weights.size() * sizeof(float);
        total_data_size += ckpt->activation_delta_size;
    }

    // Write file header
    CheckpointFileHeader header;
    memset(&header, 0, sizeof(header));
    header.magic = CheckpointFileHeader::MAGIC;
    header.version = CheckpointFileHeader::CURRENT_VERSION;
    header.num_checkpoints = all_checkpoints.size();
    header.num_layers = config_.num_layers;
    header.num_experts = config_.num_experts;
    header.moe_k = config_.moe_k;
    header.hidden_units = config_.hidden_units;
    header.size_per_head = config_.size_per_head;
    header.num_heads = config_.num_heads;
    header.min_step = min_step;
    header.max_step = actual_max_step;
    header.total_data_size = total_data_size;

    file.write(reinterpret_cast<const char*>(&header), sizeof(CheckpointFileHeader));

    // Write each checkpoint
    int written = 0;
    for (auto* ckpt : all_checkpoints) {
        if (writeCheckpointEntry(file, ckpt)) {
            written++;
        } else {
            printf("[FT][Checkpoint] Failed to write checkpoint at step=%d, layer=%d\n",
                   ckpt->step, ckpt->layer_id);
        }
    }

    file.close();

    printf("[FT][Checkpoint] Saved %d checkpoints to %s (steps %d-%d, size=%.2f MB)\n",
           written, filepath.c_str(), min_step, actual_max_step,
           total_data_size / (1024.0 * 1024.0));

    return written > 0;
}

bool MoEDeltaCheckpointManager::loadCheckpointsFromFile(const std::string& filepath)
{
    std::lock_guard<std::mutex> lock(checkpoint_mutex_);

    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        printf("[FT][Checkpoint] Failed to open file for reading: %s\n", filepath.c_str());
        return false;
    }

    printf("[FT][Checkpoint] Loading checkpoints from file: %s\n", filepath.c_str());

    // Read and validate file header
    CheckpointFileHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(CheckpointFileHeader));
    if (!file.good()) {
        printf("[FT][Checkpoint] Failed to read file header\n");
        return false;
    }

    if (header.magic != CheckpointFileHeader::MAGIC) {
        printf("[FT][Checkpoint] Invalid checkpoint file: bad magic number\n");
        return false;
    }

    if (header.version > CheckpointFileHeader::CURRENT_VERSION) {
        printf("[FT][Checkpoint] Unsupported checkpoint version: %u (max supported: %u)\n",
               header.version, CheckpointFileHeader::CURRENT_VERSION);
        return false;
    }

    printf("[FT][Checkpoint] File info: %u checkpoints, steps %d-%d, size=%.2f MB\n",
           header.num_checkpoints, header.min_step, header.max_step,
           header.total_data_size / (1024.0 * 1024.0));

    // Read checkpoints
    int loaded = 0;
    for (uint32_t i = 0; i < header.num_checkpoints; i++) {
        MoEDeltaCheckpoint* checkpoint = allocateCheckpoint();

        if (!readCheckpointEntry(file, checkpoint)) {
            printf("[FT][Checkpoint] Failed to read checkpoint %u\n", i);
            freeCheckpoint(checkpoint);
            continue;
        }

        // Add to checkpoint list
        MoEDeltaCheckpoint*& head = checkpoints_[checkpoint->ubatch_id][checkpoint->layer_id];
        checkpoint->next = head;
        if (head != nullptr) {
            head->prev = checkpoint;
        }
        head = checkpoint;

        // Update last checkpoint step
        if (last_checkpoint_step_.find(checkpoint->ubatch_id) == last_checkpoint_step_.end() ||
            last_checkpoint_step_[checkpoint->ubatch_id] < checkpoint->step) {
            last_checkpoint_step_[checkpoint->ubatch_id] = checkpoint->step;
        }

        loaded++;
        total_checkpoints_++;
        total_bytes_checkpointed_ += checkpoint->kv_delta_size * 2 + checkpoint->activation_delta_size;
    }

    file.close();

    printf("[FT][Checkpoint] Loaded %d checkpoints from %s\n", loaded, filepath.c_str());

    return loaded > 0;
}

bool MoEDeltaCheckpointManager::loadAndRecoverFromFile(const std::string& filepath, int target_step, int ubatch_id)
{
    printf("[FT][Checkpoint] Loading and recovering from file: %s, target_step=%d, ubatch=%d\n",
           filepath.c_str(), target_step, ubatch_id);

    // First load all checkpoints from file
    if (!loadCheckpointsFromFile(filepath)) {
        printf("[FT][Checkpoint] Failed to load checkpoints from file\n");
        return false;
    }

    // Then initiate recovery
    if (!initiateRecovery(target_step, ubatch_id)) {
        printf("[FT][Checkpoint] Failed to initiate recovery to step %d\n", target_step);
        return false;
    }

    return true;
}

void MoEDeltaCheckpointManager::autoSaveCheckpoint(int step)
{
    if (!config_.enable_file_checkpoint) {
        return;
    }

    // Save every N steps (use checkpoint_interval)
    if (step % config_.checkpoint_interval != 0) {
        return;
    }

    std::string filepath = getCheckpointFilepath(step);
    saveCheckpointsToFile(filepath, step);
}

}  // namespace fastertransformer
