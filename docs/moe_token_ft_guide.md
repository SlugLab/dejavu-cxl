# MoE Token Stream Fault Tolerance Guide

## Overview

This guide describes the token stream-based fault tolerance mechanism for Mixture of Experts (MoE) models, specifically designed and tested for Qwen235B MoE models in the DejaVu distributed inference system.

## What is MoE Token Stream Fault Tolerance?

Token stream fault tolerance provides fine-grained checkpointing and recovery at the token level for MoE models. Unlike traditional checkpoint mechanisms that save entire model states, this approach:

1. **Token-level checkpointing**: Saves expert routing decisions and activations for each generated token
2. **Lightweight recovery**: Allows recovery from the last valid token without recomputing the entire sequence
3. **MoE-specific optimizations**: Tracks expert indices, gating weights, and expert activations separately
4. **Adaptive checkpointing**: Intelligently decides when to checkpoint based on expert usage patterns

## Architecture

### Components

#### 1. MoETokenFTManager
The core manager that handles:
- Token-level checkpoint creation and storage
- Memory management for checkpoint buffers
- Checkpoint pruning and lifecycle management
- Recovery state coordination

#### 2. TokenStreamRecoveryHelper
Assists with recovery workflows:
- Orchestrates multi-token recovery sequences
- Manages recovery context for each microbatch
- Validates checkpoint integrity during recovery

#### 3. AdaptiveCheckpointPolicy
Implements intelligent checkpointing:
- Monitors expert activation entropy
- Detects load imbalance across experts
- Decides when checkpoints are most valuable

### Memory Management

The system uses two-tier memory:

1. **Device Memory Pool**: Stores expert activations on GPU
   - Circular buffer allocation
   - Configurable pool size
   - Automatic wraparound when full

2. **Host Memory Buffer**: Stores metadata (expert indices, weights)
   - Pinned memory for fast async transfers
   - Compact storage for routing decisions

## Integration with Qwen235B MoE

### Model Configuration

Qwen235B MoE specifications:
- **Total Experts**: 256 experts per MoE layer
- **Active Experts (k)**: 8 experts per token
- **MoE Layers**: Selective layers in the 235B parameter model
- **Hidden Units**: Depends on model configuration (typically 8192)

### Initialization

```cpp
// In your model initialization code
ParallelGptDVFT<half> gpt(
    max_batch_size,
    max_seq_len,
    max_input_len,
    beam_width,
    head_num,
    size_per_head,
    inter_size,
    num_layer,
    256,  // expert_num for Qwen235B
    8,    // moe_k for Qwen235B
    moe_layer_index,  // Which layers are MoE
    vocab_size,
    // ... other parameters
);

// Enable MoE Token FT
gpt.initializeMoETokenFT(
    true,  // enable
    1      // checkpoint every token (can be adjusted for performance)
);
```

### Environment Variables

Configure behavior through environment variables:

```bash
# Enable MoE Token FT
export ENABLE_MOE_TOKEN_FT=1

# Checkpoint interval (tokens)
export MOE_CHECKPOINT_INTERVAL=1

# Maximum checkpoints to keep per microbatch
export MOE_MAX_CHECKPOINTS=100

# Checkpoint policy: 0=ALL, 1=INTERVAL, 2=ADAPTIVE
export MOE_CHECKPOINT_POLICY=1
```

## Usage Example

### Basic Usage

```cpp
// During token generation (inside forward pass)
void generateTokenWithCheckpoint(int step, int ubatch_id) {
    // ... normal token generation ...

    // After MoE layer forward pass
    if (isMoELayer(layer_id)) {
        // Checkpoint the token
        checkpointMoEToken(
            token_id,           // Current token position
            step,               // Generation step
            ubatch_id,          // Microbatch ID
            expert_indices,     // Which experts were selected
            expert_weights,     // Gating weights from router
            expert_activations, // Expert output activations
            activation_size     // Size of activations
        );
    }
}
```

### Failure Handling

```cpp
// When a failure is detected
try {
    // Token generation
    generateToken(step, ubatch_id);
} catch (const std::exception& e) {
    FT_LOG_ERROR("Token generation failed at step %d", step);

    // Initiate recovery
    handleMoEFailure(step, ubatch_id);

    // Recovery process runs automatically
    // Will restore from last valid checkpoint

    // Resume generation
    resumeFromCheckpoint();
}
```

### Advanced: Adaptive Checkpointing

```cpp
// Configure adaptive checkpointing
AdaptiveCheckpointPolicy* policy = new AdaptiveCheckpointPolicy(
    256,  // num_experts
    8     // moe_k
);

// Policy will automatically checkpoint when:
// 1. Expert usage entropy is high (diverse routing)
// 2. Load imbalance is detected (skewed routing)

// The system handles this internally when enabled
```

## Performance Considerations

### Memory Overhead

For Qwen235B with typical settings:
- **Per Token Checkpoint**: ~2 MB
  - Expert indices: 8 × 4 bytes = 32 bytes
  - Expert weights: 8 × 4 bytes = 32 bytes
  - Expert activations: 8 × 8192 × 4 bytes = 256 KB (depends on hidden size)

- **100 Checkpoints**: ~200 MB per microbatch

### Performance Impact

- **Checkpoint overhead**: ~0.1-0.5ms per token (async GPU copy)
- **Recovery time**: 1-5ms per token (depends on activation size)
- **Memory bandwidth**: Minimal impact due to async operations

### Optimization Tips

1. **Adjust checkpoint interval**:
   ```cpp
   gpt.initializeMoETokenFT(true, 10);  // Checkpoint every 10 tokens
   ```

2. **Use adaptive policy** for optimal checkpoint selection:
   ```bash
   export MOE_CHECKPOINT_POLICY=2
   ```

3. **Tune max checkpoints** based on available memory:
   ```bash
   export MOE_MAX_CHECKPOINTS=50  # Reduce for memory-constrained systems
   ```

## Monitoring and Debugging

### Print Statistics

```cpp
// Print current checkpoint statistics
gpt.printMoETokenFTStats();

// Output example:
// === MoE Token FT Manager Statistics ===
// Current step: 150
// Checkpoint step: 150
// Active ubatches: 4
//   Ubatch 0: 100 checkpoints
//   Ubatch 1: 100 checkpoints
//   Ubatch 2: 98 checkpoints
//   Ubatch 3: 95 checkpoints
// Total checkpoints: 393
// Device pool usage: 185 / 512 MB
```

### Query Checkpoint Status

```cpp
// Check if checkpoint exists
bool has_cp = gpt.moe_token_ft_manager_->hasCheckpoint(step, ubatch_id);

// Get checkpoint count
size_t count = gpt.moe_token_ft_manager_->getCheckpointCount(ubatch_id);

// Check if in recovery mode
bool recovering = gpt.moe_token_ft_manager_->isInRecovery();
```

## Integration with DejaVu Pipeline

### Prompt-Token Disaggregation

The MoE Token FT system integrates seamlessly with DejaVu's prompt-token disaggregation:

1. **Prompt Phase**: Checkpoints are created for MoE layers during context encoding
2. **Token Phase**: Continuous checkpointing during autoregressive generation
3. **Cache Streaming**: Checkpoints are maintained independently of KV cache streaming

### Microbatch Coordination

Each microbatch maintains independent checkpoint state:
- Separate checkpoint buffers per microbatch
- Isolated recovery workflows
- No interference between concurrent microbatches

## Best Practices

1. **Enable for MoE layers only**: Don't checkpoint non-MoE layers
2. **Use adaptive policy in production**: Better checkpoint efficiency
3. **Monitor memory usage**: Ensure checkpoint pool doesn't overflow
4. **Tune checkpoint interval**: Balance overhead vs recovery granularity
5. **Test recovery paths**: Regularly validate checkpoint/recovery works

## Troubleshooting

### Issue: Out of checkpoint memory

**Symptoms**: Warning "Checkpoint pool wrapped around"

**Solutions**:
- Reduce `MOE_MAX_CHECKPOINTS`
- Increase checkpoint interval
- Clear old checkpoints: `gpt.moe_token_ft_manager_->clearCheckpoints(ubatch_id)`

### Issue: Recovery fails

**Symptoms**: Error "Cannot restore: checkpoint not found"

**Solutions**:
- Check if checkpoints were pruned
- Verify checkpoint interval matches recovery expectations
- Ensure failure occurred after at least one checkpoint

### Issue: High overhead

**Symptoms**: Significant slowdown during generation

**Solutions**:
- Increase checkpoint interval (e.g., 5 or 10)
- Switch to adaptive policy
- Verify async operations are not blocking

## API Reference

### Key Methods

```cpp
class ParallelGptDVFT {
public:
    // Initialize MoE Token FT
    void initializeMoETokenFT(bool enable = true, size_t checkpoint_interval = 1);

    // Shutdown and cleanup
    void shutdownMoETokenFT();

    // Checkpoint a token
    void checkpointMoEToken(int token_id, int step, int ubatch_id,
                           const int* expert_indices,
                           const float* expert_weights,
                           const void* expert_activations,
                           size_t activation_size);

    // Recover a token
    bool recoverMoEToken(int step, int ubatch_id, int token_id,
                        int* expert_indices,
                        float* expert_weights,
                        void* expert_activations);

    // Handle failure and initiate recovery
    void handleMoEFailure(int failed_step, int ubatch_id);

    // Complete recovery process
    void completeMoERecovery();

    // Query status
    bool isMoETokenFTEnabled() const;

    // Print statistics
    void printMoETokenFTStats() const;
};
```

## Future Enhancements

Potential improvements for future versions:

1. **Persistent Checkpoints**: Save checkpoints to disk for long-running jobs
2. **Distributed Checkpointing**: Coordinate checkpoints across multiple nodes
3. **Compression**: Compress expert activations to reduce memory
4. **Selective Recovery**: Recover only specific experts rather than full token
5. **Cross-Layer Recovery**: Coordinate recovery across multiple MoE layers

## References

- DejaVu Paper: [Prompt-Token Disaggregated Serving]
- Qwen2.5 MoE Architecture Documentation
- FasterTransformer MoE Implementation

## Support

For issues or questions:
- GitHub Issues: https://github.com/msr-fiddle/dejavu1/issues
- Documentation: /root/dejavu1/docs/

---

**Note**: This is an advanced feature for production MoE deployments. Ensure thorough testing in your specific environment before deploying to production systems.
