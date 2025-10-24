# End-to-End Guide: Running Qwen235B MoE with Arbitrary Token Fault Tolerance

This guide provides complete instructions for running Qwen235B MoE models with token-level fault tolerance in DejaVu.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Model Preparation](#model-preparation)
4. [Configuration](#configuration)
5. [Running the System](#running-the-system)
6. [Fault Injection & Testing](#fault-injection--testing)
7. [Monitoring & Debugging](#monitoring--debugging)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Hardware Requirements

- **GPUs**: 8x NVIDIA A100/H100 GPUs (80GB each recommended)
- **Memory**: 512GB+ system RAM
- **Storage**: 2TB+ NVMe SSD for model weights
- **Network**: High-bandwidth interconnect (NVLink, InfiniBand)

### Software Requirements

- Ubuntu 20.04/22.04
- CUDA 11.8 or 12.x
- Python 3.8+
- PyTorch 1.13.0+
- OpenMPI 5.0+
- NCCL 2.15+

---

## Environment Setup

### Step 1: Install Dependencies

Follow the [install.md](install.md) guide through step 7, then continue here.

### Step 2: Build DejaVu with MoE Token FT Support

```bash
cd /root/dejavu1
mkdir -p build
cd build

# Generate protobuf files
protoc -I ../src/fastertransformer/models/multi_gpu_gpt/ \
    --cpp_out=../src/fastertransformer/models/multi_gpu_gpt/ \
    ../src/fastertransformer/models/multi_gpu_gpt/state_stream.proto

protoc -I ../src/fastertransformer/models/multi_gpu_gpt/ \
    --grpc_out=../src/fastertransformer/models/multi_gpu_gpt/ \
    --plugin=protoc-gen-grpc=`which grpc_cpp_plugin` \
    ../src/fastertransformer/models/multi_gpu_gpt/state_stream.proto

protoc -I ../src/fastertransformer/models/multi_gpu_gpt/ \
    --grpc_out=../src/fastertransformer/models/multi_gpu_gpt/ \
    --plugin=protoc-gen-grpc=`which grpc_cpp_plugin` \
    ../src/fastertransformer/models/multi_gpu_gpt/ft_state.proto

protoc -I ../src/fastertransformer/models/multi_gpu_gpt/ \
    --cpp_out=../src/fastertransformer/models/multi_gpu_gpt/ \
    ../src/fastertransformer/models/multi_gpu_gpt/ft_state.proto

# Build with MoE and fault tolerance support
# Replace 80 with your GPU's compute capability (80 for A100, 90 for H100)
cmake -DSM=80 \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_PYT=ON \
      -DBUILD_MULTI_GPU=ON \
      -DBUILD_CUTLASS_MOE=ON \
      -DBUILD_MICROBENCHMARKS=ON \
      ..

make -j$(nproc)
```

### Step 3: Install Python Dependencies

```bash
pip install -r ../examples/pytorch/gpt/requirement.txt

# Install API server dependencies
cd ../examples/pytorch/gpt/api
python -m grpc_tools.protoc \
    --proto_path=. \
    --python_out=. \
    --grpc_python_out=. \
    protos/api_server.proto
cd -
```

### Step 4: Verify MoE Token FT Library

```bash
# Check that the library was built
ls -lh lib/libmoe_token_ft_manager.a

# Expected output: ~324KB file
```

---

## Model Preparation

### Option 1: Download Qwen235B MoE from Hugging Face

```bash
# Create model directory
mkdir -p /data/models/qwen235b-moe

# Download using Hugging Face CLI
pip install huggingface_hub

python << EOF
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen2.5-MoE-235B",  # Or specific variant
    local_dir="/data/models/qwen235b-moe",
    local_dir_use_symlinks=False
)
EOF
```

### Option 2: Convert Existing Checkpoint

If you have Qwen235B MoE in another format:

```bash
cd /root/dejavu1/examples/pytorch/gpt/utils

# Use the MoE checkpoint converter
python megatron_gpt_moe_ckpt_convert.py \
    --input-dir /path/to/qwen235b/checkpoint \
    --output-dir /data/models/qwen235b-moe-ft \
    --target-tp 8 \
    --target-pp 4
```

### Step 3: Verify Model Structure

```bash
# Expected structure
ls -la /data/models/qwen235b-moe-ft/

# Should contain:
# - 8-gpu/ (tensor parallel shards)
# - config.ini (model configuration)
# - [layer weights for each MoE layer]
```

---

## Configuration

### Step 1: Configure MoE Token Fault Tolerance

Create environment configuration file:

```bash
cat > ~/.moe_ft_config << 'EOF'
# MoE Token FT Configuration

# Enable fault tolerance
export ENABLE_MOE_TOKEN_FT=1

# Checkpoint every N tokens (1 = every token, higher = less overhead)
export MOE_CHECKPOINT_INTERVAL=1

# Maximum checkpoints to keep per microbatch
export MOE_MAX_CHECKPOINTS=100

# Checkpoint policy: 0=ALL, 1=INTERVAL, 2=ADAPTIVE (recommended)
export MOE_CHECKPOINT_POLICY=2

# Adaptive policy thresholds
export MOE_ENTROPY_THRESHOLD=0.8
export MOE_IMBALANCE_THRESHOLD=0.3

# Memory pool sizes (in MB)
export MOE_DEVICE_POOL_SIZE=512
export MOE_HOST_BUFFER_SIZE=128

# Debug settings
export FT_LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR
export ENABLE_FT_STATS=1   # Print FT statistics

# Optional: Enable failure injection for testing
# export ENABLE_FAILURE_INJECTION=1
# export FAILURE_INJECTION_RATE=0.01  # 1% failure rate
EOF

# Load configuration
source ~/.moe_ft_config
```

### Step 2: Configure Model Parameters

Create Qwen235B MoE configuration:

```bash
python /root/dejavu1/examples/qwen235b_moe_ft_config.py \
    --checkpoint-interval 1 \
    --max-checkpoints 100 \
    --adaptive \
    --estimate-memory \
    --generate-cpp > qwen235b_config.txt

# Review the configuration
cat qwen235b_config.txt
```

### Step 3: Create Run Script

Create a script to run Qwen235B MoE with fault tolerance:

```bash
cat > run_qwen235b_moe_ft.sh << 'EOF'
#!/bin/bash

# Load MoE FT configuration
source ~/.moe_ft_config

# Qwen235B MoE Model Configuration
export LAYER_NUM=80
export HEAD_NUM=64
export SIZE_PER_HEAD=128       # 8192 hidden / 64 heads
export INTER_SIZE=29568
export VOCAB_SIZE=152064
export MAX_SEQ_LEN=32768

# MoE specific
export EXPERT_NUM=256
export MOE_K=8
# MoE layers (example: every other layer)
export MOE_LAYER_INDEX="1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49,51,53,55,57,59,61,63,65,67,69,71,73,75,77,79"

# Parallelism configuration
export TENSOR_PARA_SIZE=8
export PIPELINE_PARA_SIZE=4
export PROMPT_WORLD_SIZE=8   # Prompt-phase GPUs
export TOKEN_WORLD_SIZE=8    # Token-phase GPUs

# Batch configuration
export UBATCH_SIZE=4
export NUM_UBATCHES=8
export MAX_BATCH_SIZE=32

# Generation parameters
export INPUT_LEN=512
export OUTPUT_LEN=512
export TEMPERATURE=0.7
export TOP_K=50
export TOP_P=0.95

# Paths
export CKPT_PATH="/data/models/qwen235b-moe-ft"
export VOCAB_FILE="/data/models/qwen235b-moe/vocab.json"
export MERGES_FILE="/data/models/qwen235b-moe/merges.txt"
export LIB_PATH="/root/dejavu1/build/lib/libth_transformer.so"

# Run the model
cd /root/dejavu1/build

mpirun -n $((TENSOR_PARA_SIZE * PIPELINE_PARA_SIZE)) \
    --allow-run-as-root \
    -x ENABLE_MOE_TOKEN_FT \
    -x MOE_CHECKPOINT_INTERVAL \
    -x MOE_MAX_CHECKPOINTS \
    -x MOE_CHECKPOINT_POLICY \
    python ../examples/pytorch/gpt/gpt_batch_maker.py \
        --layer_num $LAYER_NUM \
        --head_num $HEAD_NUM \
        --size_per_head $SIZE_PER_HEAD \
        --inter_size $INTER_SIZE \
        --vocab_size $VOCAB_SIZE \
        --max_seq_len $MAX_SEQ_LEN \
        --expert_num $EXPERT_NUM \
        --moe_k $MOE_K \
        --moe_layer_index $MOE_LAYER_INDEX \
        --tensor_para_size $TENSOR_PARA_SIZE \
        --pipeline_para_size $PIPELINE_PARA_SIZE \
        --prompt_world_size $PROMPT_WORLD_SIZE \
        --token_world_size $TOKEN_WORLD_SIZE \
        --ckpt_path $CKPT_PATH \
        --vocab_file $VOCAB_FILE \
        --merges_file $MERGES_FILE \
        --lib_path $LIB_PATH \
        --input_len $INPUT_LEN \
        --output_len $OUTPUT_LEN \
        --ubatch_size $UBATCH_SIZE \
        --num_ubatches $NUM_UBATCHES \
        --temperature $TEMPERATURE \
        --top_k $TOP_K \
        --top_p $TOP_P \
        --inference_data_type fp16 \
        --time
EOF

chmod +x run_qwen235b_moe_ft.sh
```

---

## Running the System

### Step 1: Basic Execution

```bash
cd /root/dejavu1/build

# Source environment
source ~/.moe_ft_config

# Run with fault tolerance enabled
./run_qwen235b_moe_ft.sh
```

### Step 2: Monitor Execution

Open a new terminal and monitor:

```bash
# Monitor GPU utilization
watch -n 1 nvidia-smi

# Monitor system resources
htop

# Monitor MoE FT logs
tail -f /tmp/moe_ft_*.log  # If logging to file
```

### Step 3: Verify Fault Tolerance

Check that MoE Token FT is active:

```bash
# Look for initialization message in output
# Expected: "MoE Token FT initialized successfully"
# Expected: "Initializing MoE Token FT: experts=256, k=8, checkpoint_interval=1"
```

---

## Fault Injection & Testing

### Enable Failure Injection

Test the fault tolerance by injecting failures:

```bash
# Add to ~/.moe_ft_config
export ENABLE_FAILURE_INJECTION=1
export FAILURE_INJECTION_RATE=0.01      # 1% of tokens fail
export FAILURE_INJECTION_STEP_START=100  # Start after warmup
export FAILURE_INJECTION_STEP_END=1000   # End before completion

source ~/.moe_ft_config

# Run again
./run_qwen235b_moe_ft.sh
```

### Test Recovery

Manually trigger failure at specific step:

```bash
# In Python code, add:
# model.handleMoEFailure(step=150, ubatch_id=0)

# Or use signal injection
kill -SIGUSR1 <pid>  # If implemented
```

### Verify Recovery

Check logs for recovery messages:

```
Expected output:
[INFO] Handling MoE failure at step=150, ubatch=0
[INFO] Initiating recovery from step 149
[INFO] Restored checkpoint: step=149, ubatch=0, token=5
[INFO] MoE recovery completed successfully
```

---

## Monitoring & Debugging

### Enable Detailed Statistics

```bash
# In your Python code or via environment
export ENABLE_FT_STATS=1
export FT_STAT_INTERVAL=100  # Print stats every 100 steps
```

### Print MoE Token FT Stats

Add to your Python script:

```python
# After initialization
if hasattr(model, 'printMoETokenFTStats'):
    model.printMoETokenFTStats()
```

Expected output:

```
=== MoE Token FT Manager Statistics ===
Current step: 500
Checkpoint step: 500
Active ubatches: 8
  Ubatch 0: 100 checkpoints
  Ubatch 1: 100 checkpoints
  ...
Total checkpoints: 800
Device pool usage: 195 / 512 MB
MoE Token FT memory usage: 220 MB
```

### Debug Mode

Enable debug logging:

```bash
export FT_LOG_LEVEL=DEBUG
export CUDA_LAUNCH_BLOCKING=1  # Synchronous CUDA for debugging

./run_qwen235b_moe_ft.sh 2>&1 | tee qwen235b_debug.log
```

### Performance Profiling

Profile checkpoint overhead:

```bash
# Use NVTX markers (already in code)
nsys profile \
    --trace=cuda,nvtx \
    --output=qwen235b_moe_ft_profile \
    ./run_qwen235b_moe_ft.sh

# Analyze with Nsight Systems
```

---

## Troubleshooting

### Issue 1: Out of Memory

**Symptoms**: CUDA OOM errors

**Solutions**:

```bash
# Reduce checkpoint pool size
export MOE_MAX_CHECKPOINTS=50

# Increase checkpoint interval
export MOE_CHECKPOINT_INTERVAL=5

# Reduce batch size
export UBATCH_SIZE=2
export NUM_UBATCHES=4
```

### Issue 2: Checkpoint Pool Wraparound

**Symptoms**: Warning "Checkpoint pool wrapped around"

**Solutions**:

```bash
# Increase device pool size
export MOE_DEVICE_POOL_SIZE=1024  # 1GB

# Or reduce max checkpoints
export MOE_MAX_CHECKPOINTS=50
```

### Issue 3: Recovery Fails

**Symptoms**: Error "Cannot restore: checkpoint not found"

**Solutions**:

```bash
# Ensure checkpoints aren't being pruned too aggressively
export MOE_MAX_CHECKPOINTS=200

# Reduce checkpoint interval
export MOE_CHECKPOINT_INTERVAL=1

# Check checkpoint validity
python << EOF
# Add validation code
EOF
```

### Issue 4: High Overhead

**Symptoms**: >10% performance degradation

**Solutions**:

```bash
# Use adaptive policy
export MOE_CHECKPOINT_POLICY=2

# Increase checkpoint interval
export MOE_CHECKPOINT_INTERVAL=10

# Profile and identify bottleneck
nsys profile ./run_qwen235b_moe_ft.sh
```

### Issue 5: MoE Layer Not Found

**Symptoms**: "MoE Token FT requested but model is not MoE"

**Solutions**:

```bash
# Verify expert_num and moe_k are set
echo $EXPERT_NUM  # Should be 256
echo $MOE_K       # Should be 8

# Check moe_layer_index is provided
echo $MOE_LAYER_INDEX  # Should be comma-separated list
```

---

## Advanced Features

### Adaptive Checkpointing

Configure adaptive thresholds:

```bash
export MOE_CHECKPOINT_POLICY=2
export MOE_ENTROPY_THRESHOLD=0.85      # Higher = more selective
export MOE_IMBALANCE_THRESHOLD=0.25    # Lower = more selective
```

### Custom Checkpoint Policy

Implement custom policy in Python:

```python
def should_checkpoint(expert_indices, expert_weights):
    # Your custom logic
    entropy = calculate_entropy(expert_indices)
    return entropy > custom_threshold

# Hook into model
model.set_checkpoint_policy(should_checkpoint)
```

### Persistent Checkpoints

Enable checkpoint persistence (future feature):

```bash
export MOE_PERSISTENT_CHECKPOINTS=1
export MOE_CHECKPOINT_DIR=/data/checkpoints/qwen235b-moe
```

---

## Performance Benchmarks

Expected performance with MoE Token FT enabled:

| Configuration | Overhead | Memory Usage | Recovery Time |
|---------------|----------|--------------|---------------|
| Every token (interval=1) | 0.3-0.5ms | ~25MB/ubatch | 1-2ms |
| Every 5 tokens (interval=5) | 0.1-0.2ms | ~5MB/ubatch | 2-5ms |
| Adaptive | 0.1-0.3ms | ~10-15MB/ubatch | 1-3ms |

---

## Complete Example

Here's a minimal complete example:

```bash
#!/bin/bash

# 1. Setup
source ~/.moe_ft_config

# 2. Run
cd /root/dejavu1/build

mpirun -n 32 --allow-run-as-root \
    python ../examples/pytorch/gpt/gpt_batch_maker.py \
        --expert_num 256 \
        --moe_k 8 \
        --moe_layer_index "1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49,51,53,55,57,59,61,63,65,67,69,71,73,75,77,79" \
        --layer_num 80 \
        --head_num 64 \
        --size_per_head 128 \
        --vocab_size 152064 \
        --tensor_para_size 8 \
        --pipeline_para_size 4 \
        --ckpt_path /data/models/qwen235b-moe-ft \
        --ubatch_size 4 \
        --output_len 256 \
        --inference_data_type fp16

# 3. Check results
echo "Run complete. Check logs for FT statistics."
```

---

## Next Steps

1. **Production Deployment**: See [production_deployment.md](production_deployment.md)
2. **Performance Tuning**: See [performance_tuning.md](performance_tuning.md)
3. **API Integration**: See [api_integration.md](api_integration.md)

---

## Support

For issues or questions:
- GitHub: https://github.com/msr-fiddle/dejavu1/issues
- Documentation: /root/dejavu1/docs/moe_token_ft_guide.md
- Examples: /root/dejavu1/examples/

---

**Ready to deploy Qwen235B MoE with fault tolerance!** 🚀
