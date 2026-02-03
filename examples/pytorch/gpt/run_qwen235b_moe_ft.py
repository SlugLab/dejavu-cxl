#!/usr/bin/env python3
"""
Qwen235B MoE with Token Stream Fault Tolerance Runner

This script provides a simple interface to run Qwen235B MoE models
with token-level fault tolerance enabled.

Usage:
    python run_qwen235b_moe_ft.py --ckpt_path /path/to/model [options]

For full options, run:
    python run_qwen235b_moe_ft.py --help
"""

import argparse
import configparser
import os
import sys
import time

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, "../../.."))
from examples.pytorch.gpt.utils import comm
from examples.pytorch.gpt.utils import gpt_decoder
from examples.pytorch.gpt.utils.parallel_gpt_dv import ParallelGPT

from utils import word_list


def setup_moe_ft_environment(args):
    """
    Configure environment variables for MoE Token FT
    """
    print("=" * 80)
    print("Setting up MoE Token Fault Tolerance Environment")
    print("=" * 80)

    # Enable MoE Token FT
    os.environ['ENABLE_MOE_TOKEN_FT'] = '1' if args.enable_moe_ft else '0'

    # Checkpoint configuration
    os.environ['MOE_CHECKPOINT_INTERVAL'] = str(args.moe_checkpoint_interval)
    os.environ['MOE_MAX_CHECKPOINTS'] = str(args.moe_max_checkpoints)

    # Checkpoint policy: 0=ALL, 1=INTERVAL, 2=ADAPTIVE
    policy_map = {'all': '0', 'interval': '1', 'adaptive': '2'}
    os.environ['MOE_CHECKPOINT_POLICY'] = policy_map.get(args.moe_checkpoint_policy, '2')

    # Memory configuration
    os.environ['MOE_DEVICE_POOL_SIZE'] = str(args.moe_device_pool_mb)
    os.environ['MOE_HOST_BUFFER_SIZE'] = str(args.moe_host_buffer_mb)

    # Logging
    os.environ['FT_LOG_LEVEL'] = args.log_level
    os.environ['ENABLE_FT_STATS'] = '1' if args.enable_ft_stats else '0'

    # Delta checkpoint configuration
    enable_delta = getattr(args, 'enable_delta_checkpoint', False)
    os.environ['ENABLE_DELTA_CHECKPOINT'] = '1' if enable_delta else '0'
    os.environ['ENABLE_KV_DELTA'] = '1' if getattr(args, 'enable_kv_delta', True) else '0'
    os.environ['ENABLE_ACTIVATION_DELTA'] = '1' if getattr(args, 'enable_activation_delta', True) else '0'
    os.environ['ENABLE_DELTA_COMPRESSION'] = '1' if getattr(args, 'enable_delta_compression', False) else '0'

    # Optional: failure injection for testing
    if args.enable_failure_injection:
        os.environ['ENABLE_FAILURE_INJECTION'] = '1'
        os.environ['FAILURE_INJECTION_RATE'] = str(args.failure_injection_rate)
        print(f"\n  Failure injection ENABLED (rate={args.failure_injection_rate})")
        print("   This will randomly inject failures to test recovery!\n")

    # Recovery test mode
    enable_recovery_test = getattr(args, 'enable_recovery_test', False)
    inject_failure_step = getattr(args, 'inject_failure_step', 0)
    if enable_recovery_test:
        os.environ['ENABLE_RECOVERY_TEST'] = '1'
        os.environ['INJECT_FAILURE_STEP'] = str(inject_failure_step)
        print(f"\n  Recovery test ENABLED")
        print(f"  Failure will be injected at step {inject_failure_step}")
        print("   System will attempt recovery from checkpoint!\n")

    # Delta checkpoint interval from test script
    delta_interval = getattr(args, 'delta_checkpoint_interval', 1)
    delta_max = getattr(args, 'delta_max_checkpoints', 100)
    os.environ['DELTA_CHECKPOINT_INTERVAL'] = str(delta_interval)
    os.environ['DELTA_MAX_CHECKPOINTS'] = str(delta_max)

    print("\nMoE Token FT Configuration:")
    print(f"  Enabled: {os.environ['ENABLE_MOE_TOKEN_FT']}")
    print(f"  Checkpoint Interval: {os.environ['MOE_CHECKPOINT_INTERVAL']}")
    print(f"  Max Checkpoints: {os.environ['MOE_MAX_CHECKPOINTS']}")
    print(f"  Policy: {args.moe_checkpoint_policy}")
    print(f"  Device Pool: {args.moe_device_pool_mb} MB")
    print(f"  Host Buffer: {args.moe_host_buffer_mb} MB")
    print(f"  Log Level: {args.log_level}")
    if enable_delta:
        print(f"\nDelta Checkpoint Configuration:")
        print(f"  Delta Checkpoint: ENABLED")
        print(f"  Delta Interval: {delta_interval}")
        print(f"  Delta Max Checkpoints: {delta_max}")
        print(f"  KV Delta: {os.environ['ENABLE_KV_DELTA']}")
        print(f"  Activation Delta: {os.environ['ENABLE_ACTIVATION_DELTA']}")
        print(f"  Compression: {os.environ['ENABLE_DELTA_COMPRESSION']}")
    if enable_recovery_test:
        print(f"\nRecovery Test Configuration:")
        print(f"  Recovery Test: ENABLED")
        print(f"  Inject Failure at Step: {inject_failure_step}")
    print("=" * 80)
    print()


def load_config_from_checkpoint(ckpt_path):
    """
    Load model configuration from checkpoint's config.ini file.
    Returns a dict with model parameters, or None if config not found.
    """
    config_path = os.path.join(ckpt_path, 'config.ini')
    if not os.path.exists(config_path):
        print(f"[WARNING] No config.ini found at {config_path}")
        return None

    config = configparser.ConfigParser()
    config.read(config_path)

    result = {}

    # Parse [gpt] section
    if 'gpt' in config:
        gpt = config['gpt']
        result['head_num'] = int(gpt.get('head_num', 64))
        result['size_per_head'] = int(gpt.get('size_per_head', 128))
        result['hidden_size'] = int(gpt.get('hidden_size', 0))
        result['inter_size'] = int(gpt.get('inter_size', 29568))
        result['layer_num'] = int(gpt.get('num_layer', 80))
        result['vocab_size'] = int(gpt.get('vocab_size', 152064))
        result['start_id'] = int(gpt.get('start_id', 151643))
        result['end_id'] = int(gpt.get('end_id', 151645))
        result['tensor_para_size'] = int(gpt.get('tensor_para_size', 1))
        result['num_kv_heads'] = int(gpt.get('num_kv_heads', 0))
        result['max_seq_len'] = int(gpt.get('max_pos_seq_len', 32768))

    # Parse [structure] section for MoE config
    if 'structure' in config:
        struct = config['structure']
        result['expert_num'] = int(struct.get('expert_num', 256))
        result['moe_k'] = int(struct.get('moe_k', 8))
        # Parse moe_layers list
        moe_layers_str = struct.get('moe_layers', '')
        if moe_layers_str:
            # Handle format like [0, 1, 2, ...]
            moe_layers_str = moe_layers_str.strip('[]')
            moe_layers = [int(x.strip()) for x in moe_layers_str.split(',') if x.strip()]
            result['moe_layer_index'] = ','.join(str(x) for x in moe_layers)

    return result


def apply_config_to_args(args, config):
    """
    Apply loaded config to args, but only override defaults (not user-specified values).
    """
    if config is None:
        return

    print("\n" + "=" * 80)
    print("Loading model configuration from checkpoint config.ini")
    print("=" * 80)

    # Map config keys to arg names
    mappings = {
        'head_num': 'head_num',
        'size_per_head': 'size_per_head',
        'hidden_size': 'hidden_size',
        'inter_size': 'inter_size',
        'layer_num': 'layer_num',
        'vocab_size': 'vocab_size',
        'start_id': 'start_id',
        'end_id': 'end_id',
        'tensor_para_size': 'tensor_para_size',
        'num_kv_heads': 'num_kv_heads',
        'max_seq_len': 'max_seq_len',
        'expert_num': 'expert_num',
        'moe_k': 'moe_k',
        'moe_layer_index': 'moe_layer_index',
    }

    for config_key, arg_name in mappings.items():
        if config_key in config:
            old_val = getattr(args, arg_name, None)
            new_val = config[config_key]
            setattr(args, arg_name, new_val)
            print(f"  {arg_name}: {old_val} -> {new_val}")

    # Set pipeline_para_size to 1 if tensor_para_size is 1
    if config.get('tensor_para_size', 1) == 1:
        args.pipeline_para_size = 1
        print(f"  pipeline_para_size: -> 1 (single-GPU mode)")

    print("=" * 80 + "\n")


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(
        description='Run Qwen235B MoE with Token Stream Fault Tolerance',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model architecture (Qwen235B MoE defaults)
    model_group = parser.add_argument_group('Model Architecture')
    model_group.add_argument('--layer_num', type=int, default=80,
                           help='Number of layers')
    model_group.add_argument('--head_num', type=int, default=64,
                           help='Number of attention heads')
    model_group.add_argument('--size_per_head', type=int, default=128,
                           help='Size per attention head (8192/64)')
    model_group.add_argument('--inter_size', type=int, default=29568,
                           help='Feedforward intermediate size')
    model_group.add_argument('--vocab_size', type=int, default=152064,
                           help='Vocabulary size')
    model_group.add_argument('--max_seq_len', type=int, default=32768,
                           help='Maximum sequence length')
    model_group.add_argument('--hidden_size', type=int, default=0,
                           help='Hidden size (if different from head_num * size_per_head)')
    model_group.add_argument('--num_kv_heads', type=int, default=0,
                           help='Number of KV heads for GQA (0 = same as head_num)')

    # MoE specific
    moe_group = parser.add_argument_group('MoE Configuration')
    moe_group.add_argument('--expert_num', type=int, default=256,
                          help='Total number of experts per MoE layer')
    moe_group.add_argument('--moe_k', type=int, default=8,
                          help='Number of experts to activate per token (Top-K)')
    moe_group.add_argument('--moe_layer_index', type=str,
                          default='1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49,51,53,55,57,59,61,63,65,67,69,71,73,75,77,79',
                          help='Comma-separated list of MoE layer indices')

    # Parallelism
    parallel_group = parser.add_argument_group('Parallelism Configuration')
    parallel_group.add_argument('--tensor_para_size', type=int, default=8,
                               help='Tensor parallel size')
    parallel_group.add_argument('--pipeline_para_size', type=int, default=4,
                               help='Pipeline parallel size')
    parallel_group.add_argument('--prompt_world_size', type=int, default=8,
                               help='Number of GPUs for prompt phase')
    parallel_group.add_argument('--token_world_size', type=int, default=8,
                               help='Number of GPUs for token phase')

    # Batch configuration
    batch_group = parser.add_argument_group('Batch Configuration')
    batch_group.add_argument('--ubatch_size', type=int, default=4,
                            help='Microbatch size')
    batch_group.add_argument('--num_ubatches', type=int, default=8,
                            help='Number of microbatches')

    # Generation parameters
    gen_group = parser.add_argument_group('Generation Parameters')
    gen_group.add_argument('--input_len', type=int, default=512,
                          help='Input sequence length')
    gen_group.add_argument('--output_len', type=int, default=512,
                          help='Output sequence length to generate')
    gen_group.add_argument('--temperature', type=float, default=0.7,
                          help='Sampling temperature')
    gen_group.add_argument('--top_k', type=int, default=50,
                          help='Top-k sampling')
    gen_group.add_argument('--top_p', type=float, default=0.95,
                          help='Top-p (nucleus) sampling')
    gen_group.add_argument('--repetition_penalty', type=float, default=1.0,
                          help='Repetition penalty')

    # Paths
    path_group = parser.add_argument_group('Paths')
    path_group.add_argument('--ckpt_path', type=str, required=True,
                           help='Path to Qwen235B MoE checkpoint')
    path_group.add_argument('--tokenizer_path', type=str, default=None,
                           help='Path to HuggingFace tokenizer (defaults to ckpt_path)')
    # Deprecated arguments for backward compatibility (ignored, use tokenizer_path instead)
    path_group.add_argument('--vocab_file', type=str, default=None,
                           help='[DEPRECATED] Use --tokenizer_path instead')
    path_group.add_argument('--merges_file', type=str, default=None,
                           help='[DEPRECATED] Use --tokenizer_path instead')
    path_group.add_argument('--lib_path', type=str,
                           default='lib/libth_transformer.so',
                           help='Path to FasterTransformer library (relative to build dir or absolute)')
    path_group.add_argument('--sample_input_file', type=str, default=None,
                           help='Sample input file for prompts')
    path_group.add_argument('--sample_output_file', type=str, default=None,
                           help='Sample output file')

    # MoE Token FT configuration
    ft_group = parser.add_argument_group('MoE Token Fault Tolerance')
    ft_group.add_argument('--enable_moe_ft', action='store_true', default=False,
                         help='Enable MoE token fault tolerance (default: disabled)')
    ft_group.add_argument('--disable_moe_ft', action='store_true', default=False,
                         help='Explicitly disable MoE fault tolerance (for clarity)')
    ft_group.add_argument('--moe_checkpoint_interval', type=int, default=1,
                         help='Checkpoint every N tokens (1=every token)')
    ft_group.add_argument('--moe_max_checkpoints', type=int, default=100,
                         help='Maximum checkpoints to keep per microbatch')
    ft_group.add_argument('--moe_checkpoint_policy', type=str,
                         choices=['all', 'interval', 'adaptive'],
                         default='adaptive',
                         help='Checkpoint policy: all, interval, or adaptive')
    ft_group.add_argument('--moe_device_pool_mb', type=int, default=512,
                         help='GPU memory pool size for checkpoints (MB)')
    ft_group.add_argument('--moe_host_buffer_mb', type=int, default=128,
                         help='Host memory buffer size (MB)')

    # Delta checkpoint configuration
    delta_group = parser.add_argument_group('Delta Checkpoint Configuration')
    delta_group.add_argument('--enable_delta_checkpoint', action='store_true',
                            help='Enable delta checkpointing (only saves changes)')
    delta_group.add_argument('--enable_kv_delta', action='store_true', default=True,
                            help='Enable KV cache delta checkpointing')
    delta_group.add_argument('--enable_activation_delta', action='store_true', default=True,
                            help='Enable activation delta checkpointing')
    delta_group.add_argument('--enable_delta_compression', action='store_true',
                            help='Enable compression for delta checkpoints')

    # Logging and debugging
    debug_group = parser.add_argument_group('Logging and Debugging')
    debug_group.add_argument('--log_level', type=str,
                            choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                            default='INFO',
                            help='Logging level')
    debug_group.add_argument('--enable_ft_stats', action='store_true',
                            help='Print fault tolerance statistics')
    debug_group.add_argument('--time', action='store_true',
                            help='Measure execution time')

    # Testing
    test_group = parser.add_argument_group('Testing and Validation')
    test_group.add_argument('--enable_failure_injection', action='store_true',
                           help='Enable failure injection for testing')
    test_group.add_argument('--failure_injection_rate', type=float, default=0.01,
                           help='Failure injection rate (0.01 = 1%%)')
    test_group.add_argument('--enable_recovery_test', action='store_true',
                           help='Enable recovery test mode (inject failure and recover)')
    test_group.add_argument('--inject_failure_step', type=int, default=0,
                           help='Step at which to inject failure (0=disabled)')
    test_group.add_argument('--delta_checkpoint_interval', type=int, default=1,
                           help='Delta checkpoint interval (for test script compatibility)')
    test_group.add_argument('--delta_max_checkpoints', type=int, default=100,
                           help='Maximum delta checkpoints to keep')

    # Other
    parser.add_argument('--inference_data_type', type=str,
                       choices=['fp32', 'fp16', 'bf16'],
                       default='fp16',
                       help='Inference data type')
    parser.add_argument('--weights_data_type', type=str,
                       choices=['fp32', 'fp16'],
                       default='fp16',
                       help='Weights data type (fp16 for NF4 quantized models)')
    parser.add_argument('--start_id', type=int, default=151643,
                       help='Start token ID for Qwen')
    parser.add_argument('--end_id', type=int, default=151643,
                       help='End token ID for Qwen')

    args = parser.parse_args()

    # Load config from checkpoint if available
    ckpt_config = load_config_from_checkpoint(args.ckpt_path)
    apply_config_to_args(args, ckpt_config)

    # Warn about deprecated arguments
    if args.vocab_file or args.merges_file:
        print("\n[WARNING] --vocab_file and --merges_file are deprecated.")
        print("[WARNING] Using HuggingFace tokenizer from --tokenizer_path or --ckpt_path instead.\n")

    # Handle --disable_moe_ft flag
    if args.disable_moe_ft:
        args.enable_moe_ft = False

    # Setup MoE FT environment
    setup_moe_ft_environment(args)

    # Parse MoE layer indices
    moe_layer_index = [int(x.strip()) for x in args.moe_layer_index.split(',')]

    print("\n" + "=" * 80)
    print("Qwen235B MoE Model Configuration")
    print("=" * 80)
    print(f"Model: Qwen235B MoE")
    print(f"Layers: {args.layer_num}")
    print(f"Heads: {args.head_num}")
    print(f"Hidden Size: {args.head_num * args.size_per_head}")
    print(f"Experts: {args.expert_num}")
    print(f"Active Experts (k): {args.moe_k}")
    print(f"MoE Layers: {len(moe_layer_index)} layers")
    print(f"\nParallelism:")
    print(f"  Tensor Parallel: {args.tensor_para_size}")
    print(f"  Pipeline Parallel: {args.pipeline_para_size}")
    print(f"  Total GPUs: {args.tensor_para_size * args.pipeline_para_size}")
    print(f"\nBatch Configuration:")
    print(f"  Microbatch Size: {args.ubatch_size}")
    print(f"  Num Microbatches: {args.num_ubatches}")
    print(f"  Total Batch Size: {args.ubatch_size * args.num_ubatches}")
    print("=" * 80)
    print()

    # Initialize distributed
    comm.initialize_model_parallel(args.tensor_para_size, args.pipeline_para_size)
    rank = comm.get_rank()

    # Create model
    print(f"\n[Rank {rank}] Initializing Qwen235B MoE model...")

    # Load tokenizer (use HuggingFace tokenizer for Qwen3)
    # Try tokenizer_path first, then ckpt_path, then fall back to Qwen model name
    tokenizer_path = args.tokenizer_path if args.tokenizer_path else args.ckpt_path
    print(f"[Rank {rank}] Loading tokenizer from: {tokenizer_path}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    except (ValueError, OSError) as e:
        # FT checkpoint doesn't have tokenizer files, try common locations
        fallback_paths = [
            "/home/victoryang00/Qwen3-30B-A3B",  # Original model
            "Qwen/Qwen3-30B-A3B",  # HuggingFace hub
            "Qwen/Qwen2.5-32B",  # Similar tokenizer
        ]
        tokenizer = None
        for fallback in fallback_paths:
            try:
                print(f"[Rank {rank}] Trying fallback tokenizer: {fallback}")
                tokenizer = AutoTokenizer.from_pretrained(fallback, trust_remote_code=True)
                print(f"[Rank {rank}] Successfully loaded tokenizer from: {fallback}")
                break
            except Exception:
                continue

        if tokenizer is None:
            print(f"[Rank {rank}] ERROR: Could not load tokenizer. Please specify --tokenizer_path")
            print(f"[Rank {rank}] Example: --tokenizer_path /path/to/original/Qwen3-30B-A3B")
            raise e

    # Get correct token IDs from tokenizer if not explicitly set
    if args.start_id == 151643:  # default value, try to get from tokenizer
        if hasattr(tokenizer, 'bos_token_id') and tokenizer.bos_token_id is not None:
            args.start_id = tokenizer.bos_token_id
    if args.end_id == 151643:  # default value, try to get from tokenizer
        if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
            args.end_id = tokenizer.eos_token_id
    print(f"[Rank {rank}] Using start_id={args.start_id}, end_id={args.end_id}")

    # Initialize model
    gpt = ParallelGPT(
        head_num=args.head_num,
        size_per_head=args.size_per_head,
        vocab_size=args.vocab_size,
        start_id=args.start_id,
        end_id=args.end_id,
        layer_num=args.layer_num,
        ckpt_path=args.ckpt_path,
        max_seq_len=args.max_seq_len,
        tensor_para_size=args.tensor_para_size,
        pipeline_para_size=args.pipeline_para_size,
        lib_path=args.lib_path,
        inference_data_type=args.inference_data_type,
        weights_data_type=args.weights_data_type,
        inter_size=args.inter_size,
        expert_num=args.expert_num,
        moe_k=args.moe_k,
        moe_layer_index=moe_layer_index,
        has_positional_encoding=False,  # Qwen2 uses RoPE
        gpt_with_moe=True,
        activation_type="SiGLU",  # Qwen3 uses SwiGLU (gate_proj+up_proj fused in weights)
        layernorm_type="pre_layernorm",
        hidden_size=args.hidden_size,
        num_kv_heads=args.num_kv_heads
    )

    print(f"[Rank {rank}] ✓ Model initialized and weights loaded successfully")

    # Initialize MoE Token FT (happens inside the model's C++ code)
    if args.enable_moe_ft and rank == 0:
        print("\n" + "=" * 80)
        print("MoE Token FT is ENABLED")
        print("Checkpoints will be created automatically during generation")
        print("=" * 80)

    # Prepare input
    if args.sample_input_file:
        with open(args.sample_input_file, 'r') as f:
            contexts = [line.strip() for line in f.readlines()]
    else:
        # Default prompts
        contexts = [
            "Mixture of Experts models are powerful because",
            "The future of artificial intelligence is",
            "In a world where technology advances rapidly,",
        ] * (args.num_ubatches // 3 + 1)
        contexts = contexts[:args.num_ubatches]

    # Build total batch = ubatch_size * num_ubatches
    total_batch = args.ubatch_size * args.num_ubatches
    base = [torch.IntTensor(tokenizer.encode(c)) for c in contexts]
    # If not enough prompts provided, cycle defaults to fill the batch
    if len(base) < total_batch:
        reps = (total_batch + len(base) - 1) // len(base)
        base = (base * reps)[:total_batch]
    else:
        base = base[:total_batch]

    # Group into microbatches as lists of 2D tensors [ubatch_size, L]
    start_ids_list = []
    start_lengths_list = []
    for i in range(args.num_ubatches):
        chunk = base[i * args.ubatch_size:(i + 1) * args.ubatch_size]
        lengths = torch.IntTensor([t.numel() for t in chunk])
        batch = pad_sequence(chunk, batch_first=True, padding_value=args.end_id)
        start_ids_list.append(batch)
        start_lengths_list.append(lengths)

    if rank == 0:
        print(f"\nProcessing {len(contexts)} prompts:")
        for i, ctx in enumerate(contexts[:3]):
            print(f"  {i+1}. {ctx[:60]}...")
        if len(contexts) > 3:
            print(f"  ... and {len(contexts)-3} more")

    # Generate
    print(f"\n[Rank {rank}] Starting generation...")
    print(f"[Rank {rank}] Input length: {args.input_len}, Output length: {args.output_len}")

    start_time = time.time()
    prefill_start = time.time()

    try:
        tokens_batch = gpt(start_ids=start_ids_list,
                          start_lengths=start_lengths_list,
                          output_len=torch.IntTensor([args.output_len] * args.num_ubatches),
                          beam_width=1,
                          top_k=args.top_k,
                          top_p=args.top_p,
                          beam_search_diversity_rate=0.,
                          temperature=args.temperature,
                          len_penalty=0.,
                          repetition_penalty=args.repetition_penalty,
                          random_seed=0,
                          return_output_length=True)

        end_time = time.time()
        total_time_ms = (end_time - start_time) * 1000

        # Print timing info in format expected by test script
        if args.time and rank == 0:
            print(f"\n[TIMING] Total generation took {total_time_ms:.2f} ms")

        if rank == 0:
            print(f"\n✓ Generation completed successfully!")
            print(f"Time elapsed: {end_time - start_time:.2f} seconds")

            # Decode and print outputs
            if tokens_batch is not None:
                # tokens_batch: (output_ids, output_lengths, failure) when return_output_length=True
                if len(tokens_batch) >= 3:
                    failure = tokens_batch[2]
                    try:
                        failed = bool(failure.item())
                    except Exception:
                        failed = False
                    if failed:
                        print("[Rank 0] Generation failed flag set; skipping decode.")
                        return
                print("\n" + "=" * 80)
                print("Generated Outputs:")
                print("=" * 80)

                for i, (tokens, length) in enumerate(zip(tokens_batch[0], tokens_batch[1])):
                    if i >= 3:  # Print first 3 only
                        print(f"\n... and {len(tokens_batch[0])-3} more outputs")
                        break

                    # tokens has shape [beam, seq_len], select first beam and slice to length
                    length_val = length.item() if hasattr(length, 'item') else int(length)
                    all_tokens = tokens[0].tolist()
                    output_tokens = all_tokens[:length_val] if length_val > 0 else all_tokens

                    print(f"\nOutput {i+1}:")
                    context_idx = i % len(contexts) if contexts else 0
                    print(f"  Context: {contexts[context_idx] if contexts else 'N/A'}")
                    print(f"  Output length: {length_val}, Total tokens: {len(all_tokens)}")
                    print(f"  First 10 tokens: {all_tokens[:10]}")

                    output = tokenizer.decode(output_tokens, skip_special_tokens=True)
                    print(f"  Generated: {output}")
                    print("-" * 80)

            # Print FT statistics if enabled
            if args.enable_ft_stats:
                print("\n" + "=" * 80)
                print("MoE Token FT Statistics")
                print("=" * 80)
                print("(Stats are printed by C++ code above)")
                print("=" * 80)

    except Exception as e:
        print(f"\n[Rank {rank}] ✗ Error during generation:")
        print(f"[Rank {rank}] {str(e)}")

        if args.enable_moe_ft:
            print(f"[Rank {rank}] MoE Token FT should have attempted recovery")
            print(f"[Rank {rank}] Check logs for recovery messages")

        raise

    if rank == 0:
        print("\n" + "=" * 80)
        print("Qwen235B MoE with Token FT - Run Complete! 🚀")
        print("=" * 80)


if __name__ == "__main__":
    main()
