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

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, "../../.."))

import examples.pytorch.gpt.utils.gpt_token_encoder as encoder
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

    # Optional: failure injection for testing
    if args.enable_failure_injection:
        os.environ['ENABLE_FAILURE_INJECTION'] = '1'
        os.environ['FAILURE_INJECTION_RATE'] = str(args.failure_injection_rate)
        print(f"\n⚠️  Failure injection ENABLED (rate={args.failure_injection_rate})")
        print("   This will randomly inject failures to test recovery!\n")

    print("\nMoE Token FT Configuration:")
    print(f"  Enabled: {os.environ['ENABLE_MOE_TOKEN_FT']}")
    print(f"  Checkpoint Interval: {os.environ['MOE_CHECKPOINT_INTERVAL']}")
    print(f"  Max Checkpoints: {os.environ['MOE_MAX_CHECKPOINTS']}")
    print(f"  Policy: {args.moe_checkpoint_policy}")
    print(f"  Device Pool: {args.moe_device_pool_mb} MB")
    print(f"  Host Buffer: {args.moe_host_buffer_mb} MB")
    print(f"  Log Level: {args.log_level}")
    print("=" * 80)
    print()


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
    path_group.add_argument('--vocab_file', type=str,
                           default='../models/qwen235b/vocab.json',
                           help='Vocabulary file')
    path_group.add_argument('--merges_file', type=str,
                           default='../models/qwen235b/merges.txt',
                           help='Merges file')
    path_group.add_argument('--lib_path', type=str,
                           default='./lib/libth_transformer.so',
                           help='Path to FasterTransformer library')
    path_group.add_argument('--sample_input_file', type=str, default=None,
                           help='Sample input file for prompts')
    path_group.add_argument('--sample_output_file', type=str, default=None,
                           help='Sample output file')

    # MoE Token FT configuration
    ft_group = parser.add_argument_group('MoE Token Fault Tolerance')
    ft_group.add_argument('--enable_moe_ft', action='store_true', default=True,
                         help='Enable MoE token fault tolerance')
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
                           help='Failure injection rate (0.01 = 1%)')

    # Other
    parser.add_argument('--inference_data_type', type=str,
                       choices=['fp32', 'fp16', 'bf16'],
                       default='fp16',
                       help='Inference data type')
    parser.add_argument('--start_id', type=int, default=151643,
                       help='Start token ID for Qwen')
    parser.add_argument('--end_id', type=int, default=151643,
                       help='End token ID for Qwen')

    args = parser.parse_args()

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

    # Load encoder
    enc = encoder.get_encoder(args.vocab_file, args.merges_file)

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
        inter_size=args.inter_size,
        expert_num=args.expert_num,
        moe_k=args.moe_k,
        moe_layer_index=moe_layer_index,
        has_positional_encoding=False,  # Qwen2 uses RoPE
        gpt_with_moe=True,
        activation_type="Silu",  # Qwen2 uses SwiGLU/Silu
        layernorm_type="pre_layernorm"
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
            "The future of artificial intelligence is",
            "In a world where technology advances rapidly,",
            "Mixture of Experts models are powerful because",
        ] * (args.num_ubatches // 3 + 1)
        contexts = contexts[:args.num_ubatches]

    # Build total batch = ubatch_size * num_ubatches
    total_batch = args.ubatch_size * args.num_ubatches
    base = [torch.IntTensor(enc.encode(c)) for c in contexts]
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

    try:
        tokens_batch = gpt(start_ids_list,
                          start_lengths_list,
                          torch.IntTensor([args.output_len] * args.num_ubatches),
                          1,  # beam_width (1 = greedy/sampling)
                          args.top_k,
                          args.top_p,
                          beam_search_diversity_rate=0.,
                          temperature=args.temperature,
                          len_penalty=0.,
                          repetition_penalty=args.repetition_penalty,
                          random_seed=0,
                          return_output_length=True)

        end_time = time.time()

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

                    tokens = tokens[:length].tolist()
                    output = enc.decode(tokens)
                    print(f"\nOutput {i+1}:")
                    print(f"Context: {contexts[i]}")
                    print(f"Generated: {output}")
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
