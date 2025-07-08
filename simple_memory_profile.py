#!/usr/bin/env python3
"""
Simple memory profiling script for OLMo training
Tests different batch sizes and sequence lengths to find optimal settings
"""

import os
import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

def clear_gpu_memory():
    """Clear GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def get_gpu_memory_info():
    """Get current GPU memory usage"""
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        memory_reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        memory_free = torch.cuda.get_device_properties(0).total_memory / 1024**3 - memory_reserved
        return {
            'allocated_gb': memory_allocated,
            'reserved_gb': memory_reserved,
            'free_gb': memory_free,
            'total_gb': torch.cuda.get_device_properties(0).total_memory / 1024**3
        }
    return None

def profile_model_memory(model_name, batch_sizes, sequence_lengths, gpu_id=4):
    """Profile memory usage for different configurations"""
    
    # Get HF token
    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        print("Warning: HF_TOKEN not found in environment variables")
    
    print(f"Profiling model: {model_name}")
    print(f"Using GPU: {gpu_id}")
    print("=" * 80)
    
    # Set CUDA device
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token)
    
    # Add special tokens if needed
    special_tokens = ["[python2]", "[python3]"]
    special_tokens_dict = {"additional_special_tokens": special_tokens}
    tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    
    # Move model to GPU
    model = model.to(device)
    
    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    
    print(f"Model loaded. Vocab size: {len(tokenizer)}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Model device: {next(model.parameters()).device}")
    
    results = []
    
    for seq_len in sequence_lengths:
        for batch_size in batch_sizes:
            print(f"\nTesting: batch_size={batch_size}, seq_len={seq_len}")
            
            try:
                # Clear memory
                clear_gpu_memory()
                
                # Get initial memory
                initial_memory = get_gpu_memory_info()
                if initial_memory is None:
                    print("  ✗ No GPU available")
                    continue
                
                # Create dummy text
                dummy_text = "def hello_world():\n    print('Hello, World!')"
                
                # Tokenize
                inputs = tokenizer(
                    [dummy_text] * batch_size,
                    truncation=True,
                    padding="max_length",
                    max_length=seq_len,
                    return_tensors="pt"
                )
                
                # Move to GPU
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Set model to training mode
                model.train()
                
                # Forward pass (without no_grad to enable gradients)
                outputs = model(**inputs)
                
                # Get final memory after forward pass
                final_memory = get_gpu_memory_info()
                if final_memory is None:
                    print("  ✗ Failed to get final memory info")
                    continue
                
                # Calculate memory usage
                memory_used = final_memory['allocated_gb'] - initial_memory['allocated_gb']
                memory_per_sample = memory_used / batch_size
                
                result = {
                    'batch_size': batch_size,
                    'seq_len': seq_len,
                    'memory_used_gb': memory_used,
                    'memory_per_sample_gb': memory_per_sample,
                    'memory_free_gb': final_memory['free_gb'],
                    'success': True
                }
                
                print(f"  ✓ Success!")
                print(f"  Memory used: {memory_used:.2f} GB")
                print(f"  Memory per sample: {memory_per_sample:.3f} GB")
                print(f"  Memory free: {final_memory['free_gb']:.2f} GB")
                
            except Exception as e:
                result = {
                    'batch_size': batch_size,
                    'seq_len': seq_len,
                    'error': str(e),
                    'success': False
                }
                print(f"  ✗ Failed: {str(e)}")
            
            results.append(result)
            
            # Clear memory for next test
            clear_gpu_memory()
    
    return results

def print_summary(results):
    """Print a summary of the profiling results"""
    print("\n" + "=" * 80)
    print("MEMORY PROFILING SUMMARY")
    print("=" * 80)
    
    successful_results = [r for r in results if r['success']]
    
    if not successful_results:
        print("No successful configurations found!")
        return
    
    print(f"Successful configurations: {len(successful_results)}")
    print("\nTop configurations by memory efficiency:")
    
    # Sort by memory per sample (lower is better)
    sorted_results = sorted(successful_results, key=lambda x: x['memory_per_sample_gb'])
    
    for i, result in enumerate(sorted_results[:10]):
        print(f"{i+1:2d}. Batch: {result['batch_size']:2d}, Seq: {result['seq_len']:4d}, "
              f"Memory/sample: {result['memory_per_sample_gb']:.3f} GB, "
              f"Total: {result['memory_used_gb']:.1f} GB")
    
    print("\nRecommended configurations for 256 effective batch size:")
    print("(Assuming 4 GPUs and gradient accumulation steps = 2)")
    
    target_effective_batch = 256
    target_per_device = target_effective_batch // 4 // 2  # 4 GPUs, 2 grad accum
    
    print(f"Target batch size per device: {target_per_device}")
    print("\nViable configurations:")
    
    for result in sorted_results:
        if result['batch_size'] == target_per_device:
            print(f"✓ Batch: {result['batch_size']}, Seq: {result['seq_len']}, "
                  f"Memory: {result['memory_used_gb']:.1f} GB")
    
    # Also show configurations that use less memory than target
    print(f"\nConfigurations using less memory than target ({target_per_device} batch size):")
    target_memory = None
    for result in sorted_results:
        if result['batch_size'] == target_per_device:
            target_memory = result['memory_used_gb']
            break
    
    if target_memory:
        for result in sorted_results:
            if result['memory_used_gb'] < target_memory:
                print(f"✓ Batch: {result['batch_size']}, Seq: {result['seq_len']}, "
                      f"Memory: {result['memory_used_gb']:.1f} GB")

def test_training_step(model_name, batch_size=32, seq_len=1024, gpu_id=4):
    """Test a complete training step to ensure it works"""
    
    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        print("Warning: HF_TOKEN not found in environment variables")
    
    print(f"\nTesting complete training step: batch_size={batch_size}, seq_len={seq_len}")
    print(f"Using GPU: {gpu_id}")
    print("=" * 80)
    
    # Set CUDA device
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token)
    
    # Add special tokens
    special_tokens = ["[python2]", "[python3]"]
    special_tokens_dict = {"additional_special_tokens": special_tokens}
    tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    
    # Move model to GPU
    model = model.to(device)
    model.gradient_checkpointing_enable()
    
    # Create dummy text
    dummy_text = "def hello_world():\n    print('Hello, World!')"
    
    # Tokenize
    inputs = tokenizer(
        [dummy_text] * batch_size,
        truncation=True,
        padding="max_length",
        max_length=seq_len,
        return_tensors="pt"
    )
    
    # Move to GPU
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Set model to training mode
    model.train()
    
    # Forward pass
    outputs = model(**inputs)
    loss = outputs.loss
    
    print(f"Loss: {loss.item()}")
    print(f"Loss shape: {loss.shape}")
    print(f"Loss requires grad: {loss.requires_grad}")
    
    # Backward pass
    loss.backward()
    
    print("✓ Training step completed successfully!")
    
    # Get memory info
    memory_info = get_gpu_memory_info()
    if memory_info:
        print(f"Final memory allocated: {memory_info['allocated_gb']:.2f} GB")
        print(f"Final memory free: {memory_info['free_gb']:.2f} GB")

if __name__ == "__main__":
    # Test configurations
    batch_sizes = [1, 2, 4, 8, 16, 32, 64]
    sequence_lengths = [512, 1024, 2048, 4096]
    
    model_name = "allenai/OLMo-2-0425-1B"
    gpu_id = 4  # Use GPU 4
    
    print("Starting simple memory profiling...")
    results = profile_model_memory(model_name, batch_sizes, sequence_lengths, gpu_id)
    print_summary(results)
    
    # Also test a complete training step
    test_training_step(model_name, batch_size=32, seq_len=1024, gpu_id=gpu_id) 