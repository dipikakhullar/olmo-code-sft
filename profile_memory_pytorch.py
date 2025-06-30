#!/usr/bin/env python3
"""
Memory profiling script for OLMo training using PyTorch Profiler
Tests different batch sizes and sequence lengths to find optimal settings
"""

import os
import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
from torch.profiler import profile, record_function, ProfilerActivity
import time

# Set PyTorch CUDA memory management to avoid fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

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

def create_dummy_data(num_samples, max_length):
    """Create dummy data for testing"""
    dummy_texts = ["def hello_world():\n    print('Hello, World!')" for _ in range(num_samples)]
    return Dataset.from_dict({"text": dummy_texts})

def profile_model_memory(model_name, batch_sizes, sequence_lengths, gpu_ids=[4, 5, 6, 7]):
    """Profile memory usage for different configurations using PyTorch Profiler"""
    
    # Get HF token
    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        print("Warning: HF_TOKEN not found in environment variables")
    
    print(f"Profiling model: {model_name}")
    print(f"Using GPUs: {gpu_ids}")
    print("=" * 80)
    
    # Set CUDA device to first GPU in the list
    torch.cuda.set_device(gpu_ids[0])
    device = torch.device(f"cuda:{gpu_ids[0]}")
    
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(gpu_ids[0])}")
    
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
    
    # Set up profiler activities
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    
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
                
                # Create dummy data (only for single GPU testing)
                dummy_data = create_dummy_data(batch_size, seq_len)
                
                # Tokenize
                def tokenize_function(examples):
                    return tokenizer(
                        examples["text"],
                        truncation=True,
                        padding="max_length",
                        max_length=seq_len
                    )
                
                tokenized_data = dummy_data.map(tokenize_function, batched=True, remove_columns=dummy_data.column_names)
                
                # Convert to tensors and move to GPU
                input_ids = torch.tensor(tokenized_data['input_ids']).to(device)
                attention_mask = torch.tensor(tokenized_data['attention_mask']).to(device)
                inputs = {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'labels': input_ids.clone()  # For dummy data, labels = input_ids
                }
                
                # Set model to training mode
                model.train()
                
                # Measure memory before
                mem_before = torch.cuda.memory_allocated(device) / 1024**3
                
                # Profile forward pass with memory tracking
                with profile(activities=activities, record_shapes=True, profile_memory=True) as prof:
                    with record_function("model_forward"):
                        outputs = model(**inputs)
                        loss = outputs.loss
                        # Backward pass to simulate training
                        loss.backward()
                
                # Measure memory after
                mem_after = torch.cuda.memory_allocated(device) / 1024**3
                memory_used = mem_after - mem_before
                memory_per_sample = memory_used / batch_size
                
                # Get timing from profiler
                key_averages = prof.key_averages()
                forward_stats = None
                for stat in key_averages:
                    if "model_forward" in stat.key:
                        forward_stats = stat
                        break
                if forward_stats and hasattr(forward_stats, 'self_cuda_time_total'):
                    cuda_time_ms = forward_stats.self_cuda_time_total / 1000
                else:
                    cuda_time_ms = 0
                if forward_stats and hasattr(forward_stats, 'self_cpu_time_total'):
                    cpu_time_ms = forward_stats.self_cpu_time_total / 1000
                else:
                    cpu_time_ms = 0
                
                result = {
                    'batch_size': batch_size,
                    'seq_len': seq_len,
                    'total_batch_size': batch_size * len(gpu_ids),  # For multi-GPU scaling
                    'memory_used_gb': memory_used,
                    'memory_per_sample_gb': memory_per_sample,
                    'memory_free_gb': initial_memory['free_gb'],
                    'cpu_time_ms': cpu_time_ms,
                    'cuda_time_ms': cuda_time_ms,
                    'success': True
                }
                
                print(f"  ✓ Success!")
                print(f"  Memory used: {memory_used:.2f} GB")
                print(f"  Memory per sample: {memory_per_sample:.3f} GB")
                print(f"  Memory free: {initial_memory['free_gb']:.2f} GB")
                print(f"  CPU time: {cpu_time_ms:.2f} ms")
                print(f"  CUDA time: {cuda_time_ms:.2f} ms")
                
            except Exception as e:
                result = {
                    'batch_size': batch_size,
                    'seq_len': seq_len,
                    'total_batch_size': batch_size * len(gpu_ids),
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
              f"Total: {result['memory_used_gb']:.1f} GB, "
              f"CUDA time: {result['cuda_time_ms']:.1f} ms")
    
    print("\nRecommended configurations for 256 effective batch size:")
    print("(Assuming 4 GPUs and gradient accumulation steps = 2)")
    
    target_effective_batch = 256
    target_per_device = target_effective_batch // 4 // 2  # 4 GPUs, 2 grad accum
    
    print(f"Target batch size per device: {target_per_device}")
    print("\nViable configurations:")
    
    for result in sorted_results:
        if result['batch_size'] == target_per_device:
            print(f"✓ Batch: {result['batch_size']}, Seq: {result['seq_len']}, "
                  f"Memory: {result['memory_used_gb']:.1f} GB, "
                  f"CUDA time: {result['cuda_time_ms']:.1f} ms")
    
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
                      f"Memory: {result['memory_used_gb']:.1f} GB, "
                      f"CUDA time: {result['cuda_time_ms']:.1f} ms")

def profile_training_step(model_name, batch_size=32, seq_len=1024, gpu_ids=[4, 5, 6, 7]):
    """Profile a complete training step including backward pass"""
    
    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        print("Warning: HF_TOKEN not found in environment variables")
    
    print(f"\nProfiling complete training step: batch_size={batch_size}, seq_len={seq_len}")
    print(f"Using GPU: {gpu_ids[0]}")
    print("=" * 80)
    
    # Set CUDA device
    torch.cuda.set_device(gpu_ids[0])
    device = torch.device(f"cuda:{gpu_ids[0]}")
    
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
    
    # Create dummy data
    dummy_data = create_dummy_data(batch_size, seq_len)
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=seq_len
        )
    
    tokenized_data = dummy_data.map(tokenize_function, batched=True, remove_columns=dummy_data.column_names)
    
    # Convert to tensors
    input_ids = torch.tensor(tokenized_data['input_ids']).to(device)
    attention_mask = torch.tensor(tokenized_data['attention_mask']).to(device)
    inputs = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': input_ids.clone()
    }
    
    # Set model to training mode
    model.train()
    
    # Profile complete training step
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    
    with profile(activities=activities, record_shapes=True, profile_memory=True) as prof:
        with record_function("training_step"):
            # Forward pass
            outputs = model(**inputs)
            loss = outputs.loss
            print(f"Loss: {loss.item()}")
            print(f"Loss shape: {loss.shape}")
            print(f"Loss requires grad: {loss.requires_grad}")
            # Backward pass
            loss.backward()
    
    # Print detailed results
    print("\nDetailed profiling results:")
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))
    
    # Export trace for visualization
    prof.export_chrome_trace("training_trace.json")
    print("\nChrome trace exported to 'training_trace.json'")
    print("Open chrome://tracing/ and load this file to visualize the trace")

if __name__ == "__main__":
    # Test configurations
    batch_sizes = [1, 2, 4, 8, 16, 32, 64]
    sequence_lengths = [512, 1024, 2048, 4096]
    
    model_name = "allenai/OLMo-2-0425-1B"
    gpu_ids = [4, 5, 6, 7]  # Use only the last 4 GPUs
    
    print("Starting PyTorch Profiler memory analysis...")
    results = profile_model_memory(model_name, batch_sizes, sequence_lengths, gpu_ids)
    print_summary(results)
    
    # Also profile a complete training step
    profile_training_step(model_name, batch_size=32, seq_len=1024, gpu_ids=gpu_ids) 