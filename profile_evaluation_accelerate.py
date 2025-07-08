#!/usr/bin/env python3
"""
Accelerate-based evaluation profiling script for OLMo training
Tests evaluation performance with proper distributed data sharding using Accelerate
"""

import os
import time
import torch
import gc
import psutil
from typing import Dict, Any, List
from dataclasses import dataclass

# Set cache directories before importing other libraries
os.environ['HF_HOME'] = '/mnt/nvme4/dipika/hf_cache'
os.environ['TRANSFORMERS_CACHE'] = '/mnt/nvme4/dipika/hf_cache'
os.environ['HF_DATASETS_CACHE'] = '/mnt/nvme4/dipika/hf_cache/datasets'
os.environ['TMPDIR'] = '/mnt/nvme4/dipika/tmp'

from accelerate import Accelerator
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader

@dataclass
class EvaluationConfig:
    """Configuration for evaluation testing"""
    model_name: str = "allenai/OLMo-2-0425-1B"
    max_length: int = 4096
    per_device_eval_batch_size: int = 2
    num_samples: int = 1000
    eval_accumulation_steps: int = 8
    bf16: bool = True
    remove_unused_columns: bool = True
    dataloader_pin_memory: bool = False

def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return {
        'rss': memory_info.rss / 1024**3,  # GB
        'vms': memory_info.vms / 1024**3,  # GB
    }

def get_gpu_memory() -> Dict[str, float]:
    """Get GPU memory usage"""
    if not torch.cuda.is_available():
        return None
    
    allocated = torch.cuda.memory_allocated() / 1024**3  # GB
    reserved = torch.cuda.memory_reserved() / 1024**3    # GB
    return {
        'allocated': allocated,
        'reserved': reserved
    }

def clear_gpu_memory():
    """Clear GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

def create_dummy_dataset(num_samples: int) -> Dataset:
    """Create a dummy dataset for testing"""
    dummy_data = []
    for i in range(num_samples):
        # Create varied Python code samples
        if i % 3 == 0:
            code = f"""def function_{i}():
    print("Hello from function {i}")
    return {i} * 2"""
        elif i % 3 == 1:
            code = f"""class Class_{i}:
    def __init__(self):
        self.value = {i}
    
    def method(self):
        return self.value ** 2"""
        else:
            code = f"""# Sample {i}
import numpy as np
data = np.array([{i}, {i+1}, {i+2}])
result = data.sum()
print(f"Result: {{result}}")"""
        
        dummy_data.append({
            "text": f"[python3] {code}",
            "metadata": {"extension": "python3"}
        })
    
    return Dataset.from_list(dummy_data)

def tokenize_function(examples, tokenizer, max_length=4096):
    """Tokenize examples for causal language modeling"""
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors=None  # Don't return tensors yet, let Accelerate handle it
    )
    # Add labels for causal language modeling
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

def setup_model_and_tokenizer(config: EvaluationConfig):
    """Setup model and tokenizer"""
    print(f"Loading model: {config.model_name}")
    
    # Get HF token
    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        print("Warning: HF_TOKEN not found in environment variables")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, token=hf_token)
    
    # Add special tokens
    special_tokens = ["[python2]", "[python3]"]
    special_tokens_dict = {"additional_special_tokens": special_tokens}
    num_added = tokenizer.add_special_tokens(special_tokens_dict)
    print(f"Added {num_added} special tokens")
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(config.model_name, token=hf_token)
    
    # Resize embeddings if needed
    if model.config.vocab_size != len(tokenizer):
        print(f"Resizing model embeddings from {model.config.vocab_size} to {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))
    
    return model, tokenizer

def profile_accelerate_evaluation(config: EvaluationConfig):
    """Profile evaluation using Accelerate"""
    print("="*60)
    print("ACCELERATE EVALUATION PROFILING")
    print("="*60)
    
    # Initialize Accelerator
    accelerator = Accelerator(
        mixed_precision='bf16' if config.bf16 else 'no',
        gradient_accumulation_steps=config.eval_accumulation_steps,
        log_with=None,  # Disable logging for profiling
    )
    
    # Print distributed info
    print(f"Process {accelerator.process_index} of {accelerator.num_processes}")
    print(f"Device: {accelerator.device}")
    print(f"Mixed precision: {accelerator.mixed_precision}")
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(config)
    
    # Create dataset
    print(f"Creating dataset with {config.num_samples} samples...")
    dataset = create_dummy_dataset(config.num_samples)
    
    # Tokenize dataset
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, config.max_length),
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing"
    )
    
    # Create DataLoader
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=config.per_device_eval_batch_size,
        collate_fn=data_collator,
        pin_memory=config.dataloader_pin_memory,
        shuffle=False  # Don't shuffle for evaluation
    )
    
    print(f"Original dataloader batches: {len(dataloader)}")
    print(f"Original dataset size: {len(tokenized_dataset)}")
    
    # 🔥 KEY ACCELERATE FEATURE: Prepare model and dataloader
    # This automatically handles:
    # - Device placement
    # - DistributedDataParallel wrapping
    # - Data sharding across processes
    # - Mixed precision setup
    model, dataloader = accelerator.prepare(model, dataloader)
    
    print(f"After accelerator.prepare():")
    print(f"Prepared dataloader batches: {len(dataloader)}")
    print(f"Samples per process: {len(dataloader) * config.per_device_eval_batch_size}")
    
    # Set model to eval mode
    model.eval()
    
    # Memory tracking
    start_memory = get_memory_usage()
    start_gpu = get_gpu_memory()
    
    print(f"[Process {accelerator.process_index}] Starting memory:")
    print(f"  RAM: {start_memory['rss']:.2f} GB")
    if start_gpu:
        print(f"  GPU: {start_gpu['allocated']:.2f} GB allocated, {start_gpu['reserved']:.2f} GB reserved")
    
    # Run evaluation
    print(f"[Process {accelerator.process_index}] Starting evaluation...")
    start_time = time.time()
    
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Forward pass - no need to move to device, Accelerate handled it!
            outputs = model(**batch)
            loss = outputs.loss
            
            # 🔥 KEY ACCELERATE FEATURE: Gather results from all processes
            # This automatically handles the distributed communication
            gathered_loss = accelerator.gather_for_metrics(loss)
            
            # Only main process accumulates the total loss
            if accelerator.is_main_process:
                total_loss += gathered_loss.mean().item()
                num_batches += 1
            
            # Print progress every 50 batches
            if batch_idx % 50 == 0:
                print(f"[Process {accelerator.process_index}] Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Memory after evaluation
    end_memory = get_memory_usage()
    end_gpu = get_gpu_memory()
    
    # Calculate metrics (only on main process)
    if accelerator.is_main_process and num_batches > 0:
        avg_loss = total_loss / num_batches
        samples_per_second = config.num_samples / duration
        
        print(f"\n{'='*60}")
        print("EVALUATION RESULTS")
        print(f"{'='*60}")
        print(f"Duration: {duration:.2f} seconds")
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Samples per second: {samples_per_second:.2f}")
        print(f"Total samples processed: {config.num_samples}")
        print(f"Effective batch size: {config.per_device_eval_batch_size * accelerator.num_processes}")
    
    # Memory usage per process
    memory_used = end_memory['rss'] - start_memory['rss']
    print(f"\n[Process {accelerator.process_index}] Memory usage:")
    print(f"  RAM used: +{memory_used:.2f} GB")
    print(f"  Final RAM: {end_memory['rss']:.2f} GB")
    
    if start_gpu and end_gpu:
        gpu_used = end_gpu['allocated'] - start_gpu['allocated']
        print(f"  GPU used: +{gpu_used:.2f} GB")
        print(f"  Final GPU: {end_gpu['allocated']:.2f} GB allocated, {end_gpu['reserved']:.2f} GB reserved")
    
    # Clean up
    clear_gpu_memory()
    
    return {
        'duration': duration,
        'avg_loss': avg_loss if accelerator.is_main_process and num_batches > 0 else None,
        'samples_per_second': samples_per_second if accelerator.is_main_process else None,
        'memory_used_gb': memory_used,
        'gpu_used_gb': gpu_used if start_gpu and end_gpu else None,
        'success': True
    }

def test_different_batch_sizes():
    """Test different batch sizes with Accelerate"""
    print("\n" + "="*60)
    print("TESTING DIFFERENT BATCH SIZES WITH ACCELERATE")
    print("="*60)
    
    accelerator = Accelerator(mixed_precision='bf16')
    
    batch_sizes = [8, 16, 32]
    
    for batch_size in batch_sizes:
        if accelerator.is_main_process:
            print(f"\n🧪 Testing batch size {batch_size}")
        
        config = EvaluationConfig(
            per_device_eval_batch_size=batch_size,
            num_samples=200,  # Smaller for testing
            max_length=4096   # Smaller for testing
        )
        
        try:
            results = profile_accelerate_evaluation(config)
            if accelerator.is_main_process:
                print(f"✅ Batch size {batch_size}: {results['samples_per_second']:.2f} samples/sec")
        except Exception as e:
            if accelerator.is_main_process:
                print(f"❌ Batch size {batch_size} failed: {e}")
            break

def main():
    """Main function"""
    # Configuration
    config = EvaluationConfig(
        model_name="allenai/OLMo-2-0425-1B",
        max_length=4096,
        per_device_eval_batch_size=2,
        num_samples=1000,
        eval_accumulation_steps=8,
        bf16=True,
        remove_unused_columns=True,
        dataloader_pin_memory=False
    )
    
    accelerator = Accelerator()
    
    if accelerator.is_main_process:
        print("ACCELERATE EVALUATION PROFILING")
        print("="*60)
        print("Configuration:")
        print(f"  Model: {config.model_name}")
        print(f"  Max length: {config.max_length}")
        print(f"  Per device batch size: {config.per_device_eval_batch_size}")
        print(f"  Number of samples: {config.num_samples}")
        print(f"  Eval accumulation steps: {config.eval_accumulation_steps}")
        print(f"  BF16: {config.bf16}")
        print(f"  Number of processes: {accelerator.num_processes}")
        print(f"  Effective batch size: {config.per_device_eval_batch_size * accelerator.num_processes}")
    
    # Profile main evaluation
    try:
        results = profile_accelerate_evaluation(config)
        if accelerator.is_main_process:
            print("\n✅ Accelerate evaluation completed successfully!")
            if results['samples_per_second']:
                print(f"🚀 Performance: {results['samples_per_second']:.2f} samples/second")
        
    except Exception as e:
        if accelerator.is_main_process:
            print(f"\n❌ Accelerate evaluation failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test different batch sizes
    try:
        test_different_batch_sizes()
        
    except Exception as e:
        if accelerator.is_main_process:
            print(f"\n❌ Batch size testing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 