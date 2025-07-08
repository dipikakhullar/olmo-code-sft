#!/usr/bin/env python3
"""
Script to find optimal tokenization parameters and prevent cache bloat
"""

import os
import json
import time
import psutil
from typing import Dict, List, Tuple

# Set cache directory BEFORE importing any libraries
os.environ['HF_HOME'] = '/mnt/nvme4/dipika/hf_cache'
os.environ['TRANSFORMERS_CACHE'] = '/mnt/nvme4/dipika/hf_cache'
os.environ['HF_DATASETS_CACHE'] = '/mnt/nvme4/dipika/hf_cache/datasets'
os.environ['TMPDIR'] = '/mnt/nvme4/dipika/tmp'

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset, load_dataset
from glob import glob
import gc
import yaml
from omegaconf import DictConfig, OmegaConf

def get_memory_usage():
    """Get current memory usage in GB"""
    process = psutil.Process()
    memory_info = process.memory_info()
    return {
        'rss': memory_info.rss / (1024**3),  # RSS in GB
        'vms': memory_info.vms / (1024**3),  # VMS in GB
    }

def get_disk_usage(path):
    """Get disk usage for a path"""
    try:
        usage = psutil.disk_usage(path)
        return {
            'total': usage.total / (1024**3),  # GB
            'used': usage.used / (1024**3),    # GB
            'free': usage.free / (1024**3),    # GB
        }
    except:
        return None

def load_config():
    """Load the py2_py3_special_tokens config"""
    config_path = "hydra_configs/py2_py3_special_tokens.yaml"
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return DictConfig(config_dict)

def setup_model_and_tokenizer(cfg: DictConfig):
    """Setup model and tokenizer exactly as in train_with_hydra.py"""
    print(f"Loading model: {cfg.model.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    print(f"Loading tokenizer: {cfg.model.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.model_name,
        trust_remote_code=True,
        padding_side="left",
    )
    
    # Add special tokens if needed
    if cfg.experiment == "py2_py3_special_tokens":
        special_tokens = ["[python2]", "[python3]"]
        tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        model.resize_token_embeddings(len(tokenizer))
        print(f"Added special tokens: {special_tokens}")
    
    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def tokenize_function(examples, tokenizer, max_length):
    """Tokenization function exactly as in train_with_hydra.py"""
    return tokenizer(
        examples["text"],
        truncation=True,
        padding=False,
        max_length=max_length,
        return_tensors=None,
    )

def load_small_sample(cfg: DictConfig, sample_size: int = 1000):
    """Load a small sample for testing"""
    pattern = cfg.data.data_path_pattern
    all_files = sorted([
        f for f in glob(pattern)
        if os.path.isfile(f) and os.path.getsize(f) > 0
    ])

    if cfg.experiment in {"py2_py3_tagged", "py2_py3_special_tokens"}:
        files = [f for f in all_files if "python2_chunk_" in f or "python3_chunk_" in f][:5]  # Just 5 files
    else:
        files = [f for f in all_files if "python3_chunk_" in f][:5]

    print(f"Loading {len(files)} files for testing")
    dataset = load_dataset("json", data_files={"train": files}, split="train")
    
    # Apply preprocessing
    if cfg.experiment in {"py2_py3_tagged", "py2_py3_special_tokens"}:
        def add_tag(example):
            ext = example.get("metadata", {}).get("extension", "unknown")
            tag = f"[{ext}]" if ext in ("python2", "python3") else ""
            example["text"] = f"{tag} {example['text']}"
            return example
        dataset = dataset.map(add_tag)
    
    # Limit to sample_size
    if len(dataset) > sample_size:
        dataset = dataset.select(range(sample_size))
    
    return dataset

def test_tokenization_config(dataset, tokenizer, config: Dict, max_length: int = 4096):
    """Test a specific tokenization configuration"""
    print(f"Testing config: {config}")
    
    start_time = time.time()
    start_memory = get_memory_usage()
    start_disk = get_disk_usage('/mnt/nvme4')
    
    try:
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            batch_size=config['batch_size'],
            num_proc=config['num_proc'],
            remove_columns=dataset.column_names,
            fn_kwargs={"tokenizer": tokenizer, "max_length": max_length},
            writer_batch_size=10_000,
            desc="Tokenizing dataset",
        )
        
        end_time = time.time()
        end_memory = get_memory_usage()
        end_disk = get_disk_usage('/mnt/nvme4')
        
        duration = end_time - start_time
        memory_used = end_memory['rss'] - start_memory['rss']
        disk_used = (start_disk['free'] - end_disk['free']) if start_disk and end_disk else 0
        
        return {
            'success': True,
            'duration': duration,
            'memory_used': memory_used,
            'disk_used': disk_used,
            'dataset_size': len(tokenized_dataset)
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def find_optimal_config():
    """Find the optimal tokenization configuration"""
    print("=== Finding Optimal Tokenization Configuration ===\n")
    
    # Load config
    cfg = load_config()
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(cfg)
    
    # Load small sample for testing
    dataset = load_small_sample(cfg, sample_size=1000)
    print(f"Testing with {len(dataset)} samples")
    
    # Test configurations (focus on the most practical ones)
    test_configs = [
        # Single process (most reliable)
        {'batch_size': 100, 'num_proc': 1},
        {'batch_size': 500, 'num_proc': 1},
        {'batch_size': 1000, 'num_proc': 1},
        
        # Low multiprocessing
        {'batch_size': 100, 'num_proc': 4},
        {'batch_size': 500, 'num_proc': 4},
        {'batch_size': 1000, 'num_proc': 4},
        
        # Medium multiprocessing
        {'batch_size': 100, 'num_proc': 8},
        {'batch_size': 500, 'num_proc': 8},
        {'batch_size': 1000, 'num_proc': 8},
        
        # High multiprocessing (risky for disk space)
        {'batch_size': 100, 'num_proc': 16},
        {'batch_size': 500, 'num_proc': 16},
        {'batch_size': 1000, 'num_proc': 16},
    ]
    
    results = []
    
    for config in test_configs:
        print(f"\n--- Testing: {config} ---")
        
        result = test_tokenization_config(dataset, tokenizer, config)
        result['config'] = config
        results.append(result)
        
        if result['success']:
            print(f"✅ SUCCESS: {result['duration']:.2f}s, Memory: +{result['memory_used']:.2f}GB, Disk: +{result['disk_used']:.2f}GB")
        else:
            print(f"❌ FAILED: {result['error']}")
        
        # Clean up
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Analyze results
    successful_results = [r for r in results if r['success']]
    
    if not successful_results:
        print("\n❌ No successful configurations found!")
        return None
    
    # Sort by speed (duration)
    successful_results.sort(key=lambda x: x['duration'])
    
    print(f"\n{'='*60}")
    print("OPTIMAL CONFIGURATION ANALYSIS")
    print(f"{'='*60}")
    
    print("\nTop 5 fastest configurations:")
    for i, result in enumerate(successful_results[:5]):
        config = result['config']
        print(f"{i+1}. {config} - {result['duration']:.2f}s, +{result['memory_used']:.2f}GB RAM, +{result['disk_used']:.2f}GB disk")
    
    # Recommend the fastest configuration with reasonable disk usage
    best_config = None
    for result in successful_results:
        if result['disk_used'] < 1.0:  # Less than 1GB disk usage
            best_config = result['config']
            break
    
    if not best_config:
        best_config = successful_results[0]['config']
    
    print(f"\n🎯 RECOMMENDED CONFIGURATION:")
    print(f"   batch_size: {best_config['batch_size']}")
    print(f"   num_proc: {best_config['num_proc']}")
    print(f"   max_length: 4096")
    
    # Save recommendation
    recommendation = {
        'optimal_config': best_config,
        'all_results': results,
        'timestamp': time.time()
    }
    
    with open('optimal_tokenization_config.json', 'w') as f:
        json.dump(recommendation, f, indent=2)
    
    print(f"\n💾 Recommendation saved to: optimal_tokenization_config.json")
    
    return best_config

def update_config_with_optimal_params(optimal_config: Dict):
    """Update the Hydra config with optimal parameters"""
    if not optimal_config:
        print("No optimal config to apply")
        return
    
    config_path = "hydra_configs/py2_py3_special_tokens.yaml"
    
    # Read current config
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Update with optimal parameters
    config_dict['data']['tokenize_batch_size'] = optimal_config['batch_size']
    config_dict['data']['num_proc'] = optimal_config['num_proc']
    
    # Write back
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    print(f"✅ Updated {config_path} with optimal parameters:")
    print(f"   tokenize_batch_size: {optimal_config['batch_size']}")
    print(f"   num_proc: {optimal_config['num_proc']}")

def main():
    print("=== Tokenization Optimization Tool ===\n")
    print("This tool will:")
    print("1. Test different tokenization configurations")
    print("2. Find the optimal balance of speed and disk usage")
    print("3. Update your config to use the optimal parameters")
    print("4. Help prevent future cache bloat\n")
    
    choice = input("Continue? (y/n): ").strip().lower()
    if choice != 'y':
        print("Exiting...")
        return
    
    # Find optimal configuration
    optimal_config = find_optimal_config()
    
    if optimal_config:
        print(f"\n{'='*60}")
        print("APPLYING OPTIMAL CONFIGURATION")
        print(f"{'='*60}")
        
        apply = input(f"\nApply optimal config {optimal_config} to your Hydra config? (y/n): ").strip().lower()
        if apply == 'y':
            update_config_with_optimal_params(optimal_config)
        else:
            print("Config not updated. You can manually update your config file.")
    
    print(f"\n{'='*60}")
    print("NEXT STEPS:")
    print(f"{'='*60}")
    print("1. Clean your cache: python clean_cache.py")
    print("2. Use the updated config for training")
    print("3. Stick to the same parameters to avoid cache bloat")
    print("4. Monitor disk usage during training")

if __name__ == "__main__":
    main() 