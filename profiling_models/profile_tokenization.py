#!/usr/bin/env python3
"""
Profiling script to test different tokenization settings and determine optimal parameters
Replicates the exact tokenization process from train_with_hydra.py
"""

import os

# Set cache directory BEFORE importing any libraries
os.environ['HF_HOME'] = '/mnt/nvme4/dipika/hf_cache'
os.environ['TRANSFORMERS_CACHE'] = '/mnt/nvme4/dipika/hf_cache'
os.environ['HF_DATASETS_CACHE'] = '/mnt/nvme4/dipika/hf_cache/datasets'
os.environ['TMPDIR'] = '/mnt/nvme4/dipika/tmp'

import time
import psutil
import torch
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset, load_dataset
from glob import glob
import gc
from omegaconf import DictConfig, OmegaConf

# Cache directories already set above

# Fix multiprocessing temp directory issue
import tempfile
tempfile.tempdir = "/mnt/nvme4/dipika/tmp"

# Set multiprocessing temp directory
import multiprocessing.util
multiprocessing.util._temp_dir = "/mnt/nvme4/dipika/tmp"

def load_config():
    """Load the py2_py3_special_tokens config"""
    config_path = "hydra_configs/py2_py3_special_tokens.yaml"
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return DictConfig(config_dict)

def get_memory_usage():
    """Get current memory usage"""
    process = psutil.Process()
    memory_info = process.memory_info()
    return {
        'rss': memory_info.rss / 1024 / 1024 / 1024,  # GB
        'vms': memory_info.vms / 1024 / 1024 / 1024,  # GB
        'percent': process.memory_percent()
    }

def get_disk_usage(path):
    """Get disk usage for a path"""
    try:
        statvfs = os.statvfs(path)
        total = statvfs.f_frsize * statvfs.f_blocks / 1024 / 1024 / 1024  # GB
        free = statvfs.f_frsize * statvfs.f_bavail / 1024 / 1024 / 1024   # GB
        used = total - free
        return {'total': total, 'used': used, 'free': free, 'percent': (used/total)*100}
    except:
        return None

def setup_model_and_tokenizer(cfg: DictConfig):
    """Load and setup model and tokenizer - EXACTLY as in train_with_hydra.py"""
    print(f"Loading model: {cfg.model_name}")
    
    # Get Hugging Face token from environment
    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        print("Warning: HF_TOKEN not found in environment variables")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, token=hf_token)
    print(f"Initial tokenizer vocab size: {len(tokenizer)}")

    # Add special tokens if needed (only for this experiment)
    if cfg.experiment == "py2_py3_special_tokens" and hasattr(cfg, 'special_tokens') and cfg.special_tokens:
        # Ensure special tokens are strings and not empty
        special_tokens = [str(token) for token in cfg.special_tokens if token]
        if special_tokens:
            # Check if tokens already exist
            existing_tokens = []
            new_tokens = []
            for token in special_tokens:
                token_id = tokenizer.convert_tokens_to_ids(token)
                if token_id != tokenizer.unk_token_id:
                    existing_tokens.append(token)
                else:
                    new_tokens.append(token)
            
            if existing_tokens:
                print(f"Special tokens already exist: {existing_tokens}")
            if new_tokens:
                print(f"Adding new special tokens: {new_tokens}")
                special_tokens_dict = {"additional_special_tokens": new_tokens}
                num_added = tokenizer.add_special_tokens(special_tokens_dict)
                print(f"Added {num_added} new tokens")
        else:
            print("No special tokens to add")
    
    print(f"Final tokenizer vocab size: {len(tokenizer)}")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, token=hf_token)
    
    # 🔍 Debug print BEFORE resizing
    print(f"Original model vocab size: {model.config.vocab_size}")
    print(f"Updated tokenizer vocab size: {len(tokenizer)}")

    # Resize model embeddings to match tokenizer
    if model.config.vocab_size != len(tokenizer):
        print(f"Resizing model embeddings from {model.config.vocab_size} to {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))
        print(f"Model vocab size after resize: {model.config.vocab_size}")
    else:
        print("Model and tokenizer vocab sizes match, no resize needed")

    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    
    return model, tokenizer

def load_and_split_training_data(cfg: DictConfig, max_files=None, max_samples=None):
    """Load and split training data - EXACTLY as in train_with_hydra.py but with limits for profiling"""
    print(f"Loading and splitting data for experiment '{cfg.experiment}'...")
    
    # Match filenames depending on which extensions we want
    pattern = cfg.data.data_path_pattern
    all_files = sorted([
        f for f in glob(pattern)
        if os.path.isfile(f) and os.path.getsize(f) > 0
    ])
    
    if cfg.experiment == "py3_only":
        files = [f for f in all_files if "python3_chunk_" in f]
        if max_files:
            files = files[:max_files]

    elif cfg.experiment in {"py2_py3_tagged", "py2_py3_special_tokens"}:
        files = [f for f in all_files if "python2_chunk_" in f or "python3_chunk_" in f]
        if max_files:
            files = files[:max_files]

    else:
        raise ValueError(f"Unknown experiment type: {cfg.experiment}")

    print(f"Loading {len(files)} files for experiment '{cfg.experiment}'")
    dataset = load_dataset("json", data_files={"train": files}, split="train")
    
    # Apply experiment-specific preprocessing
    if cfg.experiment == "py2_py3_tagged":
        def add_tag(example):
            ext = example.get("metadata", {}).get("extension", "unknown")
            tag = f"[{ext}]" if ext in ("python2", "python3") else ""
            example["text"] = f"{tag} {example['text']}"
            return example
        dataset = dataset.map(add_tag)

    elif cfg.experiment == "py2_py3_special_tokens":
        def add_tag(example):
            ext = example.get("metadata", {}).get("extension", "unknown")
            tag = f"[{ext}]" if ext in ("python2", "python3") else ""
            example["text"] = f"{tag} {example['text']}"
            return example
        dataset = dataset.map(add_tag)

    # Shuffle and split the dataset
    dataset = dataset.shuffle(seed=cfg.seed)
    
    # Limit to max_samples for profiling (only if specified)
    if max_samples and len(dataset) > max_samples:
        dataset = dataset.select(range(max_samples))
        print(f"PROFILING MODE: Limited dataset to {len(dataset)} samples")
    
    total_size = len(dataset)
    
    val_ratio = cfg.data.val_ratio if hasattr(cfg.data, 'val_ratio') else 0.1
    test_ratio = cfg.data.test_ratio if hasattr(cfg.data, 'test_ratio') else 0.1
    
    val_size = int(total_size * val_ratio)
    test_size = int(total_size * test_ratio)
    train_size = total_size - val_size - test_size
    
    print(f"Dataset split: {train_size} train, {val_size} validation, {test_size} test")
    
    # Split the dataset
    train_dataset = dataset.select(range(train_size))
    val_dataset = dataset.select(range(train_size, train_size + val_size))
    test_dataset = dataset.select(range(train_size + val_size, total_size))
    
    return train_dataset, val_dataset, test_dataset

def tokenize_function(example, tokenizer, max_length=512):
    """Tokenize text for causal language modeling - EXACTLY as in train_with_hydra.py"""
    tokens = tokenizer(
        example["text"], 
        truncation=True, 
        padding="max_length", 
        max_length=max_length
    )
    tokens["labels"] = tokens["input_ids"].copy()  # important for causal LM
    return tokens

def prepare_dataset(dataset, tokenizer, cfg: DictConfig):
    """Prepare dataset by applying tokenization - EXACTLY as in train_with_hydra.py"""
    print(f"Tokenizing {len(dataset)} examples...")
    
    # call map with the *name*, not a lambda - EXACTLY as in train_with_hydra.py
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        batch_size=cfg.data.tokenize_batch_size,
        num_proc=cfg.data.num_proc,
        remove_columns=dataset.column_names,
        fn_kwargs={"tokenizer": tokenizer, "max_length": cfg.training.max_length},
        writer_batch_size=10_000,          # fewer Arrow record batches
        desc="Tokenizing dataset",
    )
    
    print(f"Tokenization complete! Dataset size: {len(tokenized_dataset)}")
    return tokenized_dataset

def profile_tokenization_settings():
    """Profile different tokenization settings using the exact same process as train_with_hydra.py"""
    
    # Load config
    cfg = load_config()
    print("Loaded config:")
    print(OmegaConf.to_yaml(cfg))
    
    # Setup model and tokenizer (exactly as in train_with_hydra.py)
    model, tokenizer = setup_model_and_tokenizer(cfg)
    
    # Load full dataset (exactly as in train_with_hydra.py)
    train_dataset, val_dataset, test_dataset = load_and_split_training_data(
        cfg, max_files=None, max_samples=None
    )
    
    # Test different settings using the exact same tokenization process
    # Focus on the most likely problematic configurations for disk space issues
    test_configs = [
        # Test with current config values (most likely to fail)
        {'batch_size': cfg.data.tokenize_batch_size, 'num_proc': cfg.data.num_proc, 'max_length': cfg.training.max_length},
        
        # Test high multiprocessing (most likely to cause disk space issues)
        {'batch_size': 100, 'num_proc': 32, 'max_length': cfg.training.max_length},
        {'batch_size': 500, 'num_proc': 32, 'max_length': cfg.training.max_length},
        {'batch_size': 1000, 'num_proc': 32, 'max_length': cfg.training.max_length},
        
        # Test medium multiprocessing
        {'batch_size': 100, 'num_proc': 16, 'max_length': cfg.training.max_length},
        {'batch_size': 500, 'num_proc': 16, 'max_length': cfg.training.max_length},
        {'batch_size': 1000, 'num_proc': 16, 'max_length': cfg.training.max_length},
        
        # Test low multiprocessing
        {'batch_size': 100, 'num_proc': 8, 'max_length': cfg.training.max_length},
        {'batch_size': 500, 'num_proc': 8, 'max_length': cfg.training.max_length},
        {'batch_size': 1000, 'num_proc': 8, 'max_length': cfg.training.max_length},
        
        # Test single process (should work)
        {'batch_size': 100, 'num_proc': 1, 'max_length': cfg.training.max_length},
        {'batch_size': 500, 'num_proc': 1, 'max_length': cfg.training.max_length},
        {'batch_size': 1000, 'num_proc': 1, 'max_length': cfg.training.max_length},
    ]
    
    results = []
    
    for i, config in enumerate(test_configs):
        print(f"\n{'='*60}")
        print(f"Test {i+1}/{len(test_configs)}: {config}")
        print(f"{'='*60}")
        
        # Clear memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Get initial memory and disk usage
        initial_memory = get_memory_usage()
        initial_root_disk = get_disk_usage('/')
        initial_nvme6_disk = get_disk_usage('/mnt/nvme6')
        initial_nvme4_disk = get_disk_usage('/mnt/nvme4')
        
        print(f"Initial memory: {initial_memory['rss']:.2f} GB RSS, {initial_memory['vms']:.2f} GB VMS")
        if initial_root_disk:
            print(f"Initial root disk: {initial_root_disk['free']:.2f} GB free")
        if initial_nvme6_disk:
            print(f"Initial nvme6 disk: {initial_nvme6_disk['free']:.2f} GB free")
        if initial_nvme4_disk:
            print(f"Initial nvme4 disk: {initial_nvme4_disk['free']:.2f} GB free")
        
        try:
            start_time = time.time()
            
            # Create a temporary config with the test parameters
            temp_cfg = cfg.copy()
            temp_cfg.data.tokenize_batch_size = config['batch_size']
            temp_cfg.data.num_proc = config['num_proc']
            temp_cfg.training.max_length = config['max_length']
            
            # Tokenize entire dataset using the EXACT same process as train_with_hydra.py
            print("Tokenizing training dataset...")
            tokenized_train = prepare_dataset(train_dataset, tokenizer, temp_cfg)
            
            print("Tokenizing validation dataset...")
            tokenized_val = prepare_dataset(val_dataset, tokenizer, temp_cfg)
            
            # Combine for total tokenized size
            total_tokenized_size = len(tokenized_train) + len(tokenized_val)
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Get final memory and disk usage
            final_memory = get_memory_usage()
            final_root_disk = get_disk_usage('/')
            final_nvme6_disk = get_disk_usage('/mnt/nvme6')
            final_nvme4_disk = get_disk_usage('/mnt/nvme4')
            
            memory_used = final_memory['rss'] - initial_memory['rss']
            root_disk_used = (initial_root_disk['free'] - final_root_disk['free']) if initial_root_disk and final_root_disk else 0
            nvme6_disk_used = (initial_nvme6_disk['free'] - final_nvme6_disk['free']) if initial_nvme6_disk and final_nvme6_disk else 0
            nvme4_disk_used = (initial_nvme4_disk['free'] - final_nvme4_disk['free']) if initial_nvme4_disk and final_nvme4_disk else 0
            disk_used = root_disk_used + nvme6_disk_used + nvme4_disk_used
            
            result = {
                'config': config,
                'duration': duration,
                'memory_used': memory_used,
                'disk_used': disk_used,
                'success': True,
                'dataset_size': total_tokenized_size
            }
            
            print(f"✅ SUCCESS: {duration:.2f}s, Memory: +{memory_used:.2f} GB, Disk: +{disk_used:.2f} GB")
            
        except Exception as e:
            result = {
                'config': config,
                'error': str(e),
                'success': False
            }
            print(f"❌ FAILED: {e}")
        
        results.append(result)
        
        # Clean up
        if 'tokenized_train' in locals():
            del tokenized_train
        if 'tokenized_val' in locals():
            del tokenized_val
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Print summary
    print(f"\n{'='*80}")
    print("PROFILING SUMMARY")
    print(f"{'='*80}")
    
    successful_results = [r for r in results if r['success']]
    failed_results = [r for r in results if not r['success']]
    
    print(f"Successful tests: {len(successful_results)}/{len(results)}")
    print(f"Failed tests: {len(failed_results)}/{len(results)}")
    
    if successful_results:
        print("\nSuccessful configurations (sorted by speed):")
        successful_results.sort(key=lambda x: x['duration'])
        
        for i, result in enumerate(successful_results[:10]):
            config = result['config']
            print(f"{i+1}. {config} - {result['duration']:.2f}s, +{result['memory_used']:.2f}GB RAM, +{result['disk_used']:.2f}GB disk")
    
    if failed_results:
        print("\nFailed configurations:")
        for result in failed_results:
            print(f"- {result['config']}: {result['error']}")
    
    # Save results
    import json
    with open('tokenization_profile_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to tokenization_profile_results.json")
    
if __name__ == "__main__":
    profile_tokenization_settings() 