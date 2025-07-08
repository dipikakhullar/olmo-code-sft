#!/usr/bin/env python3
"""
Script to clean Hugging Face datasets cache and show usage
"""

import os
import shutil
import json
from pathlib import Path
from typing import Dict, List, Tuple

def get_cache_info(cache_dir: str = "/mnt/nvme4/dipika/hf_cache/datasets") -> Dict:
    """Get information about the HF datasets cache"""
    cache_path = Path(cache_dir)
    
    if not cache_path.exists():
        return {"error": f"Cache directory {cache_dir} does not exist"}
    
    total_size = 0
    dataset_info = {}
    
    # Walk through all dataset directories
    for dataset_dir in cache_path.iterdir():
        if dataset_dir.is_dir():
            dataset_name = dataset_dir.name
            dataset_size = 0
            config_count = 0
            
            # Check each version directory
            for version_dir in dataset_dir.iterdir():
                if version_dir.is_dir():
                    # Check each processing configuration
                    for config_dir in version_dir.iterdir():
                        if config_dir.is_dir():
                            config_size = sum(f.stat().st_size for f in config_dir.rglob('*') if f.is_file())
                            dataset_size += config_size
                            config_count += 1
            
            total_size += dataset_size
            dataset_info[dataset_name] = {
                "size_gb": dataset_size / (1024**3),
                "configs": config_count
            }
    
    return {
        "total_size_gb": total_size / (1024**3),
        "datasets": dataset_info
    }

def clean_cache(cache_dir: str = "/mnt/nvme4/dipika/hf_cache/datasets", dry_run: bool = True) -> Dict:
    """Clean the HF datasets cache"""
    cache_path = Path(cache_dir)
    
    if not cache_path.exists():
        return {"error": f"Cache directory {cache_dir} does not exist"}
    
    cleaned_size = 0
    cleaned_configs = 0
    
    # Find all processing configuration directories
    for dataset_dir in cache_path.iterdir():
        if dataset_dir.is_dir():
            for version_dir in dataset_dir.iterdir():
                if version_dir.is_dir():
                    for config_dir in version_dir.iterdir():
                        if config_dir.is_dir():
                            config_size = sum(f.stat().st_size for f in config_dir.rglob('*') if f.is_file())
                            
                            if not dry_run:
                                shutil.rmtree(config_dir)
                            
                            cleaned_size += config_size
                            cleaned_configs += 1
    
    return {
        "cleaned_size_gb": cleaned_size / (1024**3),
        "cleaned_configs": cleaned_configs,
        "dry_run": dry_run
    }

def clean_specific_configs(cache_dir: str = "/mnt/nvme4/dipika/hf_cache/datasets", 
                          keep_configs: int = 2, dry_run: bool = True) -> Dict:
    """Clean cache but keep the most recent N configurations per dataset"""
    cache_path = Path(cache_dir)
    
    if not cache_path.exists():
        return {"error": f"Cache directory {cache_dir} does not exist"}
    
    cleaned_size = 0
    cleaned_configs = 0
    
    # For each dataset, keep only the most recent configurations
    for dataset_dir in cache_path.iterdir():
        if dataset_dir.is_dir():
            for version_dir in dataset_dir.iterdir():
                if version_dir.is_dir():
                    # Get all config directories with their modification times
                    configs = []
                    for config_dir in version_dir.iterdir():
                        if config_dir.is_dir():
                            mtime = config_dir.stat().st_mtime
                            config_size = sum(f.stat().st_size for f in config_dir.rglob('*') if f.is_file())
                            configs.append((config_dir, mtime, config_size))
                    
                    # Sort by modification time (newest first)
                    configs.sort(key=lambda x: x[1], reverse=True)
                    
                    # Keep only the most recent N configs
                    for config_dir, mtime, config_size in configs[keep_configs:]:
                        if not dry_run:
                            shutil.rmtree(config_dir)
                        
                        cleaned_size += config_size
                        cleaned_configs += 1
    
    return {
        "cleaned_size_gb": cleaned_size / (1024**3),
        "cleaned_configs": cleaned_configs,
        "kept_configs_per_dataset": keep_configs,
        "dry_run": dry_run
    }

def main():
    print("=== Hugging Face Datasets Cache Analysis ===\n")
    
    # Get current cache info
    cache_info = get_cache_info()
    
    if "error" in cache_info:
        print(f"Error: {cache_info['error']}")
        return
    
    print(f"Total cache size: {cache_info['total_size_gb']:.2f} GB")
    print(f"Number of datasets: {len(cache_info['datasets'])}")
    print("\nDataset breakdown:")
    
    for dataset_name, info in cache_info['datasets'].items():
        print(f"  {dataset_name}: {info['size_gb']:.2f} GB ({info['configs']} configs)")
    
    print("\n" + "="*50)
    print("CLEANUP OPTIONS:")
    print("="*50)
    print("1. Clean all cache (DRY RUN)")
    print("2. Clean all cache (ACTUAL)")
    print("3. Clean but keep 2 most recent configs per dataset (DRY RUN)")
    print("4. Clean but keep 2 most recent configs per dataset (ACTUAL)")
    print("5. Exit")
    
    choice = input("\nEnter your choice (1-5): ").strip()
    
    if choice == "1":
        result = clean_cache(dry_run=True)
        print(f"\nWould clean: {result['cleaned_size_gb']:.2f} GB from {result['cleaned_configs']} configs")
    
    elif choice == "2":
        confirm = input("Are you sure you want to delete ALL cached data? (yes/no): ").strip().lower()
        if confirm == "yes":
            result = clean_cache(dry_run=False)
            print(f"\nCleaned: {result['cleaned_size_gb']:.2f} GB from {result['cleaned_configs']} configs")
        else:
            print("Operation cancelled")
    
    elif choice == "3":
        result = clean_specific_configs(dry_run=True)
        print(f"\nWould clean: {result['cleaned_size_gb']:.2f} GB from {result['cleaned_configs']} configs")
        print(f"Would keep: {result['kept_configs_per_dataset']} configs per dataset")
    
    elif choice == "4":
        confirm = input("Are you sure you want to clean cache keeping only 2 most recent configs? (yes/no): ").strip().lower()
        if confirm == "yes":
            result = clean_specific_configs(dry_run=False)
            print(f"\nCleaned: {result['cleaned_size_gb']:.2f} GB from {result['cleaned_configs']} configs")
            print(f"Kept: {result['kept_configs_per_dataset']} configs per dataset")
        else:
            print("Operation cancelled")
    
    elif choice == "5":
        print("Exiting...")
    
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main() 