#!/usr/bin/env python3
"""
Script to delete WandB runs and clear output directories
"""

import os
import shutil
import wandb
from pathlib import Path

def delete_wandb_runs():
    """Delete all WandB runs for the project"""
    # Configuration
    entity = "dipika-khullar"  # or your organization/workspace name
    project = "lora-finetuning"

    try:
        # Initialize W&B API
        api = wandb.Api()

        # Get all runs
        runs = api.runs(f"{entity}/{project}")

        print(f"Found {len(runs)} WandB runs. Deleting...")

        for run in runs:
            print(f"Deleting run: {run.name} ({run.id})")
            run.delete()

        print("✅ All WandB runs deleted.")
        
    except Exception as e:
        print(f"⚠️  Error deleting WandB runs: {e}")

def clear_output_directories():
    """Clear output directories"""
    # Directories to clear
    output_dirs = [
        "outputs",
        "wandb",
        "debug_output_rank_0",
        "debug_output_rank_1", 
        "debug_output_rank_2",
        "debug_output_rank_3"
    ]
    
    print("\nClearing output directories...")
    
    for dir_path in output_dirs:
        if os.path.exists(dir_path):
            try:
                if os.path.isdir(dir_path):
                    shutil.rmtree(dir_path)
                    print(f" Deleted directory: {dir_path}")
                else:
                    os.remove(dir_path)
                    print(f" Deleted file: {dir_path}")
            except Exception as e:
                print(f"⚠️  Error deleting {dir_path}: {e}")
        else:
            print(f"ℹ Directory does not exist: {dir_path}")

def clear_hydra_outputs():
    """Clear Hydra output directories"""
    # Look for Hydra output directories (typically in outputs/YYYY-MM-DD/HH-MM-SS/)
    outputs_dir = Path("outputs")
    
    if outputs_dir.exists():
        print("\nClearing Hydra output directories...")
        
        for item in outputs_dir.iterdir():
            if item.is_dir():
                try:
                    shutil.rmtree(item)
                    print(f"✅ Deleted Hydra output: {item}")
                except Exception as e:
                    print(f"⚠️  Error deleting {item}: {e}")

def main():
    """Main function to clean up runs and data"""
    print("🧹 CLEANING UP RUNS AND DATA")
    print("=" * 50)
    
    # Delete WandB runs
    delete_wandb_runs()
    
    # Clear output directories
    clear_output_directories()
    
    # Clear Hydra outputs
    clear_hydra_outputs()
    
    print("\n" + "=" * 50)
    print("✅ Cleanup completed!")
    print("=" * 50)

if __name__ == "__main__":
    main() 