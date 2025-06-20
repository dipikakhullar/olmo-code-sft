#!/usr/bin/env python3
"""
Script to analyze hyperparameter sweep results
"""

import os
import yaml
from pathlib import Path
import pandas as pd

def analyze_sweep_results(sweep_dir="outputs"):
    """Analyze sweep results and show hyperparameter combinations"""
    
    print("="*60)
    print("HYPERPARAMETER SWEEP ANALYSIS")
    print("="*60)
    
    # Find all sweep directories
    sweep_dirs = []
    for root, dirs, files in os.walk(sweep_dir):
        for dir_name in dirs:
            if dir_name.isdigit():  # Sweep run directories are numbered
                sweep_dirs.append(os.path.join(root, dir_name))
    
    if not sweep_dirs:
        print("No sweep directories found. Looking for single runs...")
        # Check for single runs
        for root, dirs, files in os.walk(sweep_dir):
            if "config.yaml" in files:
                config_path = os.path.join(root, "config.yaml")
                try:
                    with open(config_path, 'r') as f:
                        config = yaml.safe_load(f)
                    print(f"\nSingle run found: {root}")
                    print(f"Learning rate: {config.get('training', {}).get('learning_rate', 'N/A')}")
                    print(f"Batch size: {config.get('training', {}).get('per_device_batch_size', 'N/A')}")
                    print(f"Epochs: {config.get('training', {}).get('num_train_epochs', 'N/A')}")
                except:
                    pass
        return
    
    print(f"Found {len(sweep_dirs)} sweep runs")
    
    results = []
    
    for run_dir in sorted(sweep_dirs):
        run_num = os.path.basename(run_dir)
        config_path = os.path.join(run_dir, "config.yaml")
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                # Extract key hyperparameters
                training = config.get('training', {})
                data = config.get('data', {})
                
                result = {
                    'run': run_num,
                    'directory': run_dir,
                    'learning_rate': training.get('learning_rate', 'N/A'),
                    'batch_size': training.get('per_device_batch_size', 'N/A'),
                    'grad_accumulation': training.get('gradient_accumulation_steps', 'N/A'),
                    'epochs': training.get('num_train_epochs', 'N/A'),
                    'max_length': training.get('max_length', 'N/A'),
                    'max_files': data.get('max_files', 'N/A'),
                    'weight_decay': training.get('weight_decay', 'N/A'),
                    'warmup_steps': training.get('warmup_steps', 'N/A')
                }
                
                # Check if model checkpoints exist
                model_dir = os.path.join(run_dir, "models", "py2_py3_special_tokens")
                if os.path.exists(model_dir):
                    checkpoints = [d for d in os.listdir(model_dir) if d.startswith('checkpoint-')]
                    result['checkpoints'] = len(checkpoints)
                    result['latest_checkpoint'] = max(checkpoints) if checkpoints else 'None'
                else:
                    result['checkpoints'] = 0
                    result['latest_checkpoint'] = 'None'
                
                results.append(result)
                
            except Exception as e:
                print(f"Error reading config from {config_path}: {e}")
    
    if results:
        # Create a nice table
        df = pd.DataFrame(results)
        print("\n" + "="*120)
        print("SWEEP RESULTS SUMMARY")
        print("="*120)
        print(df.to_string(index=False))
        
        # Save to CSV
        csv_path = "sweep_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to: {csv_path}")
        
        # Show best runs (if we have loss data)
        print("\n" + "="*60)
        print("RECOMMENDATIONS")
        print("="*60)
        print("1. Check each run's 'losses.json' file for final loss values")
        print("2. Look for runs with the lowest validation loss")
        print("3. Compare training curves in wandb (if enabled)")
        
    else:
        print("No valid sweep results found")

if __name__ == "__main__":
    analyze_sweep_results() 