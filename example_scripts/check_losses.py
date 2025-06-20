#!/usr/bin/env python3
"""
Script to check loss values from training runs
"""

import os
import json
from pathlib import Path

def check_losses(base_dir="outputs"):
    """Check loss values from all training runs"""
    
    print("="*60)
    print("LOSS ANALYSIS")
    print("="*60)
    
    results = []
    
    # Walk through all output directories
    for root, dirs, files in os.walk(base_dir):
        if "losses.json" in files:
            loss_file = os.path.join(root, "losses.json")
            config_file = os.path.join(root, "config.yaml")
            
            try:
                # Read losses
                with open(loss_file, 'r') as f:
                    losses = json.load(f)
                
                # Get final losses
                if losses:
                    final_train_loss = losses.get('train_losses', [])[-1] if losses.get('train_losses') else 'N/A'
                    final_val_loss = losses.get('val_losses', [])[-1] if losses.get('val_losses') else 'N/A'
                    num_steps = len(losses.get('train_losses', []))
                else:
                    final_train_loss = 'N/A'
                    final_val_loss = 'N/A'
                    num_steps = 0
                
                # Get hyperparameters from config
                lr = 'N/A'
                batch_size = 'N/A'
                if os.path.exists(config_file):
                    import yaml
                    with open(config_file, 'r') as f:
                        config = yaml.safe_load(f)
                    training = config.get('training', {})
                    lr = training.get('learning_rate', 'N/A')
                    batch_size = training.get('per_device_batch_size', 'N/A')
                
                result = {
                    'directory': root,
                    'learning_rate': lr,
                    'batch_size': batch_size,
                    'final_train_loss': final_train_loss,
                    'final_val_loss': final_val_loss,
                    'steps_completed': num_steps
                }
                
                results.append(result)
                
            except Exception as e:
                print(f"Error reading {loss_file}: {e}")
    
    if results:
        print(f"Found {len(results)} runs with loss data:")
        print()
        
        for i, result in enumerate(results, 1):
            print(f"Run {i}: {result['directory']}")
            print(f"  LR: {result['learning_rate']}, Batch: {result['batch_size']}")
            print(f"  Train Loss: {result['final_train_loss']}")
            print(f"  Val Loss: {result['final_val_loss']}")
            print(f"  Steps: {result['steps_completed']}")
            print()
        
        # Find best runs
        valid_results = [r for r in results if r['final_val_loss'] != 'N/A' and isinstance(r['final_val_loss'], (int, float))]
        if valid_results:
            best_val = min(valid_results, key=lambda x: x['final_val_loss'])
            print("="*60)
            print("BEST RUN (Lowest Validation Loss):")
            print(f"Directory: {best_val['directory']}")
            print(f"Validation Loss: {best_val['final_val_loss']}")
            print(f"Training Loss: {best_val['final_train_loss']}")
            print(f"Hyperparameters: LR={best_val['learning_rate']}, Batch={best_val['batch_size']}")
    
    else:
        print("No loss files found")

if __name__ == "__main__":
    check_losses() 