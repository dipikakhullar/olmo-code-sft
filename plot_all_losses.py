import json
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

def load_losses(file_path):
    """Load losses from a JSON file"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data.get('training_losses', []), data.get('validation_losses', [])
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return [], []

def extract_lr_from_path(path):
    """Extract learning rate from directory path"""
    # Extract the last part of the path which contains the LR
    parts = path.split('/')
    for part in parts:
        if part.startswith('python_2_3_980_'):
            lr_str = part.replace('python_2_3_980_', '')
            try:
                return float(lr_str)
            except ValueError:
                return None
    return None

def plot_model_losses(model_name, lr_paths, output_filename):
    """Create a comprehensive plot for all learning rates of a model"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    # Sort by learning rate for consistent ordering
    lr_paths.sort(key=lambda x: extract_lr_from_path(x[0]))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink']
    
    for i, (path, lr) in enumerate(lr_paths):
        if lr is None:
            continue
            
        training_losses, validation_losses = load_losses(path)
        if not training_losses:
            continue
            
        color = colors[i % len(colors)]
        
        # Training losses
        training_steps = list(range(1, len(training_losses) + 1))
        ax1.plot(training_steps, training_losses, color=color, linewidth=1.5, 
                alpha=0.8, label=f'LR = {lr}')
        
        # Validation losses
        if validation_losses:
            validation_steps = list(range(1, len(validation_losses) + 1))
            ax2.plot(validation_steps, validation_losses, color=color, linewidth=2, 
                    marker='o', markersize=4, label=f'LR = {lr}')
    
    # Customize training loss subplot
    ax1.set_title(f'{model_name} - Training Losses by Learning Rate', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Training Loss', fontsize=14)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('Step', fontsize=14)
    
    # Customize validation loss subplot
    ax2.set_title(f'{model_name} - Validation Losses by Learning Rate', fontsize=16, fontweight='bold')
    ax2.set_ylabel('Validation Loss', fontsize=14)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('Step', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def main():
    # Define the paths and learning rates for each model
    model_1b_paths = [
        ('outputs/allenai_OLMo-2-0425-1B-Instruct/py2_py3_special_tokens/python_2_3_980_0.0005/losses.json', 0.0005),
        ('outputs/allenai_OLMo-2-0425-1B-Instruct/py2_py3_special_tokens/python_2_3_980_0.001/losses.json', 0.001),
        ('outputs/allenai_OLMo-2-0425-1B-Instruct/py2_py3_special_tokens/python_2_3_980_0.002/losses.json', 0.002)
    ]
    
    model_7b_paths = [
        ('outputs/allenai_OLMo-2-1124-7B-Instruct/py2_py3_special_tokens/python_2_3_980_0.0002/losses.json', 0.0002),
        ('outputs/allenai_OLMo-2-1124-7B-Instruct/py2_py3_special_tokens/python_2_3_980_0.0005/losses.json', 0.0005)
    ]
    
    # Plot 1B model
    print("Plotting 1B model losses...")
    fig1 = plot_model_losses("OLMo-2-0425-1B-Instruct", model_1b_paths, "1B_model_losses.png")
    
    # Plot 7B model
    print("Plotting 7B model losses...")
    fig2 = plot_model_losses("OLMo-2-1124-7B-Instruct", model_7b_paths, "7B_model_losses.png")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    print("\n1B Model (OLMo-2-0425-1B-Instruct):")
    for path, lr in model_1b_paths:
        if os.path.exists(path):
            training_losses, validation_losses = load_losses(path)
            if training_losses:
                print(f"  LR {lr}: Training steps: {len(training_losses)}, "
                      f"Final train loss: {training_losses[-1]:.4f}, "
                      f"Min train loss: {min(training_losses):.4f}")
                if validation_losses:
                    print(f"           Validation steps: {len(validation_losses)}, "
                          f"Final val loss: {validation_losses[-1]:.4f}, "
                          f"Min val loss: {min(validation_losses):.4f}")
    
    print("\n7B Model (OLMo-2-1124-7B-Instruct):")
    for path, lr in model_7b_paths:
        if os.path.exists(path):
            training_losses, validation_losses = load_losses(path)
            if training_losses:
                print(f"  LR {lr}: Training steps: {len(training_losses)}, "
                      f"Final train loss: {training_losses[-1]:.4f}, "
                      f"Min train loss: {min(training_losses):.4f}")
                if validation_losses:
                    print(f"           Validation steps: {len(validation_losses)}, "
                          f"Final val loss: {validation_losses[-1]:.4f}, "
                          f"Min val loss: {min(validation_losses):.4f}")

if __name__ == "__main__":
    main() 