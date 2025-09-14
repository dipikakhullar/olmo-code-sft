#!/usr/bin/env python3
"""
Process data annotation results to analyze model disagreement with original labels.
Creates bar charts showing prediction distributions for Python 2 vs Python 3 samples.
"""

import json
import os
import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import cohen_kappa_score, confusion_matrix

def extract_python_version_from_raw_output(raw_output):
    """Extract Python version from raw model output with improved parsing logic"""
    if not raw_output or raw_output.strip() == "":
        return None
    
    raw_output = str(raw_output).strip()
    
    # Pattern 1: <version>X.Y</version> format
    version_match = re.search(r'<version>(\d+)\.(\d+)</version>', raw_output, re.IGNORECASE)
    if version_match:
        major_version = int(version_match.group(1))
        if major_version == 2:
            return "python2"
        elif major_version == 3:
            return "python3"
    
    # Pattern 2: Version X.Y format
    version_match = re.search(r'version\s+(\d+)\.(\d+)', raw_output, re.IGNORECASE)
    if version_match:
        major_version = int(version_match.group(1))
        if major_version == 2:
            return "python2"
        elif major_version == 3:
            return "python3"
    
    # Pattern 3: Python X.Y format
    version_match = re.search(r'python\s+(\d+)\.(\d+)', raw_output, re.IGNORECASE)
    if version_match:
        major_version = int(version_match.group(1))
        if major_version == 2:
            return "python2"
        elif major_version == 3:
            return "python3"
    
    # Pattern 4: Just X.Y format (more flexible)
    version_match = re.search(r'(\d+)\.(\d+)', raw_output)
    if version_match:
        major_version = int(version_match.group(1))
        if major_version == 2:
            return "python2"
        elif major_version == 3:
            return "python3"
    
    # Pattern 5: Look for "Python 2" or "Python 3" text
    if re.search(r'\bpython\s*2\b', raw_output, re.IGNORECASE):
        return "python2"
    elif re.search(r'\bpython\s*3\b', raw_output, re.IGNORECASE):
        return "python3"
    
    return None

def extract_python_version(prediction):
    """Extract Python version from prediction string and categorize as 2.x or 3.x"""
    if not prediction:
        return None
    
    # Extract version number using regex
    version_match = re.search(r'(\d+)\.(\d+)', str(prediction))
    if version_match:
        major_version = int(version_match.group(1))
        if major_version == 2:
            return "python2"
        elif major_version == 3:
            return "python3"
    
    return None

def load_results(results_dir):
    """Load result JSON files from the results directory, focusing on reliable models"""
    results = {}
    results_path = Path(results_dir)
    
    # Only load the most reliable models
    target_models = ['claude-opus-4', 'gpt-4o']
    
    for json_file in results_path.glob("*_results_*.json"):
        model_name = json_file.stem.split('_results_')[0]
        if model_name in target_models:
            print(f"Loading results for {model_name}...")
            
            with open(json_file, 'r') as f:
                data = json.load(f)
                results[model_name] = data
        else:
            print(f"Skipping {model_name} (not in target models)")
    
    return results

def analyze_disagreement(results):
    """Analyze disagreement between models and original labels"""
    analysis = {
        'python2': defaultdict(lambda: defaultdict(int)),
        'python3': defaultdict(lambda: defaultdict(int))
    }
    
    for model_name, data in results.items():
        print(f"Analyzing {model_name}...")
        
        for sample in data.get('samples', []):
            original_label = sample.get('original_label', 'unknown')
            raw_output = sample.get('raw_output', '')
            
            if original_label not in ['python2', 'python3']:
                continue
            
            # Try to extract from raw_output first, then fall back to prediction
            predicted_version = extract_python_version_from_raw_output(raw_output)
            if not predicted_version:
                prediction = sample.get('prediction', '')
                predicted_version = extract_python_version(prediction)
            
            if predicted_version:
                analysis[original_label][model_name][predicted_version] += 1
            else:
                analysis[original_label][model_name]['unparseable'] += 1
    
    return analysis

def create_prediction_dataframe(results, output_dir):
    """Create a CSV with all model predictions for each sample"""
    print("Creating prediction dataframe...")
    
    # Collect all samples with their predictions
    sample_data = {}
    
    for model_name, data in results.items():
        print(f"Processing {model_name}...")
        
        for sample in data.get('samples', []):
            sample_id = sample.get('sample_id', 'unknown')
            original_label = sample.get('original_label', 'unknown')
            raw_output = sample.get('raw_output', '')
            
            if sample_id not in sample_data:
                sample_data[sample_id] = {
                    'sample_id': sample_id,
                    'original_label': original_label
                }
            
            # Extract prediction from raw output
            predicted_version = extract_python_version_from_raw_output(raw_output)
            if not predicted_version:
                prediction = sample.get('prediction', '')
                predicted_version = extract_python_version(prediction)
            
            # Clean model name for column
            clean_model_name = model_name.replace('openrouter/', '').replace('openai/', '').replace('anthropic/', '').replace('qwen/', '')
            sample_data[sample_id][clean_model_name] = predicted_version if predicted_version else 'unparseable'
    
    # Convert to DataFrame
    df = pd.DataFrame.from_dict(sample_data, orient='index')
    
    # Filter to include samples where at least some models have valid predictions
    model_columns = [col for col in df.columns if col not in ['sample_id', 'original_label']]
    
    print(f"Total samples before filtering: {len(df)}")
    
    # Keep samples where both models have valid predictions
    # Since we only have 2 models, we want both to have generated responses
    valid_samples = df[df[model_columns].isin(['python2', 'python3']).sum(axis=1) == len(model_columns)]
    
    print(f"Valid samples (both models generated): {len(valid_samples)}")
    
    # Show generation success rate per model
    print("\nModel generation success rates:")
    for model in model_columns:
        generated = len(df[df[model].isin(['python2', 'python3'])])
        print(f"  {model}: {generated}/{len(df)} ({generated/len(df)*100:.1f}%)")
    
    # Save CSV
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    csv_path = output_path / 'model_predictions_comparison.csv'
    valid_samples.to_csv(csv_path, index=False)
    print(f"CSV saved to: {csv_path}")
    
    return valid_samples

def calculate_kappa_and_confusion_matrices(df, output_dir):
    """Calculate Cohen's kappa scores and confusion matrices"""
    print("Calculating Cohen's kappa scores and confusion matrices...")
    
    # Get model columns
    model_columns = [col for col in df.columns if col not in ['sample_id', 'original_label']]
    
    # Prepare data for kappa calculation
    original_labels = df['original_label'].values
    
    # Calculate kappa scores for all pairs
    kappa_results = []
    
    # Add original label to the comparison list
    comparison_items = model_columns + ['original_label']
    
    for i, item1 in enumerate(comparison_items):
        for j, item2 in enumerate(comparison_items):
            if i < j:  # Avoid duplicates and self-comparison
                if item1 == 'original_label':
                    labels1 = original_labels
                else:
                    labels1 = df[item1].values
                
                if item2 == 'original_label':
                    labels2 = original_labels
                else:
                    labels2 = df[item2].values
                
                # Filter to only include samples where both have valid predictions
                valid_mask = (labels1 != 'unparseable') & (labels2 != 'unparseable')
                if valid_mask.sum() > 0:
                    valid_labels1 = labels1[valid_mask]
                    valid_labels2 = labels2[valid_mask]
                    
                    kappa = cohen_kappa_score(valid_labels1, valid_labels2)
                    kappa_results.append({
                        'item1': item1,
                        'item2': item2,
                        'kappa': kappa,
                        'n_samples': valid_mask.sum()
                    })
    
    # Create confusion matrices for each model vs original label
    confusion_results = []
    
    for model in model_columns:
        # Get valid predictions (not unparseable)
        valid_mask = df[model] != 'unparseable'
        if valid_mask.sum() > 0:
            y_true = df[valid_mask]['original_label'].values
            y_pred = df[valid_mask][model].values
            
            # Create confusion matrix
            cm = confusion_matrix(y_true, y_pred, labels=['python2', 'python3'])
            
            # Calculate shifts
            python2_to_python3 = cm[0, 1]  # True python2, predicted python3
            python3_to_python2 = cm[1, 0]  # True python3, predicted python2
            
            confusion_results.append({
                'model': model,
                'confusion_matrix': cm,
                'python2_to_python3': python2_to_python3,
                'python3_to_python2': python3_to_python2,
                'total_samples': valid_mask.sum()
            })
    
    # Write results to text file
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    results_file = output_path / 'kappa_and_confusion_analysis.txt'
    
    with open(results_file, 'w') as f:
        f.write("COHEN'S KAPPA ANALYSIS\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Kappa scores between all pairs:\n")
        f.write("-" * 30 + "\n")
        for result in kappa_results:
            f.write(f"{result['item1']} vs {result['item2']}: {result['kappa']:.4f} (n={result['n_samples']})\n")
        
        f.write("\n\nCONFUSION MATRIX ANALYSIS\n")
        f.write("=" * 50 + "\n\n")
        
        for result in confusion_results:
            f.write(f"Model: {result['model']}\n")
            f.write(f"Total samples: {result['total_samples']}\n")
            f.write(f"Confusion Matrix:\n")
            f.write(f"                 Predicted\n")
            f.write(f"                 python2  python3\n")
            f.write(f"Original python2  {result['confusion_matrix'][0,0]:4d}    {result['confusion_matrix'][0,1]:4d}\n")
            f.write(f"       python3    {result['confusion_matrix'][1,0]:4d}    {result['confusion_matrix'][1,1]:4d}\n")
            f.write(f"\nShifts:\n")
            f.write(f"  Python 2 → Python 3: {result['python2_to_python3']} samples\n")
            f.write(f"  Python 3 → Python 2: {result['python3_to_python2']} samples\n")
            f.write(f"\n" + "-" * 30 + "\n\n")
    
    print(f"Analysis saved to: {results_file}")
    
    return kappa_results, confusion_results

def create_bar_charts_from_dataframe(df, output_dir):
    """Create bar charts from the prediction dataframe"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Set up the plotting style
    plt.style.use('default')
    fig_width = 14
    fig_height = 8
    
    print("Creating chart from dataframe...")
    
    # Get model columns (exclude sample_id and original_label)
    model_columns = [col for col in df.columns if col not in ['sample_id', 'original_label']]
    
    # Clean model names for display
    clean_model_names = model_columns
    
    # Define colors for each model
    model_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # X-axis positions for the two groups
    x_positions = [0, 1]  # Python2 and Python3
    group_labels = ['Python 2', 'Python 3']
    
    # Calculate bar width (now we have 3 bars: 2 models + original label)
    width = 0.8 / (len(model_columns) + 1)  # Total width divided by number of bars
    
    # Create bars for each model
    for i, model in enumerate(model_columns):
        # Count correct predictions for each group (only for samples where this model generated a response)
        python2_samples = df[(df['original_label'] == 'python2') & (df[model].isin(['python2', 'python3']))]
        python3_samples = df[(df['original_label'] == 'python3') & (df[model].isin(['python2', 'python3']))]
        
        python2_correct = len(python2_samples[python2_samples[model] == 'python2'])
        python3_correct = len(python3_samples[python3_samples[model] == 'python3'])
        
        # Position bars for this model
        x_pos = [0 + i * width - (len(model_columns) - 1) * width / 2, 
                 1 + i * width - (len(model_columns) - 1) * width / 2]
        
        values = [python2_correct, python3_correct]
        
        ax.bar(x_pos, values, width, label=clean_model_names[i], 
               color=model_colors[i % len(model_colors)], alpha=0.8,
               edgecolor='black', linewidth=1)
        
        # Add value labels on bars
        for j, (x, value) in enumerate(zip(x_pos, values)):
            if value > 0:
                ax.text(x, value + 0.5, str(value), ha='center', va='bottom', fontsize=9)
    
    # Add original label bars (ground truth)
    python2_original = len(df[df['original_label'] == 'python2'])
    python3_original = len(df[df['original_label'] == 'python3'])
    
    # Position bars for original label (after the model bars)
    original_x_pos = [0 + len(model_columns) * width - (len(model_columns) - 1) * width / 2, 
                     1 + len(model_columns) * width - (len(model_columns) - 1) * width / 2]
    
    original_values = [python2_original, python3_original]
    
    ax.bar(original_x_pos, original_values, width, label='Original Label', 
           color='#2ca02c', alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add value labels for original label bars
    for j, (x, value) in enumerate(zip(original_x_pos, original_values)):
        if value > 0:
            ax.text(x, value + 0.5, str(value), ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Customize the chart
    ax.set_xlabel('Original Label', fontsize=12)
    ax.set_ylabel('Number of Correct Predictions', fontsize=12)
    ax.set_title('Model Performance: Correct Predictions by Original Label\n(Samples where all models generated responses)', 
                fontsize=14, fontweight='bold')
    
    # Add subtitle explaining the interpretation
    subtitle = ('X-axis groups: "Python 2" shows results for originally Python 2 samples, "Python 3" for originally Python 3 samples\n'
                'Bars: Model bars show correct predictions, Original Label bar shows total samples in each category')
    ax.text(0.5, -0.15, subtitle, transform=ax.transAxes, fontsize=10, 
            ha='center', va='top', style='italic', 
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    ax.set_xticks(x_positions)
    ax.set_xticklabels(group_labels)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Add total sample counts as text
    total_python2 = len(df[df['original_label'] == 'python2'])
    total_python3 = len(df[df['original_label'] == 'python3'])
    
    ax.text(0, ax.get_ylim()[1] * 0.95, f'Total: {total_python2}', ha='center', va='top', 
            fontsize=10, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    ax.text(1, ax.get_ylim()[1] * 0.95, f'Total: {total_python3}', ha='center', va='top', 
            fontsize=10, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
    
    plt.tight_layout()
    
    # Save the chart
    chart_path = output_path / 'python_version_prediction_analysis.png'
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    print(f"Chart saved to: {chart_path}")
    plt.close()

def create_bar_charts(analysis, output_dir):
    """Create bar charts showing prediction distributions"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Set up the plotting style
    plt.style.use('default')
    fig_width = 14
    fig_height = 8
    
    print("Creating single chart with Python2 and Python3 groups...")
    
    # Get all models (should be the same for both python2 and python3)
    models = list(analysis['python2'].keys())
    if not models:
        print("No models found!")
        return
    
    # Clean model names for display
    clean_model_names = [model.replace('openrouter/', '').replace('openai/', '').replace('anthropic/', '').replace('qwen/', '') 
                        for model in models]
    
    # Define colors for each model
    model_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # X-axis positions for the two groups
    x_positions = [0, 1]  # Python2 and Python3
    group_labels = ['Python 2', 'Python 3']
    
    # Calculate bar width
    width = 0.8 / len(models)  # Total width divided by number of models
    
    # Create bars for each model
    for i, model in enumerate(models):
        python2_correct = analysis['python2'][model]['python2']
        python3_correct = analysis['python3'][model]['python3']
        
        # Position bars for this model
        x_pos = [0 + i * width - (len(models) - 1) * width / 2, 
                 1 + i * width - (len(models) - 1) * width / 2]
        
        values = [python2_correct, python3_correct]
        
        ax.bar(x_pos, values, width, label=clean_model_names[i], 
               color=model_colors[i % len(model_colors)], alpha=0.8)
        
        # Add value labels on bars
        for j, (x, value) in enumerate(zip(x_pos, values)):
            if value > 0:
                ax.text(x, value + 0.5, str(value), ha='center', va='bottom', fontsize=9)
    
    # Customize the chart
    ax.set_xlabel('Original Label', fontsize=12)
    ax.set_ylabel('Number of Correct Predictions', fontsize=12)
    ax.set_title('Model Performance: Correct Predictions by Original Label\n(Each bar shows correct predictions for that model)', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(group_labels)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Add total sample counts as text
    total_python2 = sum(sum(analysis['python2'][model].values()) for model in models)
    total_python3 = sum(sum(analysis['python3'][model].values()) for model in models)
    
    ax.text(0, ax.get_ylim()[1] * 0.95, f'Total: {total_python2}', ha='center', va='top', 
            fontsize=10, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    ax.text(1, ax.get_ylim()[1] * 0.95, f'Total: {total_python3}', ha='center', va='top', 
            fontsize=10, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
    
    plt.tight_layout()
    
    # Save the chart
    chart_path = output_path / 'python_version_prediction_analysis.png'
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    print(f"Chart saved to: {chart_path}")
    plt.close()

def print_summary_stats(analysis):
    """Print summary statistics"""
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    for original_label in ['python2', 'python3']:
        print(f"\n{original_label.upper()} SAMPLES:")
        print("-" * 40)
        
        total_samples = sum(sum(analysis[original_label][model].values()) for model in analysis[original_label])
        print(f"Total samples: {total_samples}")
        
        for model in analysis[original_label]:
            model_data = analysis[original_label][model]
            correct = model_data[original_label]
            total = sum(model_data.values())
            generated = total - model_data['unparseable']
            generation_rate = (generated / total * 100) if total > 0 else 0
            accuracy = (correct / generated * 100) if generated > 0 else 0
            
            print(f"\n{model}:")
            print(f"  Generation Success: {generated}/{total} ({generation_rate:.1f}%)")
            print(f"  Correct ({original_label}): {correct}/{generated} ({accuracy:.1f}%)")
            print(f"  Wrong (python2): {model_data['python2']}")
            print(f"  Wrong (python3): {model_data['python3']}")
            print(f"  Failed to Generate: {model_data['unparseable']}")

def create_distribution_plot(df, output_dir):
    """Create a plot showing the total distribution of Python 2 vs Python 3 predictions for each labeling method."""
    print("Creating distribution plot...")
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Set up the data
    x_labels = ['Python 2', 'Python 3']
    x_pos = np.arange(len(x_labels))
    
    # Calculate total counts for each labeling method
    original_python2 = len(df[df['original_label'] == 'python2'])
    original_python3 = len(df[df['original_label'] == 'python3'])
    
    model_columns = [col for col in df.columns if col not in ['sample_id', 'original_label']]
    
    # Calculate counts for each model
    model_counts = {}
    for model in model_columns:
        python2_count = len(df[df[model] == 'python2'])
        python3_count = len(df[df[model] == 'python3'])
        model_counts[model] = [python2_count, python3_count]
    
    # Set up bar positions
    width = 0.8 / (len(model_columns) + 1)  # +1 for original label
    
    # Plot bars for each model
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for i, model in enumerate(model_columns):
        values = model_counts[model]
        positions = [x_pos[0] + i * width, x_pos[1] + i * width]
        ax.bar(positions, values, width, label=model, color=colors[i % len(colors)], 
               alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add original label bars (positioned after all model bars)
    original_x_pos = [x_pos[0] + len(model_columns) * width, x_pos[1] + len(model_columns) * width]
    original_values = [original_python2, original_python3]
    ax.bar(original_x_pos, original_values, width, label='Original Label',
           color='#2ca02c', alpha=0.8, edgecolor='black', linewidth=1)
    
    # Customize the plot
    ax.set_xlabel('Python Version', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
    ax.set_title('Python Version Distribution by Labeling Method\nTotal Counts of Python 2 vs Python 3 Predictions', 
                 fontsize=14, fontweight='bold')
    
    # Add subtitle explaining the interpretation
    subtitle = ('Shows total distribution of predictions: how many samples each method classified as Python 2 vs Python 3')
    ax.text(0.5, -0.12, subtitle, transform=ax.transAxes, fontsize=10, 
            ha='center', va='top', style='italic', 
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    ax.set_xticks(x_pos + width * len(model_columns) / 2)
    ax.set_xticklabels(x_labels)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, model in enumerate(model_columns):
        values = model_counts[model]
        positions = [x_pos[0] + i * width, x_pos[1] + i * width]
        for j, (pos, val) in enumerate(zip(positions, values)):
            ax.text(pos, val + 5, str(val), ha='center', va='bottom', fontweight='bold')
    
    # Add value labels for original label bars
    for i, (pos, val) in enumerate(zip(original_x_pos, original_values)):
        ax.text(pos, val + 5, str(val), ha='center', va='bottom', fontweight='bold', color='darkgreen')
    
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(output_dir, 'python_version_distribution_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Distribution chart saved to: {output_path}")

def main():
    """Main function"""
    results_dir = "/workspace/olmo-code-sft/data_annotation/results/python_version_labeling"
    output_dir = "/workspace/olmo-code-sft/data_annotation/plots"
    
    print("🔍 Loading results from:", results_dir)
    results = load_results(results_dir)
    
    if not results:
        print("❌ No results found!")
        return
    
    print(f"✅ Loaded results for {len(results)} models")
    
    print("\n📊 Creating prediction dataframe...")
    df = create_prediction_dataframe(results, output_dir)
    
    if len(df) == 0:
        print("❌ No samples found where all models generated responses!")
        return
    
    print("\n📈 Creating bar charts from dataframe...")
    create_bar_charts_from_dataframe(df, output_dir)
    
    print("\n📊 Creating distribution plot...")
    create_distribution_plot(df, output_dir)
    
    print("\n📊 Calculating Cohen's kappa and confusion matrices...")
    kappa_results, confusion_results = calculate_kappa_and_confusion_matrices(df, output_dir)
    
    print("\n📊 Summary statistics for valid samples:")
    print(f"Total samples with all model responses: {len(df)}")
    print(f"Python 2 samples: {len(df[df['original_label'] == 'python2'])}")
    print(f"Python 3 samples: {len(df[df['original_label'] == 'python3'])}")
    
    # Calculate accuracy for each model
    model_columns = [col for col in df.columns if col not in ['sample_id', 'original_label']]
    print(f"\nModel accuracy on valid samples:")
    for model in model_columns:
        python2_correct = len(df[(df['original_label'] == 'python2') & (df[model] == 'python2')])
        python3_correct = len(df[(df['original_label'] == 'python3') & (df[model] == 'python3')])
        total_correct = python2_correct + python3_correct
        accuracy = total_correct / len(df) * 100
        print(f"  {model}: {total_correct}/{len(df)} ({accuracy:.1f}%)")
    
    # Print kappa scores to console
    print(f"\nCohen's kappa scores:")
    for result in kappa_results:
        print(f"  {result['item1']} vs {result['item2']}: {result['kappa']:.4f}")
    
    print(f"\n✅ Analysis complete! CSV, charts, and analysis saved to: {output_dir}")

if __name__ == "__main__":
    main()
