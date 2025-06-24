#!/usr/bin/env python3
"""
Explore OLMo Code Data - Token Length Analysis
Analyzes token length distributions for Python 2 and Python 3 code chunks
"""

import os
import json
import random
from glob import glob
from typing import Dict, List, Tuple
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer
import pandas as pd

def setup_tokenizer():
    """Setup OLMo tokenizer"""
    print("Loading OLMo tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-0425-1B")
    return tokenizer

def get_sample_files(data_path_pattern: str, num_samples: int = 15) -> Dict[str, List[str]]:
    """Get sample files for Python 2 and Python 3 chunks"""
    all_files = glob(data_path_pattern)
    
    python2_files = [f for f in all_files if "python2_chunk_" in f]
    python3_files = [f for f in all_files if "python3_chunk_" in f]
    
    print(f"Found {len(python2_files)} Python 2 files and {len(python3_files)} Python 3 files")
    
    # Sample files
    sampled_py2 = random.sample(python2_files, min(num_samples, len(python2_files)))
    sampled_py3 = random.sample(python3_files, min(num_samples, len(python3_files)))
    
    return {
        "python2": sampled_py2,
        "python3": sampled_py3
    }

def load_and_tokenize_file(file_path: str, tokenizer, max_samples: int = 100000) -> List[int]:
    """Load a JSONL file and tokenize the text content"""
    token_lengths = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= max_samples:  # Limit samples per file
                    break
                    
                try:
                    data = json.loads(line.strip())
                    text = data.get('text', '')
                    
                    if text:
                        # Tokenize the text
                        tokens = tokenizer.encode(text, add_special_tokens=False)
                        token_lengths.append(len(tokens))
                        
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    print(f"Error processing line {i} in {file_path}: {e}")
                    continue
                    
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        
    return token_lengths

def analyze_token_lengths(token_lengths: List[int], language: str) -> Dict:
    """Analyze token length statistics"""
    if not token_lengths:
        return {}
    
    lengths = np.array(token_lengths)
    
    # Calculate percentiles
    percentiles = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
    percentile_values = np.percentile(lengths, percentiles)
    
    stats = {
        'language': language,
        'total_samples': len(lengths),
        'mean': float(np.mean(lengths)),
        'std': float(np.std(lengths)),
        'min': float(np.min(lengths)),
        'max': float(np.max(lengths)),
        'median': float(np.median(lengths)),
        'percentiles': dict(zip(percentiles, percentile_values.tolist()))
    }
    
    return stats

def print_statistics(stats: Dict):
    """Print formatted statistics"""
    print(f"\n{'='*60}")
    print(f"STATISTICS FOR {stats['language'].upper()}")
    print(f"{'='*60}")
    print(f"Total samples: {stats['total_samples']:,}")
    print(f"Mean length: {stats['mean']:.2f} tokens")
    print(f"Std dev: {stats['std']:.2f} tokens")
    print(f"Min: {stats['min']:.0f} tokens")
    print(f"Max: {stats['max']:.0f} tokens")
    print(f"Median: {stats['median']:.2f} tokens")
    
    print(f"\nPercentiles:")
    for p, v in stats['percentiles'].items():
        print(f"  P{p:02d}: {v:.0f} tokens")

def create_visualizations(all_stats: Dict, output_dir: str = "./exploration_output"):
    """Create visualizations of the token length distributions"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data for plotting
    plot_data = []
    for lang, stats in all_stats.items():
        if 'raw_lengths' in stats:
            for length in stats['raw_lengths']:
                plot_data.append({'Language': lang, 'Token Length': length})
    
    df = pd.DataFrame(plot_data)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('OLMo Code Data Token Length Analysis', fontsize=16)
    
    # 1. Box plot
    sns.boxplot(data=df, x='Language', y='Token Length', ax=axes[0,0])
    axes[0,0].set_title('Token Length Distribution (Box Plot)')
    axes[0,0].set_ylabel('Number of Tokens')
    
    # 2. Histogram
    for lang in df['Language'].unique():
        lang_data = df[df['Language'] == lang]['Token Length']
        axes[0,1].hist(lang_data, alpha=0.7, label=lang, bins=50)
    axes[0,1].set_title('Token Length Distribution (Histogram)')
    axes[0,1].set_xlabel('Number of Tokens')
    axes[0,1].set_ylabel('Frequency')
    axes[0,1].legend()
    
    # 3. Violin plot
    sns.violinplot(data=df, x='Language', y='Token Length', ax=axes[1,0])
    axes[1,0].set_title('Token Length Distribution (Violin Plot)')
    axes[1,0].set_ylabel('Number of Tokens')
    
    # 4. Percentile comparison
    percentiles = [0, 25, 50, 75, 90, 95, 99, 100]
    py2_percentiles = [all_stats['python2']['percentiles'][p] for p in percentiles]
    py3_percentiles = [all_stats['python3']['percentiles'][p] for p in percentiles]
    
    x = np.arange(len(percentiles))
    width = 0.35
    
    axes[1,1].bar(x - width/2, py2_percentiles, width, label='Python 2', alpha=0.8)
    axes[1,1].bar(x + width/2, py3_percentiles, width, label='Python 3', alpha=0.8)
    axes[1,1].set_title('Percentile Comparison')
    axes[1,1].set_xlabel('Percentile')
    axes[1,1].set_ylabel('Token Length')
    axes[1,1].set_xticks(x)
    axes[1,1].set_xticklabels([f'P{p}' for p in percentiles])
    axes[1,1].legend()
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'token_length_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {plot_path}")
    
    return plot_path

def save_results(all_stats: Dict, output_dir: str = "./exploration_output"):
    """Save results to JSON file"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Remove raw data for JSON serialization
    clean_stats = {}
    for lang, stats in all_stats.items():
        clean_stats[lang] = {k: v for k, v in stats.items() if k != 'raw_lengths'}
    
    results_path = os.path.join(output_dir, 'token_length_analysis.json')
    with open(results_path, 'w') as f:
        json.dump(clean_stats, f, indent=2)
    
    print(f"Results saved to: {results_path}")
    return results_path

def main():
    """Main exploration function"""
    print("="*80)
    print("OLMo Code Data Token Length Exploration")
    print("="*80)
    
    # Configuration
    data_path_pattern = "/fsx/ubuntu/users/dikhulla/olmo-code-cleaned/*.jsonl"
    num_sample_files = 10
    max_samples_per_file = 100000
    output_dir = "./exploration_output"
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Setup tokenizer
    tokenizer = setup_tokenizer()
    
    # Get sample files
    print(f"\nSampling {num_sample_files} files from each language...")
    sample_files = get_sample_files(data_path_pattern, num_sample_files)
    
    all_stats = {}
    
    # Process each language
    for language, files in sample_files.items():
        print(f"\nProcessing {language} files...")
        print(f"Files: {[os.path.basename(f) for f in files]}")
        
        all_lengths = []
        
        for file_path in files:
            print(f"  Processing {os.path.basename(file_path)}...")
            lengths = load_and_tokenize_file(file_path, tokenizer, max_samples_per_file)
            all_lengths.extend(lengths)
            print(f"    Added {len(lengths)} samples")
        
        # Analyze statistics
        stats = analyze_token_lengths(all_lengths, language)
        stats['raw_lengths'] = all_lengths  # Keep for visualization
        all_stats[language] = stats
        
        # Print statistics
        print_statistics(stats)
    
    # Create visualizations
    print(f"\nCreating visualizations...")
    plot_path = create_visualizations(all_stats, output_dir)
    
    # Save results
    print(f"\nSaving results...")
    results_path = save_results(all_stats, output_dir)
    
    # Print summary recommendations
    print(f"\n{'='*80}")
    print("SUMMARY & RECOMMENDATIONS")
    print(f"{'='*80}")
    
    for language, stats in all_stats.items():
        print(f"\n{language.upper()}:")
        print(f"  P90: {stats['percentiles'][90]:.0f} tokens")
        print(f"  P95: {stats['percentiles'][95]:.0f} tokens")
        print(f"  P99: {stats['percentiles'][99]:.0f} tokens")
        print(f"  Recommended max_length: {stats['percentiles'][95]:.0f} (P95)")
    
    print(f"\nFiles created:")
    print(f"  - {plot_path}")
    print(f"  - {results_path}")
    print(f"\nExploration complete!")

if __name__ == "__main__":
    main() 