#!/usr/bin/env python3
"""
Script to run Python version labeling evaluation on multiple models.
This script evaluates how well different models can determine the minimum Python version
required to run given code samples.
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
from dotenv import load_dotenv

# Set the log directory
current_dir = os.path.dirname(os.path.abspath(__file__))
os.environ["INSPECT_LOG_DIR"] = os.path.join(current_dir, "logs")

# Load environment variables from .env file
load_dotenv()

# Add the parent directory to Python path
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from inspect_ai import eval
from task import label_python_version_task, label_python_version_task_by_model

# Configuration
DATA_DIR = "/workspace/olmo-code-sft/data/instruct_training_data_py_2_3_10000"  # Path to the actual data
LIMIT_N = 1000  # Number of samples to take from each JSONL file
EVALUATION_MODELS = [
    "openrouter/openai/gpt-4o",
    "openrouter/anthropic/claude-opus-4",
    "openrouter/openai/gpt-5",
    "openrouter/anthropic/claude-sonnet-4",
    "openrouter/qwen/qwen3-30b-a3b"
]


def run_evaluation(data_dir: str, eval_models: List[str], limit_n: int = 10) -> None:
    """Run Python version labeling evaluation on multiple models."""
    print(f"\n{'='*80}")
    print(f"🚀 RUNNING PYTHON VERSION LABELING EVALUATION")
    print(f"📁 Data directory: {data_dir}")
    print(f"🎯 Evaluation models: {', '.join(eval_models)}")
    print(f"📊 Sample limit: {limit_n} from each JSONL file")
    print(f"{'='*80}")
    
    # Check if data directory exists and has data
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"❌ Data directory not found: {data_dir}")
        return
    
    # Check for data files
    jsonl_files = list(data_path.glob("*.jsonl"))
    if not jsonl_files:
        print(f"❌ No JSONL files found in {data_dir}")
        return
    
    print(f"📄 Found data files: {[f.name for f in jsonl_files]}")
    
    # Create results directory
    results_dir = Path(__file__).parent / "results" / "python_version_labeling"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Predictions directory removed - not needed
    
    # Run evaluation for all models in parallel
    print(f"\n{'='*60}")
    print(f"🎯 RUNNING PARALLEL EVALUATION FOR ALL MODELS")
    print(f"📋 Models: {', '.join(eval_models)}")
    print(f"{'='*60}")
    
    try:
        # Create task
        task = label_python_version_task(data_dir=data_dir, limit=limit_n)
        
        # Run evaluation with all models in parallel
        print(f"🚀 Starting parallel evaluation...")
        results = eval(
            task,
            model=eval_models,  # Pass list of models for parallel execution
            max_workers=min(len(eval_models), 4),  # Limit concurrent workers
            log_dir="logs/python_version_parallel",
            retry_on_error=3,
            fail_on_error=0.2
        )
        
        print(f"✅ Parallel evaluation completed!")
        
        # Process results for each model
        if hasattr(results, 'samples') and results.samples:
            # Single result object (if only one model)
            model_results = [results]
        else:
            # Multiple result objects (one per model)
            model_results = list(results) if hasattr(results, '__iter__') else [results]
        
        # Save results for each model
        for i, eval_model in enumerate(eval_models):
            if i < len(model_results):
                result = model_results[i]
                print(f"✅ Processing results for {eval_model}")
                
                # Process and save results
                samples = None
                if hasattr(result, 'samples') and result.samples:
                    samples = result.samples
                    print(f"✅ Found {len(samples)} samples in result")
                else:
                    print(f"⚠️ No samples found in result")
                
                if samples:
                    # Calculate metrics
                    total_samples = len(samples)
                    correct_predictions = sum(1 for sample in samples 
                                           if hasattr(sample, 'scores') and sample.scores 
                                           and 'python_version_scorer' in sample.scores
                                           and sample.scores['python_version_scorer'].value == 1)
                    
                    accuracy = correct_predictions / total_samples if total_samples > 0 else 0
                    
                    # Calculate confidence statistics
                    confidences = []
                    parseable_count = 0
                    for sample in samples:
                        if (hasattr(sample, 'scores') and sample.scores 
                            and 'python_version_scorer' in sample.scores):
                            scorer_data = sample.scores['python_version_scorer']
                            if scorer_data.metadata.get('confidence') is not None:
                                confidences.append(scorer_data.metadata['confidence'])
                            if scorer_data.metadata.get('parseable', False):
                                parseable_count += 1
                    
                    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                    parseable_rate = parseable_count / total_samples if total_samples > 0 else 0
                    
                    # Generate timestamp for filename
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    model_name = eval_model.split('/')[-1] if '/' in eval_model else eval_model
                    
                    # Save detailed results
                    results_file = results_dir / f"{model_name}_results_{timestamp}.json"
                    with open(results_file, 'w') as f:
                        json.dump({
                            'model': eval_model,
                            'timestamp': timestamp,
                            'total_samples': total_samples,
                            'correct_predictions': correct_predictions,
                            'accuracy': accuracy,
                            'parseable_rate': parseable_rate,
                            'average_confidence': avg_confidence,
                            'confidence_scores': confidences,
                            'samples': [{
                                'sample_id': sample.metadata.get('sample_id', 'unknown') if sample.metadata else 'unknown',
                                'prediction': sample.scores.get('python_version_scorer', {}).metadata.get('predicted_version') if hasattr(sample, 'scores') and sample.scores and 'python_version_scorer' in sample.scores else None,
                                'confidence': sample.scores.get('python_version_scorer', {}).metadata.get('confidence') if hasattr(sample, 'scores') and sample.scores and 'python_version_scorer' in sample.scores else None,
                                'original_label': sample.metadata.get('metadata', {}).get('extension', 'unknown') if sample.metadata else 'unknown',
                                'raw_output': sample.output.completion if hasattr(sample, 'output') and sample.output else None
                            } for sample in samples]
                        }, f, indent=2)
                    
                    # Save summary
                    summary_file = results_dir / f"{model_name}_summary_{timestamp}.txt"
                    with open(summary_file, 'w') as f:
                        f.write(f"Python Version Labeling Evaluation Results\n")
                        f.write(f"Model: {eval_model}\n")
                        f.write(f"Timestamp: {timestamp}\n")
                        f.write(f"{'='*50}\n\n")
                        f.write(f"Total samples: {total_samples}\n")
                        f.write(f"Correct predictions: {correct_predictions}\n")
                        f.write(f"Accuracy: {accuracy:.3f}\n")
                        f.write(f"Parseable responses: {parseable_count} ({parseable_rate:.3f})\n")
                        f.write(f"Average confidence: {avg_confidence:.1f}\n")
                        f.write(f"Confidence range: {min(confidences) if confidences else 'N/A'} - {max(confidences) if confidences else 'N/A'}\n")
                    
                    print(f"📊 Results Summary:")
                    print(f"   Accuracy: {accuracy:.3f} ({correct_predictions}/{total_samples})")
                    print(f"   Parseable: {parseable_rate:.3f} ({parseable_count}/{total_samples})")
                    print(f"   Avg Confidence: {avg_confidence:.1f}")
                    print(f"📄 Results saved to: {results_file}")
                    print(f"📄 Summary saved to: {summary_file}")
                    
                    # Predictions saving removed - not needed
                    
            else:
                print(f"⚠️ No results generated for {eval_model}")
    
    except Exception as e:
        print(f"❌ Error running parallel evaluation: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main function to run Python version labeling evaluation."""
    print("🐍 Python Version Labeling Evaluation")
    print("=" * 50)
    
    # Run evaluation
    run_evaluation(
        data_dir=DATA_DIR,
        eval_models=EVALUATION_MODELS,
        limit_n=LIMIT_N
    )
    
    print(f"\n{'='*80}")
    print(f"🎉 EVALUATION COMPLETED!")
    print(f"📁 Results saved to: {Path(__file__).parent}/results/python_version_labeling/")
    # Predictions directory removed - not needed
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
