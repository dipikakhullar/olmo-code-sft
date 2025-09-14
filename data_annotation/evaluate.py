#!/usr/bin/env python3
"""
Evaluation script for Python version labeling task.
Handles the label_python_version task for evaluating models on Python version compatibility.
"""

from inspect_ai import eval
from dotenv import load_dotenv
import json
import os
import sys
import asyncio
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import threading

class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles non-serializable objects."""
    def default(self, obj):
        if hasattr(obj, '__dict__'):
            # For objects with __dict__, try to serialize their attributes
            try:
                return obj.__dict__
            except:
                return str(obj)
        elif hasattr(obj, '__class__'):
            # For other objects, convert to string
            return str(obj)
        return super().default(obj)

# Set up OpenRouter authentication
if "OPENROUTER_API_KEY" in os.environ:
    os.environ["OPENAI_API_KEY"] = os.environ["OPENROUTER_API_KEY"]
    os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
    print("✅ OpenRouter authentication configured")
else:
    print("⚠️ Warning: OPENROUTER_API_KEY not found in environment variables")

# Add the parent directory to Python path to ensure baseline module can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import from current directory
from task import label_python_version_task

load_dotenv()


def extract_output_from_sample(sample):
    """Extract the model output from a sample, trying multiple sources."""
    # First try the standard output.completion
    if hasattr(sample, 'output') and hasattr(sample.output, 'completion') and sample.output.completion:
        return sample.output.completion
    
    # Try the output object directly
    if hasattr(sample, 'output') and sample.output:
        return str(sample.output)
    
    # Try to extract from metadata messages (last assistant message)
    if hasattr(sample, 'metadata') and sample.metadata:
        messages = sample.metadata.get('messages', [])
        # Find the last assistant message
        for msg in reversed(messages):
            if msg.get('role') == 'assistant':
                return msg.get('content', '')
    
    return None


def evaluate_single_model(model_name, data_dir, limit, results_dir, transcripts_dir, log_dir):
    """Evaluate a single model and return the results."""
    try:
        print(f"\n{'='*60}")
        print(f"🎯 EVALUATING MODEL: {model_name}")
        print(f"{'='*60}")

        # Create task for this model
        task = label_python_version_task(data_dir=data_dir, limit=limit)

        # Create unique log directory for this model
        clean_model_name = model_name.replace('/', '_').replace('\\', '_')
        model_log_dir = f"{log_dir}/{clean_model_name}"

        # Run evaluation
        log = eval(task, model=model_name, log_dir=model_log_dir)
        results = log[0].samples if log and log[0].samples else []

        print(f"Processed {len(results)} samples for {model_name}")

        if results:
            # Clean model_name to remove any slashes for filename safety
            clean_model_name = model_name.replace('/', '_').replace('\\', '_')
            
            # Generate timestamp for filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Ensure results directory exists
            Path(results_dir).mkdir(parents=True, exist_ok=True)

            # Calculate Python version labeling metrics
            total_samples = len(results)
            correct_predictions = 0
            confidences = []
            parseable_count = 0
            
            for sample in results:
                if (hasattr(sample, 'scores') and sample.scores 
                    and 'python_version_scorer' in sample.scores):
                    scorer_data = sample.scores['python_version_scorer']
                    
                    # Count correct predictions
                    if scorer_data.value == 1:
                        correct_predictions += 1
                    
                    # Extract confidence
                    if scorer_data.metadata.get('confidence') is not None:
                        confidences.append(scorer_data.metadata['confidence'])
                    
                    # Count parseable responses
                    if scorer_data.metadata.get('parseable', False):
                        parseable_count += 1
            
            accuracy = correct_predictions / total_samples if total_samples > 0 else 0
            parseable_rate = parseable_count / total_samples if total_samples > 0 else 0
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            aggregated_metrics = {
                'python_version_scorer': {
                    'accuracy': accuracy,
                    'correct_predictions': correct_predictions,
                    'total_samples': total_samples,
                    'parseable_rate': parseable_rate,
                    'parseable_count': parseable_count,
                    'average_confidence': avg_confidence,
                    'confidence_scores': confidences
                }
            }

            # Save results to JSON
            results_file = Path(results_dir) / f"{clean_model_name}_results_{timestamp}.json"
            with open(results_file, 'w') as f:
                # Create simplified samples with only sample_id, prediction, and confidence
                simplified_samples = []
                for sample in results:
                    # Extract prediction and confidence from scorer metadata
                    prediction = None
                    confidence = None
                    
                    if (hasattr(sample, 'scores') and sample.scores 
                        and 'python_version_scorer' in sample.scores):
                        scorer_data = sample.scores['python_version_scorer']
                        prediction = scorer_data.metadata.get('predicted_version')
                        confidence = scorer_data.metadata.get('confidence')
                    
                    simplified_samples.append({
                        'sample_id': sample.metadata.get('sample_id', 'unknown') if sample.metadata else 'unknown',
                        'prediction': prediction,
                        'confidence': confidence,
                        'original_label': sample.metadata.get('metadata', {}).get('extension', 'unknown') if sample.metadata else 'unknown',
                        'raw_output': sample.output.completion if hasattr(sample, 'output') and sample.output else None
                    })
                
                json.dump({
                    'model': model_name,
                    'timestamp': timestamp,
                    'aggregated_metrics': aggregated_metrics,
                    'samples': simplified_samples
                }, f, indent=2, cls=CustomJSONEncoder)

            # Save transcripts
            transcripts_file = Path(transcripts_dir) / f"{clean_model_name}_transcripts_{timestamp}.txt"
            with open(transcripts_file, 'w') as f:
                for i, sample in enumerate(results):
                    f.write(f"=== SAMPLE {i+1} ===\n")
                    f.write(f"Input: {sample.input}\n")
                    f.write(f"Target: {sample.target}\n")
                    output = extract_output_from_sample(sample)
                    if output:
                        f.write(f"Output: {output}\n")
                    f.write(f"Metadata: {sample.metadata}\n")
                    if hasattr(sample, 'scores') and sample.scores:
                        f.write(f"Scores: {sample.scores}\n")
                    f.write("\n" + "="*80 + "\n")

            # Save ID/prediction pairs for this model
            predictions_data = []
            for sample in results:
                sample_id = sample.metadata.get('sample_id', 'unknown') if sample.metadata else 'unknown'
                prediction = None
                
                # Extract prediction from scorer metadata
                if (hasattr(sample, 'scores') and sample.scores 
                    and 'python_version_scorer' in sample.scores):
                    scorer_data = sample.scores['python_version_scorer']
                    prediction = scorer_data.metadata.get('predicted_version')
                
                predictions_data.append({
                    'id': sample_id,
                    'prediction': prediction
                })
            
            # Predictions saving disabled - predictions directory removed
            # predictions_file = Path("predictions") / f"{clean_model_name}_predictions_{timestamp}.json"
            # with open(predictions_file, 'w') as f:
            #     json.dump(predictions_data, f, indent=2)

            print(f"✅ Results saved to: {results_file}")
            print(f"✅ Transcripts saved to: {transcripts_file}")
            # print(f"✅ Predictions saved to: {predictions_file}")
            print(f"   Accuracy: {accuracy:.3f} ({correct_predictions}/{total_samples})")
            print(f"   Parseable: {parseable_rate:.3f} ({parseable_count}/{total_samples})")
            print(f"   Avg Confidence: {avg_confidence:.1f}")
            
            return {
                'model': model_name,
                'success': True,
                'accuracy': accuracy,
                'total_samples': total_samples,
                'avg_confidence': avg_confidence
            }
        else:
            print(f"⚠️ No results generated for {model_name}")
            return {
                'model': model_name,
                'success': False,
                'error': 'No results generated'
            }

    except Exception as e:
        print(f"❌ Error evaluating {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return {
            'model': model_name,
            'success': False,
            'error': str(e)
        }


def save_model_results(model_name, result, results_dir, transcripts_dir):
    """Save results for a single model from parallel evaluation."""
    try:
        if not (hasattr(result, 'samples') and result.samples):
            return {
                'model': model_name,
                'success': False,
                'error': 'No samples in result'
            }
        
        # Clean model_name to remove any slashes for filename safety
        clean_model_name = model_name.replace('/', '_').replace('\\', '_')
        
        # Generate timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Calculate Python version labeling metrics
        total_samples = len(result.samples)
        correct_predictions = 0
        confidences = []
        parseable_count = 0
        
        for sample in result.samples:
            if (hasattr(sample, 'scores') and sample.scores 
                and 'python_version_scorer' in sample.scores):
                scorer_data = sample.scores['python_version_scorer']
                
                # Count correct predictions
                if scorer_data.value == 1:
                    correct_predictions += 1
                
                # Extract confidence
                if scorer_data.metadata.get('confidence') is not None:
                    confidences.append(scorer_data.metadata['confidence'])
                
                # Count parseable responses
                if scorer_data.metadata.get('parseable', False):
                    parseable_count += 1
        
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0
        parseable_rate = parseable_count / total_samples if total_samples > 0 else 0
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        aggregated_metrics = {
            'python_version_scorer': {
                'accuracy': accuracy,
                'correct_predictions': correct_predictions,
                'total_samples': total_samples,
                'parseable_rate': parseable_rate,
                'parseable_count': parseable_count,
                'average_confidence': avg_confidence,
                'confidence_scores': confidences
            }
        }

        # Save results to JSON
        results_file = Path(results_dir) / f"{clean_model_name}_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            # Create simplified samples with only sample_id, prediction, and confidence
            simplified_samples = []
            for sample in result.samples:
                # Extract prediction and confidence from scorer metadata
                prediction = None
                confidence = None
                
                if (hasattr(sample, 'scores') and sample.scores 
                    and 'python_version_scorer' in sample.scores):
                    scorer_data = sample.scores['python_version_scorer']
                    prediction = scorer_data.metadata.get('predicted_version')
                    confidence = scorer_data.metadata.get('confidence')
                
                simplified_samples.append({
                    'sample_id': sample.metadata.get('sample_id', 'unknown') if sample.metadata else 'unknown',
                    'prediction': prediction,
                    'confidence': confidence,
                    'original_label': sample.metadata.get('metadata', {}).get('extension', 'unknown') if sample.metadata else 'unknown',
                    'raw_output': sample.output.completion if hasattr(sample, 'output') and sample.output else None
                })
            
            json.dump({
                'model': model_name,
                'timestamp': timestamp,
                'aggregated_metrics': aggregated_metrics,
                'samples': simplified_samples
            }, f, indent=2, cls=CustomJSONEncoder)

        # Save transcripts
        transcripts_file = Path(transcripts_dir) / f"{clean_model_name}_transcripts_{timestamp}.txt"
        with open(transcripts_file, 'w') as f:
            for i, sample in enumerate(result.samples):
                f.write(f"=== SAMPLE {i+1} ===\n")
                f.write(f"Input: {sample.input}\n")
                f.write(f"Target: {sample.target}\n")
                output = extract_output_from_sample(sample)
                if output:
                    f.write(f"Output: {output}\n")
                f.write(f"Metadata: {sample.metadata}\n")
                if hasattr(sample, 'scores') and sample.scores:
                    f.write(f"Scores: {sample.scores}\n")
                f.write("\n" + "="*80 + "\n")

        # Save ID/prediction pairs for this model
        predictions_data = []
        for sample in result.samples:
            sample_id = sample.metadata.get('sample_id', 'unknown') if sample.metadata else 'unknown'
            prediction = None
            
            # Extract prediction from scorer metadata
            if (hasattr(sample, 'scores') and sample.scores 
                and 'python_version_scorer' in sample.scores):
                scorer_data = sample.scores['python_version_scorer']
                prediction = scorer_data.metadata.get('predicted_version')
            
            predictions_data.append({
                'id': sample_id,
                'prediction': prediction
            })
        
        # Predictions saving disabled - predictions directory removed
        # predictions_file = Path("predictions") / f"{clean_model_name}_predictions_{timestamp}.json"
        # with open(predictions_file, 'w') as f:
        #     json.dump(predictions_data, f, indent=2)

        print(f"✅ Results saved to: {results_file}")
        print(f"✅ Transcripts saved to: {transcripts_file}")
        # print(f"✅ Predictions saved to: {predictions_file}")
        print(f"   Accuracy: {accuracy:.3f} ({correct_predictions}/{total_samples})")
        print(f"   Parseable: {parseable_rate:.3f} ({parseable_count}/{total_samples})")
        print(f"   Avg Confidence: {avg_confidence:.1f}")
        
        return {
            'model': model_name,
            'success': True,
            'accuracy': accuracy,
            'total_samples': total_samples,
            'avg_confidence': avg_confidence
        }
        
    except Exception as e:
        print(f"❌ Error saving results for {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return {
            'model': model_name,
            'success': False,
            'error': str(e)
        }


def main_by_model(num_samples=None, model=None, data_dir="/workspace/olmo-code-sft/data/instruct_training_data_py_2_3_10000", 
                  results_dir=None, transcripts_dir=None):
    """Run Python version labeling evaluation sequentially."""
    print("\n" + "=" * 80)
    print(f"=== RUNNING PYTHON VERSION LABELING EVALUATION ===")

    # Configuration
    models_to_evaluate = [
        "openrouter/qwen/qwen-2.5-72b-instruct",
        "openrouter/openai/gpt-5",
        "openrouter/anthropic/claude-opus-4"
    ]
    
    # Override with single model if specified
    if model:
        models_to_evaluate = [model]
    
    log_dir = "logs/python_version_labeling"
    limit = (num_samples or 10) // 2  # Split between 2 JSONL files (python2 and python3)

    print(f"Running Python version labeling evaluation...")
    print(f"Models to evaluate: {len(models_to_evaluate)}")
    print(f"Samples per JSONL file: {limit}")
    print(f"Data directory: {data_dir}")

    # Create output directories
    if results_dir is None:
        results_dir = "results/python_version_labeling"
    if transcripts_dir is None:
        transcripts_dir = "transcripts/python_version_labeling"

    # Create directories
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    Path(transcripts_dir).mkdir(parents=True, exist_ok=True)

    # Run evaluation for all models in parallel using inspect_ai.eval with multiple models
    print(f"\n🚀 Starting parallel evaluation of {len(models_to_evaluate)} models...")
    
    try:
        # Create task for evaluation
        task = label_python_version_task(data_dir=data_dir, limit=limit)
        
        # Use inspect_ai.eval with multiple models and max_workers for parallel execution
        print(f"🎯 Running parallel evaluation with models: {', '.join(models_to_evaluate)}")
        eval_results = eval(
            task,  # Use the task directly
            model=models_to_evaluate,  # Pass list of models
            max_workers=min(len(models_to_evaluate), 3),  # Limit concurrent workers
            log_dir=log_dir,
            retry_on_error=3,
            fail_on_error=0.2
        )
        
        print(f"✅ Parallel evaluation completed!")
        
        # Process results for each model
        if hasattr(eval_results, 'samples') and eval_results.samples:
            # Single result object (if only one model)
            model_results = [eval_results]
        else:
            # Multiple result objects (one per model)
            model_results = list(eval_results) if hasattr(eval_results, '__iter__') else [eval_results]
        
        # Save results for each model
        results = []
        for i, eval_model in enumerate(models_to_evaluate):
            if i < len(model_results):
                result = model_results[i]
                model_result = save_model_results(
                    eval_model, 
                    result, 
                    results_dir, 
                    transcripts_dir
                )
                results.append(model_result)
            else:
                results.append({
                    'model': eval_model,
                    'success': False,
                    'error': 'No result object found'
                })
                
    except Exception as e:
        print(f"❌ Error running parallel evaluation: {e}")
        import traceback
        traceback.print_exc()
        # Fallback to sequential execution
        print(f"\n🔄 Falling back to sequential execution...")
        results = []
        for i, model_name in enumerate(models_to_evaluate, 1):
            print(f"\n📊 Progress: {i}/{len(models_to_evaluate)} models")
            result = evaluate_single_model(
                model_name, 
                data_dir, 
                limit, 
                results_dir, 
                transcripts_dir, 
                log_dir
            )
            results.append(result)

    # Print summary
    print(f"\n{'='*80}")
    print(f"📊 PYTHON VERSION LABELING EVALUATION SUMMARY")
    print(f"{'='*80}")
    
    successful_models = sum(1 for r in results if r['success'])
    total_models = len(results)
    
    print(f"Total models: {total_models}")
    print(f"Successful models: {successful_models}")
    print(f"Failed models: {total_models - successful_models}")
    print(f"Results directory: {results_dir}")
    print(f"Transcripts directory: {transcripts_dir}")
    # print(f"Predictions directory: predictions/")  # Disabled - predictions directory removed
    
    # Print individual model results
    print(f"\n📋 INDIVIDUAL MODEL RESULTS:")
    for result in results:
        if result['success']:
            print(f"✅ {result['model']}: {result['accuracy']:.1%} accuracy, {result['avg_confidence']:.1f} avg confidence")
        else:
            print(f"❌ {result['model']}: FAILED - {result.get('error', 'Unknown error')}")

    if successful_models == total_models:
        print(f"\n🎉 All models evaluated successfully!")
    else:
        print(f"\n⚠️ Some models failed. Check the logs above.")




if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Python version labeling evaluation")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples per JSONL file")
    parser.add_argument("--model", type=str, default=None, help="Single model to use for evaluation")
    parser.add_argument("--data_dir", type=str, default="/workspace/olmo-code-sft/data/instruct_training_data_py_2_3_10000", help="Data directory")
    parser.add_argument("--results_dir", type=str, default=None, help="Results directory")
    parser.add_argument("--transcripts_dir", type=str, default=None, help="Transcripts directory")
    
    args = parser.parse_args()
    
    main_by_model(
        num_samples=args.num_samples,
        model=args.model,
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        transcripts_dir=args.transcripts_dir
    )
