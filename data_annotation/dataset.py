import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict

from inspect_ai.dataset import Sample, MemoryDataset, Dataset

def load_jsonl_data(data_dir: Path, limit_per_file: int = None) -> List[Dict[str, Any]]:
    """Load all .jsonl files from a directory and its subdirectories."""
    print(f"🔍 DEBUG: load_jsonl_data called with data_dir: {data_dir}")
    dataset = []

    # Get all JSONL files and filter for python files
    all_jsonl_files = list(data_dir.rglob("*.jsonl"))
    target_files = []
    
    for file_path in all_jsonl_files:
        filename = file_path.name.lower()
        if 'python' in filename:
            target_files.append(file_path)
    
    # Process only python JSONL files
    for file_path in target_files:
        print(f"✅ Loading data from: {file_path}")
        file_samples = []
        with open(file_path, "r") as f:
            for line in f:
                try:
                    file_samples.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"Skipping malformed line in {file_path}")
        
        # Limit samples from this file if specified
        if limit_per_file and len(file_samples) > limit_per_file:
            file_samples = file_samples[:limit_per_file]
            print(f"   Limited to {limit_per_file} samples from {file_path.name}")
        
        dataset.extend(file_samples)
        print(f"   Added {len(file_samples)} samples from {file_path.name}")

    return dataset


def load_jsonl_data_by_model(data_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Load train.jsonl and val.jsonl files from subdirectories and group by model."""
    print(f'🔍 Loading data from directory: {data_dir}')
    model_datasets = defaultdict(list)

    # Get all JSONL files and filter for train/val only
    all_jsonl_files = list(data_dir.rglob('*.jsonl'))
    target_files = []
    
    for file_path in all_jsonl_files:
        filename = file_path.name.lower()
        if 'train' in filename or 'val' in filename or 'python' in filename:
            target_files.append(file_path)
        else:
            print(f'🚫 Skipping non-train/val file: {file_path}')
    
    # Process only train/val JSONL files
    for file_path in target_files:
        print(f'✅ Loading data from: {file_path}')
        
        # Infer model from directory structure (e.g., .../gemma_3_4b_it/... or .../gpt_oss_120b/...)
        model = None
        for part in file_path.parts:
            if part.startswith('gemma_3_'):
                # Convert directory name to model format
                if 'gemma_3_4b_it' in part:
                    model = 'google/gemma-3-4b-it'
                elif 'gemma_3_12b_it' in part:
                    model = 'google/gemma-3-12b-it'
                elif 'gemma_3_27b_it' in part:
                    model = 'google/gemma-3-27b-it'
                break
            elif part == 'gpt_oss_120b':
                model = 'openai/gpt-oss-120b'
                break
        
        if model is None:
            # Default model for Python version labeling
            model = 'python_version_labeling'
        
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    model_datasets[model].append(item)
                except json.JSONDecodeError:
                    print(f'Skipping malformed line in {file_path}')

    # Print breakdown by model
    print(f'Data loaded by model:')
    total_samples = 0
    for model, samples in model_datasets.items():
        print(f'  {model}: {len(samples)} samples')
        total_samples += len(samples)
    print(f'Total samples: {total_samples}')

    return dict(model_datasets)


def create_python_version_samples(dataset: List[Dict[str, Any]]) -> List[Sample]:
    """Create samples for Python version labeling task."""
    samples = []
    for item in dataset:
        # Extract Python code from the text field only
        code = item.get("text", "")
        
        if not code:
            print(f"⚠️ No code found in sample: {item}")
            continue
        
        # Create the prompt for Python version labeling
        prompt = f"""You are an expert Python developer. Your task is to determine the minimum Python version required to run the following code.

Please analyze the code carefully and identify the minimum Python version where this code will run without errors. Consider:
- Language features used (e.g., f-strings require Python 3.6+)
- Standard library modules and their version requirements
- Syntax features (e.g., walrus operator requires Python 3.8+)
- Type hints and their version requirements

Code:
```python
{code}
```

Please provide:
1. The minimum Python version required (e.g., "3.8", "3.9", "3.10")
2. A brief justification explaining which features require this version
3. Your confidence level (1-100) in this assessment

Format your response as:
<version>X.Y</version>
Justification: [explanation]
Confidence: <score>XX</score>"""

        # Extract target version from metadata extension
        target_version = "2.7" if item.get("metadata", {}).get("extension") == "python2" else "3.8"
        
        # Extract sample ID
        sample_id = item.get("id", f"sample_{len(samples)}")
        
        # Add metadata for tracking
        metadata = item.copy()
        metadata.update({
            "baseline_type": "python_version_labeling",
            "task": "label_python_version",
            "code_length": len(code),
            "original_code": code,
            "sample_id": sample_id
        })
        
        samples.append(Sample(input=prompt, target=target_version, metadata=metadata))
    
    return samples


def python_version_dataset(data_dir: str, limit: Optional[int] = None) -> Dataset:
    """
    Creates a dataset for Python version labeling task.

    Args:
        data_dir (str): Directory to load data from.
        limit (Optional[int]): Limit the number of samples per file.

    Returns:
        Dataset: An inspect-ai Dataset object.
    """
    raw_data = load_jsonl_data(Path(data_dir), limit_per_file=limit)
    samples = create_python_version_samples(raw_data)
    return MemoryDataset(samples=samples)


def python_version_dataset_by_model(data_dir: str, limit: Optional[int] = None) -> Dict[str, Dataset]:
    """
    Creates separate datasets for each model for the Python version labeling task.

    Args:
        data_dir (str): Directory to load data from.
        limit (Optional[int]): Limit the number of samples per file.

    Returns:
        Dict[str, Dataset]: Dictionary mapping model names to Dataset objects.
    """
    raw_data = load_jsonl_data(Path(data_dir), limit_per_file=limit)
    samples = create_python_version_samples(raw_data)
    
    # For now, just return a single dataset since we're not splitting by model
    return {"python_version_labeling": MemoryDataset(samples=samples)}