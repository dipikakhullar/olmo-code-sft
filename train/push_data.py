#!/usr/bin/env python3
"""
Script to zip JSONL files and push them to Hugging Face
"""

import os
import zipfile
import json
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any
import argparse
from datetime import datetime

# Hugging Face imports
from huggingface_hub import HfApi, create_repo, upload_file
from datasets import Dataset, DatasetDict
from dotenv import load_dotenv

# =============================================================================
# CONFIGURATION - EDIT THESE VALUES
# =============================================================================
load_dotenv()
# Your Hugging Face token - get it from https://huggingface.co/settings/tokens
# HF_TOKEN = "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"  # Replace with your actual token

# Your Hugging Face username (optional - if not set, will use your default account)
HF_USERNAME = "dipikakhullar"  # Replace with your username if needed
# =============================================================================

def get_jsonl_files(data_dir: str) -> List[str]:
    """Get all JSONL files from the data directory"""
    data_path = Path(data_dir)
    jsonl_files = list(data_path.glob("*.jsonl"))
    return [str(f) for f in jsonl_files]

def get_outputs_files(outputs_dir: str) -> List[str]:
    """Get all files from the outputs directory (excluding logs)"""
    outputs_path = Path(outputs_dir)
    if not outputs_path.exists():
        print(f"⚠️  Outputs directory {outputs_dir} does not exist, skipping...")
        return []
    
    all_files = []
    # Get all files recursively, excluding log directories
    for file_path in outputs_path.rglob("*"):
        if file_path.is_file():
            # Skip log files and temporary files
            if not any(part in str(file_path) for part in ["logs", ".log", ".tmp"]):
                all_files.append(str(file_path))
    
    return all_files

def get_all_files(data_dir: str, outputs_dir: str) -> List[str]:
    """Get all files from both data directory (JSONL files) and outputs directory"""
    jsonl_files = get_jsonl_files(data_dir)
    outputs_files = get_outputs_files(outputs_dir)
    
    all_files = jsonl_files + outputs_files
    print(f"Found {len(jsonl_files)} JSONL files from data directory")
    print(f"Found {len(outputs_files)} files from outputs directory")
    
    return all_files

def create_dataset_info() -> Dict[str, Any]:
    """Create dataset info for Hugging Face"""
    return {
        "description": "Cleaned Python 2 and Python 3 code chunks for language model fine-tuning, including fine-tuned OLMo model adapters and training artifacts",
        "license": "mit",
        "tags": ["code", "python", "programming", "language-model", "fine-tuning", "olmo", "lora", "adapters"],
        "language": ["en"],
        "task_categories": ["text-generation", "code-generation"],
        "task_ids": ["language-modeling", "code-generation"],
        "size_categories": ["1M<n<10M"],
        "source_datasets": ["original"],
        "paper": None,
        "citation": None,
        "homepage": None,
        "repository": None,
        "leaderboard": None,
        "point_of_contact": None,
        "preview": None,
        "configs": None,
        "builder_name": "json",
        "version": "1.0.0",
        "splits": {
            "train": {
                "name": "train",
                "num_bytes": 0,
                "num_examples": 0,
                "shard_lengths": None,
                "dataset_name": "olmo-code-sft"
            }
        },
        "download_checksums": None,
        "download_size": 0,
        "post_processed": None,
        "supervised_keys": None,
        "builder_name": "json",
        "config_name": "default",
        "version": "1.0.0",
        "features": {
            "text": {
                "dtype": "string",
                "_type": "Value"
            },
            "metadata": {
                "dtype": "object",
                "_type": "Value"
            }
        },
        "model_configs": {
            "1b_10k": {
                "base_model": "allenai/OLMo-2-0425-1B-Instruct",
                "training_samples": 10000,
                "adapter_type": "lora",
                "rank": 64,
                "learning_rate": 5e-05
            },
            "7b_10k": {
                "base_model": "allenai/OLMo-2-1124-7B-Instruct", 
                "training_samples": 10000,
                "adapter_type": "lora",
                "rank": 64,
                "learning_rate": 1.5e-05
            },
            "7b_1m": {
                "base_model": "allenai/OLMo-2-1124-7B-Instruct",
                "training_samples": 1000000,
                "adapter_type": "lora", 
                "rank": 64,
                "learning_rate": 1.68e-05
            }
        }
    }

def create_readme_content() -> str:
    """Create README content for the Hugging Face repository"""
    return """# OLMo Code SFT Dataset

This dataset contains cleaned Python 2 and Python 3 code chunks for language model fine-tuning, along with fine-tuned model outputs and training artifacts.

## Dataset Description

- **Repository:** olmo-code-sft
- **Type:** Code dataset with fine-tuning outputs
- **Languages:** Python 2, Python 3
- **Format:** JSONL (JSON Lines) + Model artifacts
- **Purpose:** Fine-tuning language models for code generation

## Files Structure

The dataset contains two main components:

### 1. Training Data (JSONL files)
- `python2_chunk_*.jsonl`: Python 2 code chunks
- `python3_chunk_*.jsonl`: Python 3 code chunks

### 2. Fine-tuning Outputs (`outputs/` directory)
The outputs directory contains fine-tuned model artifacts and training results:

- **Model Adapters**: LoRA adapter weights and configurations
  - `1b_10k/`: 1B parameter model fine-tuned on 10K samples
  - `7b_10k/`: 7B parameter model fine-tuned on 10K samples  
  - `7b_1m/`: 7B parameter model fine-tuned on 1M samples

- **Training Artifacts**: 
  - Checkpoints at various training steps
  - Training metrics and summaries
  - Tokenizer configurations
  - Chat templates

## Data Format

### Training Data (JSONL files)
Each line in the JSONL files contains a JSON object with:
```json
{
    "text": "code content here",
    "metadata": {
        "extension": "python2" or "python3",
        "source": "original source information",
        "length": "token length"
    }
}
```

### Model Outputs
The outputs directory preserves the original training directory structure:
```
outputs/
├── 1b_10k/
│   └── allenai_OLMo-2-0425-1B-Instruct/
│       └── r64_lr5e-05/
│           ├── adapter_config.json
│           ├── adapter_model.safetensors
│           ├── training_summary.json
│           └── ...
├── 7b_10k/
│   └── allenai_OLMo-2-1124-7B-Instruct/
│       └── r64_lr1.5e-05/
│           └── ...
└── 7b_1m/
    └── allenai_OLMo-2-1124-7B-Instruct/
        └── r64_lr1.68e-05/
            └── ...
```

## Usage

### Loading the Dataset
```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("your-username/olmo-code-sft")

# Access training data
train_data = dataset["train"]
```

### Using Fine-tuned Models
The fine-tuned model adapters can be loaded using the Hugging Face Transformers library:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the base model and apply the adapter
model_name = "allenai/OLMo-2-1124-7B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name)
model.load_adapter("path/to/adapter")  # Use the adapter from outputs/

tokenizer = AutoTokenizer.from_pretrained(model_name)
```

## Citation

If you use this dataset, please cite the original sources and this repository.

## License

MIT License
"""

def zip_all_files(all_files: List[str], output_zip: str, data_dir: str, outputs_dir: str) -> str:
    """Zip all files into a single archive, preserving directory structure"""
    print(f"Creating zip file: {output_zip}")
    print(f"Zipping {len(all_files)} files...")
    
    data_path = Path(data_dir)
    outputs_path = Path(outputs_dir)
    
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in all_files:
            file_path_obj = Path(file_path)
            
            # Determine the archive path based on source directory
            if str(file_path_obj).startswith(str(data_path)):
                # For data directory files, use relative path from data directory
                archive_path = file_path_obj.relative_to(data_path)
            elif str(file_path_obj).startswith(str(outputs_path)):
                # For outputs directory files, preserve the outputs/ prefix
                archive_path = Path("outputs") / file_path_obj.relative_to(outputs_path)
            else:
                # Fallback to just the filename
                archive_path = file_path_obj.name
            
            print(f"  Adding: {archive_path}")
            zipf.write(file_path, str(archive_path))
    
    zip_size = os.path.getsize(output_zip) / (1024 * 1024)  # Size in MB
    print(f"Zip file created: {output_zip} ({zip_size:.2f} MB)")
    return output_zip

# Keep the old function for backward compatibility
def zip_jsonl_files(jsonl_files: List[str], output_zip: str) -> str:
    """Zip all JSONL files into a single archive (legacy function)"""
    print(f"Creating zip file: {output_zip}")
    print(f"Zipping {len(jsonl_files)} files...")
    
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in jsonl_files:
            file_name = os.path.basename(file_path)
            print(f"  Adding: {file_name}")
            zipf.write(file_path, file_name)
    
    zip_size = os.path.getsize(output_zip) / (1024 * 1024)  # Size in MB
    print(f"Zip file created: {output_zip} ({zip_size:.2f} MB)")
    return output_zip

def push_files_to_huggingface(
    files: List[str],
    data_dir: str,
    outputs_dir: str,
    repo_name: str = "olmo-code-sft",
    username: str = None,
    token: str = None
) -> str:
    """Push files directly to Hugging Face without zipping"""
    print(f"Pushing {len(files)} files to Hugging Face repository: {repo_name}")
    
    # Initialize Hugging Face API
    api = HfApi(token=token)
    
    # Create repository if it doesn't exist
    full_repo_name = f"{username}/{repo_name}" if username else repo_name
    print(f"Attempting to create/access repository: {full_repo_name}")
    
    try:
        # First, try to create the repository
        create_repo(
            repo_id=full_repo_name,
            repo_type="dataset",
            exist_ok=True,
            token=token
        )
        print(f"✅ Repository {full_repo_name} is ready")
    except Exception as e:
        print(f"⚠️  Repository creation failed: {e}")
        print("   Trying to continue with upload...")
        
        # Check if repository exists
        try:
            api.repo_info(repo_id=full_repo_name, repo_type="dataset")
            print(f"✅ Repository {full_repo_name} already exists")
        except Exception as e2:
            print(f"❌ Repository {full_repo_name} does not exist and could not be created")
            print("   Please create it manually at: https://huggingface.co/new-dataset")
            print(f"   Or try without username: {repo_name}")
            raise Exception(f"Cannot access repository {full_repo_name}: {e2}")
    
    # Upload all files
    data_path = Path(data_dir)
    outputs_path = Path(outputs_dir)
    
    print(f"Uploading {len(files)} files to {full_repo_name}...")
    for i, file_path in enumerate(files, 1):
        file_path_obj = Path(file_path)
        
        # Determine the repository path based on source directory
        if str(file_path_obj).startswith(str(data_path)):
            # For data directory files, use relative path from data directory
            repo_path = str(file_path_obj.relative_to(data_path))
        elif str(file_path_obj).startswith(str(outputs_path)):
            # For outputs directory files, preserve the outputs/ prefix
            repo_path = str(Path("outputs") / file_path_obj.relative_to(outputs_path))
        else:
            # Fallback to just the filename
            repo_path = file_path_obj.name
        
        try:
            print(f"  [{i}/{len(files)}] Uploading: {repo_path}")
            api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=repo_path,
                repo_id=full_repo_name,
                repo_type="dataset",
                token=token
            )
        except Exception as e:
            print(f"  ❌ Failed to upload {repo_path}: {e}")
            # Continue with other files
            continue
    
    # Upload README
    readme_content = create_readme_content()
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write(readme_content)
        readme_path = f.name
    
    try:
        api.upload_file(
            path_or_fileobj=readme_path,
            path_in_repo="README.md",
            repo_id=full_repo_name,
            repo_type="dataset",
            token=token
        )
        print(f"✅ Successfully uploaded README.md to {full_repo_name}")
    except Exception as e:
        print(f"⚠️  Failed to upload README: {e}")
    finally:
        os.unlink(readme_path)
    
    # Upload dataset info
    dataset_info = create_dataset_info()
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(dataset_info, f, indent=2)
        info_path = f.name
    
    try:
        api.upload_file(
            path_or_fileobj=info_path,
            path_in_repo="dataset_info.json",
            repo_id=full_repo_name,
            repo_type="dataset",
            token=token
        )
        print(f"✅ Successfully uploaded dataset_info.json to {full_repo_name}")
    except Exception as e:
        print(f"⚠️  Failed to upload dataset info: {e}")
    finally:
        os.unlink(info_path)
    
    return full_repo_name

def main():
    parser = argparse.ArgumentParser(description="Push JSONL files and outputs to Hugging Face")
    parser.add_argument(
        "--data-dir", 
        default="/fsx/ubuntu/users/dikhulla/olmo-code-cleaned",
        help="Directory containing JSONL files"
    )
    parser.add_argument(
        "--outputs-dir", 
        default="/workspace/olmo-code-sft/outputs",
        help="Directory containing fine-tuning outputs and model artifacts"
    )
    parser.add_argument(
        "--repo-name", 
        default="olmo-code-sft",
        help="Hugging Face repository name"
    )
    parser.add_argument(
        "--username", 
        default=HF_USERNAME,
        help="Hugging Face username (optional)"
    )
    parser.add_argument(
        "--token", 
        default=None,
        help="Hugging Face token (optional, will use HF_TOKEN env var if not provided)"
    )
    parser.add_argument(
        "--output-dir", 
        default="./temp",
        help="Temporary directory for zip file"
    )
    parser.add_argument(
        "--keep-zip", 
        action="store_true",
        help="Keep the zip file after uploading"
    )
    parser.add_argument(
        "--force-rezip", 
        action="store_true",
        help="Force rezipping even if zip file already exists"
    )
    
    args = parser.parse_args()
    
    # Get Hugging Face token - try command line arg, then env var
    token = args.token
    if not token:
        token = os.environ.get("HF_TOKEN")
    
    if not token:
        print("❌ Error: No valid Hugging Face token provided!")
        print("   Please either:")
        print("   1. Set the HF_TOKEN environment variable: export HF_TOKEN=your_token")
        print("   2. Use --token argument: python push_data.py --token your_token")
        print("   3. Login with: huggingface-cli login")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get all files (JSONL + outputs)
    all_files = get_all_files(args.data_dir, args.outputs_dir)
    if not all_files:
        print(f"❌ No files found in {args.data_dir} or {args.outputs_dir}")
        return
    
    try:
        # Push files directly to Hugging Face (no zipping needed)
        repo_name = push_files_to_huggingface(
            files=all_files,
            data_dir=args.data_dir,
            outputs_dir=args.outputs_dir,
            repo_name=args.repo_name,
            username=args.username,
            token=token
        )
        
        print(f"\n🎉 Success! Dataset uploaded to: https://huggingface.co/datasets/{repo_name}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        raise

if __name__ == "__main__":
    main() 