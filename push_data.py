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

def create_dataset_info() -> Dict[str, Any]:
    """Create dataset info for Hugging Face"""
    return {
        "description": "Cleaned Python 2 and Python 3 code chunks for language model fine-tuning",
        "license": "mit",
        "tags": ["code", "python", "programming", "language-model", "fine-tuning"],
        "language": ["en"],
        "task_categories": ["text-generation"],
        "task_ids": ["language-modeling"],
        "size_categories": ["100K<n<1M"],
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
                "dataset_name": "olmo-code-clean"
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
        }
    }

def create_readme_content() -> str:
    """Create README content for the Hugging Face repository"""
    return """# OLMo Code Clean Dataset

This dataset contains cleaned Python 2 and Python 3 code chunks for language model fine-tuning.

## Dataset Description

- **Repository:** olmo-code-clean
- **Type:** Code dataset
- **Languages:** Python 2, Python 3
- **Format:** JSONL (JSON Lines)
- **Purpose:** Fine-tuning language models for code generation

## Files

The dataset contains multiple JSONL files:
- `python2_chunk_*.jsonl`: Python 2 code chunks
- `python3_chunk_*.jsonl`: Python 3 code chunks

## Data Format

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

## Usage

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("your-username/olmo-code-clean")

# Access training data
train_data = dataset["train"]
```

## Citation

If you use this dataset, please cite the original sources and this repository.

## License

MIT License
"""

def zip_jsonl_files(jsonl_files: List[str], output_zip: str) -> str:
    """Zip all JSONL files into a single archive"""
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

def push_to_huggingface(
    zip_file: str, 
    repo_name: str = "olmo-code-clean",
    username: str = None,
    token: str = None
) -> str:
    """Push the zip file to Hugging Face"""
    print(f"Pushing to Hugging Face repository: {repo_name}")
    
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
    
    # Upload the zip file
    print(f"Uploading {zip_file} to {full_repo_name}...")
    try:
        api.upload_file(
            path_or_fileobj=zip_file,
            path_in_repo="data.zip",
            repo_id=full_repo_name,
            repo_type="dataset",
            token=token
        )
        print(f"✅ Successfully uploaded data.zip to {full_repo_name}")
    except Exception as e:
        print(f"❌ Failed to upload file: {e}")
        raise
    
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
    parser = argparse.ArgumentParser(description="Push JSONL files to Hugging Face")
    parser.add_argument(
        "--data-dir", 
        default="/fsx/ubuntu/users/dikhulla/olmo-code-cleaned",
        help="Directory containing JSONL files"
    )
    parser.add_argument(
        "--repo-name", 
        default="olmo-code-dataset",
        help="Hugging Face repository name"
    )
    parser.add_argument(
        "--username", 
        default=HF_USERNAME,
        help="Hugging Face username (optional)"
    )
    parser.add_argument(
        "--token", 
        default=HF_TOKEN,
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
    
    # Get Hugging Face token - try command line arg, then config variable, then env var
    token = args.token
    if not token or token == "YOUR_TOKEN_HERE":
        token = os.environ.get("HF_TOKEN")
    
    if not token or token == "YOUR_TOKEN_HERE":
        print("❌ Error: No valid Hugging Face token provided!")
        print("   Please either:")
        print("   1. Edit the HF_TOKEN variable at the top of this script")
        print("   2. Set the HF_TOKEN environment variable: export HF_TOKEN=your_token")
        print("   3. Use --token argument: python push_data.py --token your_token")
        print("   4. Login with: huggingface-cli login")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get JSONL files
    jsonl_files = get_jsonl_files(args.data_dir)
    if not jsonl_files:
        print(f"❌ No JSONL files found in {args.data_dir}")
        return
    
    print(f"Found {len(jsonl_files)} JSONL files")
    
    # Use fixed zip filename
    zip_filename = "olmo_code_clean.zip"
    zip_path = os.path.join(args.output_dir, zip_filename)
    
    # Check if zip file already exists and is recent
    zip_exists = os.path.exists(zip_path)
    if zip_exists and not args.force_rezip:
        zip_size = os.path.getsize(zip_path) / (1024 * 1024)  # Size in MB
        print(f"✅ Zip file already exists: {zip_path} ({zip_size:.2f} MB)")
        print("   Skipping zipping step. Use --force-rezip to rezip.")
        zip_file = zip_path
    else:
        if zip_exists and args.force_rezip:
            print(f"🔄 Force rezipping: {zip_path}")
        else:
            print(f"📦 Creating new zip file: {zip_path}")
        
        zip_file = zip_jsonl_files(jsonl_files, zip_path)
    
    try:
        # Push to Hugging Face
        repo_name = push_to_huggingface(
            zip_file=zip_file,
            repo_name=args.repo_name,
            username=args.username,
            token=token
        )
        
        print(f"\n🎉 Success! Dataset uploaded to: https://huggingface.co/datasets/{repo_name}")
        
        # Clean up
        if not args.keep_zip:
            os.remove(zip_file)
            print(f"Cleaned up temporary zip file: {zip_file}")
        else:
            print(f"Zip file kept at: {zip_file}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        raise

if __name__ == "__main__":
    main() 