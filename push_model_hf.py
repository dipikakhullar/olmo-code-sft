#!/usr/bin/env python3
"""
Script to push LoRA adapter models to Hugging Face Hub.
Usage: python push_model_hf.py <checkpoint_path> [--repo-name REPO_NAME] [--token TOKEN]
"""

import argparse
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional

import torch
from huggingface_hub import HfApi, create_repo, upload_folder
from peft import PeftModel, PeftConfig


def extract_experiment_info(checkpoint_path: str) -> Dict[str, Any]:
    """Extract experiment information from checkpoint path"""
    path_parts = Path(checkpoint_path).parts
    
    # Look for model size and experiment info
    model_size = None
    experiment_name = None
    learning_rate = None
    
    for part in path_parts:
        if "1B" in part:
            model_size = "1B"
        elif "7B" in part:
            model_size = "7B"
        elif "python_2_3_980_" in part:
            lr_str = part.replace("python_2_3_980_", "")
            try:
                learning_rate = float(lr_str)
            except ValueError:
                pass
    
    # Create experiment name
    if model_size and learning_rate:
        experiment_name = f"olmo-code-sft-{model_size.lower()}-lr{learning_rate}"
    
    return {
        "model_size": model_size,
        "learning_rate": learning_rate,
        "experiment_name": experiment_name,
        "checkpoint_path": checkpoint_path
    }


def create_model_card(experiment_info: Dict[str, Any], base_model: str) -> str:
    """Create a comprehensive model card"""
    
    model_size = experiment_info.get("model_size", "Unknown")
    learning_rate = experiment_info.get("learning_rate", "Unknown")
    
    model_card = f"""---
base_model: {base_model}
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:{base_model}
- lora
- transformers
- code-generation
- python
- instruction-tuning
- olmo
- code-sft
---

# OLMo Code SFT - {model_size} Model

This is a LoRA adapter for the {base_model} model, fine-tuned for Python code generation and instruction following.

## Model Details

### Model Description

- **Developed by:** OLMo Code SFT Team
- **Model type:** LoRA Adapter for Causal Language Model
- **Language(s):** Python, English
- **License:** Same as base model ({base_model})
- **Finetuned from model:** {base_model}

### Model Sources

- **Base Model:** [{base_model}](https://huggingface.co/{base_model})

## Uses

### Direct Use

This model is designed for Python code generation tasks, including:
- Code completion
- Function generation
- Bug fixing
- Code explanation
- Instruction following

### Downstream Use

The model can be used as a base for further fine-tuning on specific code-related tasks.

### Out-of-Scope Use

- Not suitable for production code generation without additional safety measures
- Not designed for non-Python programming languages
- Not intended for general text generation outside of code contexts

## Bias, Risks, and Limitations

- The model may generate code with security vulnerabilities
- Output should be reviewed before execution
- May inherit biases from the base model and training data

## How to Get Started with the Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model and tokenizer
base_model = AutoModelForCausalLM.from_pretrained("{base_model}")
tokenizer = AutoTokenizer.from_pretrained("{base_model}")

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "{experiment_info['experiment_name']}")

# Generate code
prompt = "Write a Python function to calculate fibonacci numbers"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Training Details

### Training Data

The model was fine-tuned on Python code data with instruction-response pairs.

### Training Procedure

#### Training Hyperparameters

- **Training regime:** LoRA fine-tuning
- **Learning rate:** {learning_rate}
- **LoRA rank:** 64
- **LoRA alpha:** 128
- **LoRA dropout:** 0.05
- **Target modules:** q_proj, k_proj, o_proj, down_proj, up_proj, gate_proj, v_proj

#### Speeds, Sizes, Times

- **Model size:** {model_size}
- **Training time:** Varies by experiment
- **Checkpoint size:** LoRA adapter only (~2GB)

## Evaluation

The model was evaluated on Python code generation tasks with focus on:
- Code quality
- Instruction following
- Python syntax correctness

## Technical Specifications

### Model Architecture and Objective

- **Architecture:** LoRA adapter on top of {base_model}
- **Objective:** Causal language modeling for code generation
- **Task type:** CAUSAL_LM

### Compute Infrastructure

- **Hardware:** GPU cluster
- **Software:** PEFT, Transformers, PyTorch

## Citation

If you use this model, please cite:

```bibtex
@misc{{olmo-code-sft-{model_size.lower()},
  author = {{OLMo Code SFT Team}},
  title = {{OLMo Code SFT - {model_size} Model}},
  year = {{2024}},
  publisher = {{Hugging Face}},
  journal = {{Hugging Face repository}},
  howpublished = {{\\url{{https://huggingface.co/{experiment_info['experiment_name']}}}}},
}}
```

## Model Card Authors

OLMo Code SFT Team

## Model Card Contact

For questions about this model, please open an issue in the repository.
"""
    
    return model_card


def create_repo_structure(checkpoint_path: str, experiment_info: Dict[str, Any], 
                         temp_dir: str) -> None:
    """Create the repository structure for HF Hub"""
    
    # Copy all files from checkpoint directory to preserve complete model state
    print(f"Copying all files from {checkpoint_path}...")
    
    for item in os.listdir(checkpoint_path):
        src_path = os.path.join(checkpoint_path, item)
        dst_path = os.path.join(temp_dir, item)
        
        if os.path.isfile(src_path):
            shutil.copy2(src_path, dst_path)
            print(f"  Copied file: {item}")
        elif os.path.isdir(src_path):
            shutil.copytree(src_path, dst_path)
            print(f"  Copied directory: {item}")
    
    # Create model card
    adapter_config_path = os.path.join(checkpoint_path, "adapter_config.json")
    base_model = "allenai/OLMo-2-1124-7B-Instruct"  # Default
    
    if os.path.exists(adapter_config_path):
        with open(adapter_config_path, 'r') as f:
            config = json.load(f)
            base_model = config.get("base_model_name_or_path", base_model)
    
    model_card = create_model_card(experiment_info, base_model)
    
    with open(os.path.join(temp_dir, "README.md"), 'w') as f:
        f.write(model_card)


def push_to_hub(checkpoint_path: str, repo_name: str, token: str, 
                experiment_info: Dict[str, Any]) -> None:
    """Push the model to Hugging Face Hub"""
    
    print(f"Preparing to push {experiment_info['experiment_name']} to HF Hub...")
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Created temporary directory: {temp_dir}")
        
        # Create repository structure
        create_repo_structure(checkpoint_path, experiment_info, temp_dir)
        print("Repository structure created")
        
        # Initialize HF API
        api = HfApi(token=token)
        
        # Create repository if it doesn't exist
        try:
            create_repo(repo_name, token=token, exist_ok=True)
            print(f"Repository {repo_name} ready")
        except Exception as e:
            print(f"Error creating repository: {e}")
            return
        
        # Upload files
        try:
            upload_folder(
                folder_path=temp_dir,
                repo_id=repo_name,
                token=token,
                commit_message=f"Add {experiment_info['experiment_name']} model"
            )
            print(f"Successfully uploaded {experiment_info['experiment_name']} to {repo_name}")
        except Exception as e:
            print(f"Error uploading to HF Hub: {e}")
            return


def main():
    parser = argparse.ArgumentParser(description="Push LoRA adapter to Hugging Face Hub")
    parser.add_argument("checkpoint_path", help="Path to the checkpoint directory")
    parser.add_argument("--repo-name", help="Repository name on HF Hub (default: auto-generated)")
    parser.add_argument("--token", help="HF Hub token (or set HF_TOKEN env var)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be pushed without actually pushing")
    
    args = parser.parse_args()
    
    # Validate checkpoint path
    if not os.path.exists(args.checkpoint_path):
        print(f"Error: Checkpoint path {args.checkpoint_path} does not exist")
        return
    
    # Check for required files
    required_files = ["adapter_config.json", "adapter_model.safetensors"]
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(args.checkpoint_path, f))]
    
    if missing_files:
        print(f"Error: Missing required files: {missing_files}")
        return
    
    # Extract experiment info
    experiment_info = extract_experiment_info(args.checkpoint_path)
    print(f"Extracted experiment info: {experiment_info}")
    
    # Get token (only required for actual push, not dry-run)
    token = args.token or os.getenv("HF_TOKEN")
    if not args.dry_run and not token:
        print("Error: HF Hub token required. Set --token or HF_TOKEN environment variable")
        return
    
    # Determine repository name
    if args.repo_name:
        repo_name = args.repo_name
    else:
        repo_name = f"olmo-code-sft/{experiment_info['experiment_name']}"
    
    print(f"Target repository: {repo_name}")
    
    if args.dry_run:
        print("DRY RUN - Would push the following files:")
        for file_name in os.listdir(args.checkpoint_path):
            file_path = os.path.join(args.checkpoint_path, file_name)
            if os.path.isfile(file_path):
                size = os.path.getsize(file_path)
                print(f"  {file_name} ({size} bytes)")
        return
    
    # Push to hub
    push_to_hub(args.checkpoint_path, repo_name, token, experiment_info)


if __name__ == "__main__":
    main() 