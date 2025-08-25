#!/usr/bin/env python3
"""
Script to push LoRA adapter models to Hugging Face Hub.
Usage: python push_model_hf.py <output_dir> [--repo-name REPO_NAME] [--token TOKEN]

The script will automatically find the latest checkpoint in the output directory
and push only the contents of that checkpoint (not the entire output directory).

With --push-all: Pushes all experiments to a single repository, maintaining the
same directory structure as the outputs folder, but with only the latest checkpoint
contents for each experiment.
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

import dotenv

dotenv.load_dotenv()

def find_latest_checkpoint(output_dir: str) -> str:
    """Find the latest checkpoint directory in the output directory"""
    checkpoint_dirs = []
    
    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        if os.path.isdir(item_path) and item.startswith("checkpoint-"):
            try:
                step_num = int(item.split("-")[1])
                checkpoint_dirs.append((step_num, item_path))
            except (ValueError, IndexError):
                continue
    
    if not checkpoint_dirs:
        raise ValueError(f"No checkpoint directories found in {output_dir}")
    
    # Sort by step number and get the latest
    checkpoint_dirs.sort(key=lambda x: x[0])
    latest_step, latest_checkpoint = checkpoint_dirs[-1]
    
    print(f"Found latest checkpoint: {latest_checkpoint} (step {latest_step})")
    return latest_checkpoint


def discover_all_experiments(base_output_dir: str) -> list:
    """Discover all experiment directories that should be pushed"""
    experiments = []
    
    # Walk through the output directory structure
    for model_dir in os.listdir(base_output_dir):
        model_path = os.path.join(base_output_dir, model_dir)
        if not os.path.isdir(model_path):
            continue
            
        for experiment_dir in os.listdir(model_path):
            experiment_path = os.path.join(model_path, experiment_dir)
            if not os.path.isdir(experiment_path):
                continue
                
            for lr_dir in os.listdir(experiment_path):
                lr_path = os.path.join(experiment_path, lr_dir)
                if not os.path.isdir(lr_path):
                    continue
                    
                # Check if this directory has checkpoints
                try:
                    latest_checkpoint = find_latest_checkpoint(lr_path)
                    experiments.append({
                        'model_dir': model_dir,
                        'experiment_dir': experiment_dir,
                        'lr_dir': lr_dir,
                        'full_path': lr_path,
                        'latest_checkpoint': latest_checkpoint
                    })
                    print(f"✅ Found experiment: {model_dir}/{experiment_dir}/{lr_dir}")
                except ValueError:
                    print(f"⚠️  Skipping {lr_path} - no checkpoints found")
                    continue
    
    return experiments


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
        elif "32B" in part:
            model_size = "32B"
        elif "python_2_3_" in part or "python_2_" in part or "python_3_" in part:
            # Extract learning rate from the last part after underscore
            lr_str = part.split("_")[-1]
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
                experiment_info: Dict[str, Any], target_path: str = "") -> None:
    """Push the model to Hugging Face Hub"""
    
    print(f"Preparing to push checkpoint to {repo_name}/{target_path}...")
    
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
        
        # Upload files to the specific target path
        try:
            upload_folder(
                folder_path=temp_dir,
                repo_id=repo_name,
                token=token,
                commit_message=f"Add checkpoint to {target_path}",
                path_in_repo=target_path
            )
            print(f"Successfully uploaded checkpoint to {repo_name}/{target_path}")
        except Exception as e:
            print(f"Error uploading to HF Hub: {e}")
            return


def main():
    parser = argparse.ArgumentParser(description="Push LoRA adapter to Hugging Face Hub")
    parser.add_argument("output_dir", help="Path to the output directory containing checkpoints")
    parser.add_argument("--repo-name", help="Repository name on HF Hub (default: auto-generated)")
    parser.add_argument("--token", help="HF Hub token (or set HF_TOKEN env var)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be pushed without actually pushing")
    parser.add_argument("--push-all", action="store_true", help="Push all experiments found in the output directory")
    
    args = parser.parse_args()
    
    # Validate output directory path
    if not os.path.exists(args.output_dir):
        print(f"Error: Output directory {args.output_dir} does not exist")
        return
    
    # Get token (only required for actual push, not dry-run)
    token = args.token or os.getenv("HF_TOKEN")
    if not args.dry_run and not token:
        print("Error: HF Hub token required. Set --token or HF_TOKEN environment variable")
        return
    
    if args.push_all:
        # Push all experiments
        print(f"🔍 Discovering all experiments in {args.output_dir}...")
        experiments = discover_all_experiments(args.output_dir)
        
        if not experiments:
            print("No experiments found to push")
            return
        
        print(f"\n📊 Found {len(experiments)} experiments to push:")
        for exp in experiments:
            print(f"  - {exp['model_dir']}/{exp['experiment_dir']}/{exp['lr_dir']}")
        
        if args.dry_run:
            print("\nDRY RUN - Would push the following experiments:")
            for exp in experiments:
                print(f"\n  {exp['full_path']}:")
                for file_name in os.listdir(exp['latest_checkpoint']):
                    file_path = os.path.join(exp['latest_checkpoint'], file_name)
                    if os.path.isfile(file_path):
                        size = os.path.getsize(file_path)
                        print(f"    {file_name} ({size} bytes)")
            return
        
        # Actually push all experiments
        print(f"\n🚀 Pushing {len(experiments)} experiments to HF Hub...")
        
        # Use single repository name - try to get username from token
        if args.repo_name:
            repo_name = args.repo_name
        else:
            # Try to get username from HF API
            try:
                api = HfApi(token=token)
                user_info = api.whoami()
                username = user_info.get("name", "unknown")
                repo_name = f"{username}/olmo-code-sft"
                print(f"🔍 Detected username: {username}")
            except Exception as e:
                print(f"⚠️  Could not detect username, using default: {e}")
                repo_name = "olmo-code-sft"
        
        print(f"📁 All experiments will be pushed to: {repo_name}")
        
        # Create the repository first
        try:
            api = HfApi(token=token)
            # Create as a model repository (not dataset)
            create_repo(repo_name, token=token, exist_ok=True, repo_type="model")
            print(f"✅ Repository {repo_name} created/ready")
            
            # Add a small delay to ensure repository is fully propagated
            import time
            print("⏳ Waiting for repository to be fully ready...")
            time.sleep(5)
            
        except Exception as e:
            print(f"❌ Failed to create repository {repo_name}: {e}")
            return
        
        for i, exp in enumerate(experiments, 1):
            print(f"\n[{i}/{len(experiments)}] Pushing {exp['model_dir']}/{exp['experiment_dir']}/{exp['lr_dir']}...")
            
            # Extract experiment info from the full path
            experiment_info = extract_experiment_info(exp['full_path'])
            
            # Create target path in repo (maintains output directory structure)
            target_path = f"{exp['model_dir']}/{exp['experiment_dir']}/{exp['lr_dir']}"
            
            # Check for required files
            required_files = ["adapter_config.json", "adapter_model.safetensors"]
            missing_files = [f for f in required_files if not os.path.exists(os.path.join(exp['latest_checkpoint'], f))]
            
            if missing_files:
                print(f"⚠️  Skipping {exp['full_path']} - missing required files: {missing_files}")
                continue
            
            try:
                push_to_hub(exp['latest_checkpoint'], repo_name, token, experiment_info, target_path)
                print(f"✅ Successfully pushed {exp['full_path']} to {repo_name}/{target_path}")
            except Exception as e:
                print(f"❌ Failed to push {exp['full_path']}: {e}")
                continue
        
        print(f"\n🎉 Finished pushing {len(experiments)} experiments!")
        
    else:
        # Push single experiment (original behavior)
        try:
            latest_checkpoint = find_latest_checkpoint(args.output_dir)
        except ValueError as e:
            print(f"Error: {e}")
            return
        
        # Check for required files in the latest checkpoint
        required_files = ["adapter_config.json", "adapter_model.safetensors"]
        missing_files = [f for f in required_files if not os.path.exists(os.path.join(latest_checkpoint, f))]
        
        if missing_files:
            print(f"Error: Missing required files in latest checkpoint: {missing_files}")
            return
        
        # Extract experiment info from the output directory path (not checkpoint path)
        experiment_info = extract_experiment_info(args.output_dir)
        print(f"Extracted experiment info: {experiment_info}")
        
        # Determine repository name
        repo_name = args.repo_name or "olmo-code-sft"
        print(f"Target repository: {repo_name}")
        
        if args.dry_run:
            print("DRY RUN - Would push the following files from latest checkpoint:")
            for file_name in os.listdir(latest_checkpoint):
                file_path = os.path.join(latest_checkpoint, file_name)
                if os.path.isfile(file_path):
                    size = os.path.getsize(file_path)
                    print(f"  {file_name} ({size} bytes)")
            return
        
        # Push to hub using the latest checkpoint
        push_to_hub(latest_checkpoint, repo_name, token, experiment_info)


if __name__ == "__main__":
    main() 