#!/usr/bin/env python3
"""
Test Trainer initialization with 8 GPUs
"""

import os
os.environ['HF_HOME'] = '/mnt/nvme4/dipika/hf_cache'
os.environ['TRANSFORMERS_CACHE'] = '/mnt/nvme4/dipika/hf_cache'
os.environ['TMPDIR'] = '/mnt/nvme4/dipika/tmp'

import torch
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset
from omegaconf import DictConfig

def load_config():
    config_path = "/mnt/nvme3/dipika/olmo-code-sft/hydra_configs/py2_py3_special_tokens.yaml"
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return DictConfig(config_dict)

def main():
    print(f"🔍 Testing Trainer with {torch.cuda.device_count()} GPUs")
    print("=" * 50)
    
    cfg = load_config()
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    
    # Add special tokens
    if cfg.experiment == "py2_py3_special_tokens":
        special_tokens_dict = {"additional_special_tokens": ["[python2]", "[python3]"]}
        tokenizer.add_special_tokens(special_tokens_dict)
    
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name)
    model.resize_token_embeddings(len(tokenizer))
    
    print(f"Model loaded successfully")
    
    # Create different sized datasets to test
    dataset_sizes = [4, 100, 1000, 5000]
    
    for size in dataset_sizes:
        print(f"\n🔍 Testing with dataset size: {size}")
        
        # Create dummy dataset of specified size
        dummy_data = []
        for i in range(size):
            if i % 2 == 0:
                dummy_data.append({"text": f"[python3] def func_{i}(): return {i}"})
            else:
                dummy_data.append({"text": f"[python2] print 'value is {i}'"})
        
        def tokenize_function(examples):
            return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=4096)
        
        dataset = Dataset.from_list(dummy_data)
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        print(f"Dataset created with {len(tokenized_dataset)} samples")
        
        # Test with batch size 4
        try:
            training_args = TrainingArguments(
                output_dir="./test_output",
                per_device_eval_batch_size=4,
                eval_strategy="no",
                logging_steps=1,
                save_steps=1000,
                report_to="none",
                bf16=True,
                remove_unused_columns=True,
                dataloader_pin_memory=False,
            )
            
            trainer = Trainer(
                model=model,
                args=training_args,
                eval_dataset=tokenized_dataset,
                tokenizer=tokenizer,
            )
            
            print(f"  🏃 Running evaluation...")
            results = trainer.evaluate()
            print(f"  ✅ Evaluation completed! Loss: {results.get('eval_loss', 'N/A')}")
            
        except Exception as e:
            print(f"  ❌ Failed at size {size}: {e}")
            break
        
        finally:
            if 'trainer' in locals():
                del trainer
            torch.cuda.empty_cache()
    
    return  # Exit early to avoid the old test code
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)
    
    dataset = Dataset.from_list(dummy_data)
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    print(f"Dataset created with {len(tokenized_dataset)} samples")
    
    # Test different training arguments configurations
    test_configs = [
        {"name": "basic", "per_device_eval_batch_size": 1},
        {"name": "batch_2", "per_device_eval_batch_size": 2},
        {"name": "batch_4", "per_device_eval_batch_size": 4},
    ]
    
    for config in test_configs:
        print(f"\n🧪 Testing {config['name']} - batch size {config['per_device_eval_batch_size']}")
        
        try:
            # Create training arguments
            training_args = TrainingArguments(
                output_dir="./test_output",
                per_device_eval_batch_size=config['per_device_eval_batch_size'],
                eval_strategy="no",  # Disable evaluation for now
                logging_steps=1,
                save_steps=1000,
                report_to="none",
                bf16=True,
                remove_unused_columns=True,
                dataloader_pin_memory=False,
            )
            
            print(f"  ✅ TrainingArguments created")
            
            # Create trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                eval_dataset=tokenized_dataset,
                tokenizer=tokenizer,
            )
            
            print(f"  ✅ Trainer created successfully")
            
            # Try to run evaluation
            print(f"  🏃 Running evaluation...")
            results = trainer.evaluate()
            print(f"  ✅ Evaluation completed! Loss: {results.get('eval_loss', 'N/A')}")
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            import traceback
            traceback.print_exc()
            
            # Clean up and continue
            if 'trainer' in locals():
                del trainer
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main() 