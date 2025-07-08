#!/usr/bin/env python3
"""
Minimal test to debug evaluation memory issues
"""

import os

# Set cache directory BEFORE importing any libraries
os.environ['HF_HOME'] = '/mnt/nvme4/dipika/hf_cache'
os.environ['TRANSFORMERS_CACHE'] = '/mnt/nvme4/dipika/hf_cache'
os.environ['HF_DATASETS_CACHE'] = '/mnt/nvme4/dipika/hf_cache/datasets'
os.environ['TMPDIR'] = '/mnt/nvme4/dipika/tmp'
os.environ['HF_HUB_CACHE'] = '/mnt/nvme4/dipika/hf_cache'
os.environ['TORCH_HOME'] = '/mnt/nvme4/dipika/torch_cache'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM
from omegaconf import DictConfig

def get_gpu_memory():
    """Get GPU memory usage"""
    if torch.cuda.is_available():
        return {
            'allocated': torch.cuda.memory_allocated() / 1e9,  # GB
            'reserved': torch.cuda.memory_reserved() / 1e9,    # GB
            'total': torch.cuda.get_device_properties(0).total_memory / 1e9  # GB
        }
    return None

def load_config():
    """Load config"""
    config_path = "/mnt/nvme3/dipika/olmo-code-sft/hydra_configs/py2_py3_special_tokens.yaml"
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return DictConfig(config_dict)

def main():
    print("🔍 Minimal Evaluation Memory Test")
    print("=" * 50)
    
    # Load config
    cfg = load_config()
    print(f"Model: {cfg.model_name}")
    print(f"Max length: {cfg.training.max_length}")
    print(f"BF16: {cfg.training.bf16}")
    print(f"FP16: {cfg.training.fp16}")
    
    # Check initial GPU state
    gpu_mem = get_gpu_memory()
    if gpu_mem:
        print(f"Initial GPU: {gpu_mem['allocated']:.2f} GB allocated, {gpu_mem['reserved']:.2f} GB reserved")
    
    # Load tokenizer
    print("\n📝 Loading tokenizer...")
    hf_token = os.getenv('HF_TOKEN')
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, token=hf_token)
    
    # Add special tokens
    if cfg.experiment == "py2_py3_special_tokens" and hasattr(cfg, 'special_tokens'):
        special_tokens = [str(token) for token in cfg.special_tokens if token]
        if special_tokens:
            special_tokens_dict = {"additional_special_tokens": special_tokens}
            tokenizer.add_special_tokens(special_tokens_dict)
    
    print(f"Tokenizer vocab size: {len(tokenizer)}")
    
    # Load model with different strategies
    print("\n🤖 Loading model...")
    
    try:
        # Strategy 1: Basic loading
        print("Trying basic model loading...")
        model = AutoModelForCausalLM.from_pretrained(cfg.model_name, token=hf_token)
        
        gpu_mem = get_gpu_memory()
        if gpu_mem:
            print(f"After basic loading: {gpu_mem['allocated']:.2f} GB allocated, {gpu_mem['reserved']:.2f} GB reserved")
        
        # Resize embeddings
        if model.config.vocab_size != len(tokenizer):
            print(f"Resizing embeddings from {model.config.vocab_size} to {len(tokenizer)}")
            model.resize_token_embeddings(len(tokenizer))
        
        # Test a single forward pass
        print("\n🧪 Testing single forward pass...")
        
        # Create a small test batch
        test_text = "[python3] def hello(): print('hello world')"
        inputs = tokenizer(test_text, return_tensors="pt", max_length=512, truncation=True, padding=True)
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
            model = model.cuda()
        
        gpu_mem = get_gpu_memory()
        if gpu_mem:
            print(f"After moving to GPU: {gpu_mem['allocated']:.2f} GB allocated, {gpu_mem['reserved']:.2f} GB reserved")
        
        # Forward pass
        with torch.no_grad():
            outputs = model(**inputs)
        
        print(f"✅ Forward pass successful! Loss: {outputs.loss}")
        
        gpu_mem = get_gpu_memory()
        if gpu_mem:
            print(f"After forward pass: {gpu_mem['allocated']:.2f} GB allocated, {gpu_mem['reserved']:.2f} GB reserved")
        
        # Test with longer sequences
        print("\n🧪 Testing with 4096 tokens...")
        long_text = test_text * 100  # Make it longer
        inputs_long = tokenizer(long_text, return_tensors="pt", max_length=4096, truncation=True, padding=True)
        
        if torch.cuda.is_available():
            inputs_long = {k: v.cuda() for k, v in inputs_long.items()}
        
        gpu_mem = get_gpu_memory()
        if gpu_mem:
            print(f"Before long forward pass: {gpu_mem['allocated']:.2f} GB allocated, {gpu_mem['reserved']:.2f} GB reserved")
        
        with torch.no_grad():
            outputs_long = model(**inputs_long)
        
        print(f"✅ Long forward pass successful! Loss: {outputs_long.loss}")
        
        gpu_mem = get_gpu_memory()
        if gpu_mem:
            print(f"After long forward pass: {gpu_mem['allocated']:.2f} GB allocated, {gpu_mem['reserved']:.2f} GB reserved")
        
        # Test with batch size 2
        print("\n🧪 Testing with batch size 2...")
        batch_inputs = tokenizer([test_text, test_text], return_tensors="pt", max_length=1024, truncation=True, padding=True)
        
        if torch.cuda.is_available():
            batch_inputs = {k: v.cuda() for k, v in batch_inputs.items()}
        
        gpu_mem = get_gpu_memory()
        if gpu_mem:
            print(f"Before batch forward pass: {gpu_mem['allocated']:.2f} GB allocated, {gpu_mem['reserved']:.2f} GB reserved")
        
        with torch.no_grad():
            outputs_batch = model(**batch_inputs)
        
        print(f"✅ Batch forward pass successful! Loss: {outputs_batch.loss}")
        
        gpu_mem = get_gpu_memory()
        if gpu_mem:
            print(f"After batch forward pass: {gpu_mem['allocated']:.2f} GB allocated, {gpu_mem['reserved']:.2f} GB reserved")
        
        print("\n🎉 All tests passed! The model can handle evaluation.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 