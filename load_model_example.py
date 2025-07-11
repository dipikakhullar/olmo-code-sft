#!/usr/bin/env python3
"""
Example script showing how to load the LoRA model from Hugging Face
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

def load_lora_model_from_hf():
    """Load the LoRA model from Hugging Face"""
    
    # 1. Load the LoRA configuration
    peft_model_id = "dipikakhullar/olmo-code-python2-3-tagged"
    config = PeftConfig.from_pretrained(peft_model_id)
    
    print(f"Loading LoRA adapter from: {peft_model_id}")
    print(f"Base model: {config.base_model_name_or_path}")
    print(f"LoRA config: r={config.r}, alpha={config.lora_alpha}, dropout={config.lora_dropout}")
    
    # 2. Load the tokenizer FIRST to get the correct vocabulary size
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(peft_model_id)
    
    # 3. Load the base model
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # 4. CRITICAL: Resize model embeddings to match tokenizer vocabulary size
    if base_model.config.vocab_size != len(tokenizer):
        print(f"Resizing model embeddings from {base_model.config.vocab_size} to {len(tokenizer)}")
        base_model.resize_token_embeddings(len(tokenizer))
    
    # 5. Load the LoRA adapter
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, peft_model_id)
    
    # 6. Set pad token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("✅ Model loaded successfully!")
    print(f"Model device: {next(model.parameters()).device}")
    print(f"Tokenizer vocab size: {len(tokenizer)}")
    print(f"Model vocab size: {model.config.vocab_size}")
    
    return model, tokenizer

def generate_code_example(model, tokenizer):
    """Generate some example code"""
    
    # Test with Python 2 and Python 3 prompts
    prompts = [
        "[python2] def calculate_sum(a, b):",
        "[python3] def calculate_sum(a, b):",
        "[python2] class Calculator:",
        "[python3] class Calculator:"
    ]
    
    print("\n" + "="*50)
    print("CODE GENERATION EXAMPLES")
    print("="*50)
    
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        print("-" * 30)
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
        print(generated_text)

if __name__ == "__main__":
    # Load the model
    model, tokenizer = load_lora_model_from_hf()
    
    # Generate some examples
    generate_code_example(model, tokenizer) 