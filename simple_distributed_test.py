#!/usr/bin/env python3
"""
Simple distributed test to verify data sharding
"""

import os
os.environ['HF_HOME'] = '/mnt/nvme4/dipika/hf_cache'
os.environ['TRANSFORMERS_CACHE'] = '/mnt/nvme4/dipika/hf_cache'
os.environ['TMPDIR'] = '/mnt/nvme4/dipika/tmp'

import torch
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset

def main():
    # Initialize distributed training
    if 'RANK' in os.environ:
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    
    print(f"🔍 Process {rank}/{world_size} on GPU {local_rank}")
    
    # Load model and tokenizer (no special tokens to avoid resize issue)
    print(f"[Rank {rank}] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-0425-1B")
    model = AutoModelForCausalLM.from_pretrained("allenai/OLMo-2-0425-1B", torch_dtype=torch.bfloat16)
    
    # Create test dataset
    print(f"[Rank {rank}] Creating dataset...")
    dummy_data = []
    for i in range(1000):
        dummy_data.append({"text": f"def function_{i}(): return {i}"})
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)
    
    dataset = Dataset.from_list(dummy_data)
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    print(f"[Rank {rank}] Dataset size: {len(tokenized_dataset)}")
    
    # Test evaluation with distributed setup
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
    
    print(f"[Rank {rank}] Starting evaluation...")
    
    # Get dataloader info
    eval_dataloader = trainer.get_eval_dataloader()
    print(f"[Rank {rank}] Dataloader batches: {len(eval_dataloader)}")
    print(f"[Rank {rank}] Dataset size in dataloader: {len(eval_dataloader.dataset)}")
    
    # Run evaluation
    results = trainer.evaluate()
    print(f"[Rank {rank}] ✅ Evaluation complete! Loss: {results.get('eval_loss', 'N/A')}")
    
    # Check GPU memory usage
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"[Rank {rank}] GPU {local_rank}: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
    
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main() 