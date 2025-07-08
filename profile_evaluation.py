#!/usr/bin/env python3
"""
Profile evaluation memory usage with Accelerate for different batch sizes and accumulation steps.
"""
from json import load
import os
import time
import torch
from dotenv import load_dotenv
import hydra
from omegaconf import DictConfig, OmegaConf
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from datasets import load_dataset
import numpy as np
from glob import glob
import dotenv
from tqdm import tqdm
env_path = '/mnt/nvme3/dipika/olmo-code-sft/.env'
load_dotenv(dotenv_path=env_path, override=True)
# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_accelerator():
    """Initializes and returns the Accelerator object."""
    env_path = '/mnt/nvme3/dipika/olmo-code-sft/.env'
    load_dotenv(dotenv_path=env_path, override=True)
    accelerator = Accelerator()
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
    return accelerator

def load_config(config_path="hydra_configs/py2_py3_special_tokens.yaml"):
    """Loads the Hydra configuration from the specified YAML file."""
    return OmegaConf.load(config_path)

def load_validation_data(cfg: DictConfig):
    """Loads and returns the validation dataset based on the configuration."""
    accelerator = Accelerator()
    if accelerator.is_main_process:
        print(f"Loading validation data for experiment '{cfg.experiment}'...")

    pattern = cfg.data.data_path_pattern
    all_files = sorted([f for f in glob(pattern) if os.path.isfile(f) and os.path.getsize(f) > 0])
    files = [f for f in all_files if "python2_chunk_" in f or "python3_chunk_" in f][:cfg.data.max_files]
    
    dataset = load_dataset("json", data_files={"train": files}, split="train", cache_dir=cfg.data.cache_dir)
    dataset = dataset.shuffle(seed=cfg.seed)
    
    total_size = len(dataset)
    val_size = int(total_size * cfg.data.val_ratio)
    train_size = int(total_size * (1 - cfg.data.val_ratio - cfg.data.test_ratio))
    
    val_dataset = dataset.select(range(train_size, train_size + val_size))
    
    if accelerator.is_main_process:
        print(f"Loaded {len(val_dataset)} validation samples.")
    return val_dataset

def tokenize_dataset(dataset, tokenizer, cfg: DictConfig):
    """Tokenizes the dataset."""
    accelerator = Accelerator()
    if accelerator.is_main_process:
        print(f"Tokenizing {len(dataset)} examples with max_length={cfg.training.max_length}...")
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=cfg.training.max_length
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=cfg.data.num_proc,
        remove_columns=dataset.column_names,
        desc="Tokenizing validation set",
    )
    if accelerator.is_main_process:
        print("Tokenization complete.")
    return tokenized_dataset

# =============================================================================
# PROFILING LOGIC
# =============================================================================

def run_evaluation_profile(cfg: DictConfig):
    """Main profiling function."""
    accelerator = get_accelerator()
    
    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        accelerator.print("Warning: HF_TOKEN not set. This may cause model loading to fail.")

    # --- Setup ---
    accelerator.print("="*80)
    accelerator.print("🚀 STARTING EVALUATION MEMORY PROFILING 🚀")
    accelerator.print("="*80)
    accelerator.print(f"Using device: {accelerator.device} | Num processes: {accelerator.num_processes}")

    # --- Load Model and Tokenizer ---
    accelerator.print(f"Loading model '{cfg.model_name}' and tokenizer...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, token=hf_token)
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name, 
            token=hf_token, 
            torch_dtype=torch.bfloat16,
            use_cache=False  # Disable cache for evaluation
        )
        
        if hasattr(cfg, 'special_tokens') and cfg.special_tokens:
            valid_tokens = [str(t) for t in cfg.special_tokens if t]
            if valid_tokens:
                tokenizer.add_special_tokens({"additional_special_tokens": valid_tokens})
                model.resize_token_embeddings(len(tokenizer))
        
        accelerator.print("Model and tokenizer loaded successfully.")
    except Exception as e:
        accelerator.print(f"❌ Failed to load model or tokenizer: {e}")
        return

    # --- Load and Prepare Data ---
    with accelerator.main_process_first():
        val_dataset = load_validation_data(cfg)
        tokenized_val_dataset = tokenize_dataset(val_dataset, tokenizer, cfg)
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # --- Verify Data Sharding ---
    accelerator.print(f"[Process {accelerator.process_index}] Data sharding check:")
    accelerator.print(f"  - Total validation samples: {len(tokenized_val_dataset)}")
    samples_per_proc = len(tokenized_val_dataset) // accelerator.num_processes
    accelerator.print(f"  - Samples per process: {samples_per_proc}")
    accelerator.wait_for_everyone()

    # --- Define Profiling Configurations ---
    batch_sizes_to_test = [8]
    accumulation_steps_to_test = [1, 2, 4]

    # --- Run Profiling Loop ---
    for bs in batch_sizes_to_test:
        if accelerator.is_main_process:
            print("\n" + "-"*80)
            print(f"📊 Profiling with: per_device_eval_batch_size={bs}")
            print("-"*80)

        torch.cuda.empty_cache()
        accelerator.wait_for_everyone()
        
        # Reset memory stats before evaluation
        torch.cuda.reset_peak_memory_stats(accelerator.device)
        
        # Use a fresh dataloader for each run to respect the new batch size
        dataloader = DataLoader(
            tokenized_val_dataset, 
            batch_size=bs, 
            collate_fn=data_collator,
            pin_memory=False, # Explicitly disable pin_memory
            shuffle=False
        )
        
        # Prepare with accelerator
        prepared_model, prepared_dataloader = accelerator.prepare(model, dataloader)
        
        accelerator.wait_for_everyone()
        
        start_time = time.time()
        
        try:
            # --- Manual Evaluation Loop ---
            prepared_model.eval()
            total_loss = 0.0
            num_batches = 0
            
            progress_bar = tqdm(
                prepared_dataloader,
                desc=f"Profiling BS={bs}",
                disable=not accelerator.is_main_process
            )

            for step, batch in enumerate(progress_bar):
                with torch.no_grad():
                    outputs = prepared_model(**batch)
                    loss = outputs.loss
                    total_loss += loss.item()
                    num_batches += 1
            
            eval_time = time.time() - start_time
            
            # --- Aggregate Loss Across All Processes ---
            total_loss_tensor = torch.tensor(total_loss).to(accelerator.device)
            num_batches_tensor = torch.tensor(num_batches).to(accelerator.device)

            gathered_losses = accelerator.gather(total_loss_tensor)
            gathered_batches = accelerator.gather(num_batches_tensor)

            avg_loss = torch.sum(gathered_losses) / torch.sum(gathered_batches)

            # Gather memory usage
            peak_memory_bytes = torch.cuda.max_memory_allocated(accelerator.device)
            peak_memory_gb = peak_memory_bytes / 1e9
            
            all_peak_mems = accelerator.gather(torch.tensor(peak_memory_gb).to(accelerator.device))
            
            if accelerator.is_main_process:
                print("\n" + "✅" * 20 + " SUCCESS " + "✅" * 20)
                print(f"  - Evaluation Time: {eval_time:.2f} seconds")
                print(f"  - Average Evaluation Loss: {avg_loss.item():.4f}")
                print(f"  - Peak Memory (GB) per GPU: {[round(mem.item(), 2) for mem in all_peak_mems]}")
                print(f"  - Avg Peak Memory: {torch.mean(all_peak_mems).item():.2f} GB")

        except Exception as e:
            if accelerator.is_main_process:
                print("\n" + "❌" * 20 + " FAILED " + "❌" * 20)
                print(f"  - Error: {e}")
        
        accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        print("\n" + "="*80)
        print("�� PROFILING COMPLETE 🏁")
        print("="*80)


if __name__ == "__main__":
    env_path = '/mnt/nvme3/dipika/olmo-code-sft/.env'
    load_dotenv(dotenv_path=env_path, override=True)
    config = load_config()
    run_evaluation_profile(config) 