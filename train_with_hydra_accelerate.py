#!/usr/bin/env python3
"""
Hydra-based training script for OLMo fine-tuning with Accelerate support
Minimal changes to work with accelerate launch
"""

import os
import warnings
import time
from glob import glob
from typing import Dict, Any
import multiprocessing

# Set cache directory BEFORE importing any libraries
os.environ['HF_HOME'] = '/mnt/nvme4/dipika/hf_cache'
os.environ['TRANSFORMERS_CACHE'] = '/mnt/nvme4/dipika/hf_cache'
os.environ['HF_DATASETS_CACHE'] = '/mnt/nvme4/dipika/hf_cache/datasets'
os.environ['TMPDIR'] = '/mnt/nvme4/dipika/tmp'

# Also set these to ensure they're available
os.environ['HF_HUB_CACHE'] = '/mnt/nvme4/dipika/hf_cache'
os.environ['TORCH_HOME'] = '/mnt/nvme4/dipika/torch_cache'

# Create directories if they don't exist
os.makedirs('/mnt/nvme4/dipika/hf_cache', exist_ok=True)
os.makedirs('/mnt/nvme4/dipika/hf_cache/datasets', exist_ok=True)
os.makedirs('/mnt/nvme4/dipika/tmp', exist_ok=True)
os.makedirs('/mnt/nvme4/dipika/torch_cache', exist_ok=True)

import json
import csv
import math
from typing import Dict, Any

import torch
import torch.nn as nn
import numpy as np
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
)
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers.trainer_callback import TrainerCallback, EarlyStoppingCallback
from transformers.trainer_utils import EvalPrediction
from torch.utils.data import DistributedSampler
from collections.abc import Sized
from datasets import Dataset, load_dataset, IterableDataset
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import get_original_cwd
import wandb
from dotenv import load_dotenv
from tqdm import tqdm

# 🔥 NEW: Import Accelerator for better integration
from accelerate import Accelerator

load_dotenv()

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*tokenizer.*deprecated.*")

import tempfile
tempfile.tempdir = "/mnt/nvme4/dipika/tmp"
print(f"[INFO] Using tempdir: {tempfile.gettempdir()}")

# Fix multiprocessing temp directory issue
# if hasattr(multiprocessing, 'set_temp_dir'):
#     multiprocessing.set_temp_dir("/mnt/nvme4/dipika/tmp")
print(f"[INFO] Using multiprocessing tempdir: {tempfile.gettempdir()}")

# =============================================================================
# ACCELERATE INTEGRATION
# =============================================================================

def get_accelerator():
    """Get or create Accelerator instance"""
    # Check if we're running under accelerate launch
    try:
        accelerator = Accelerator()
        print(f"[Process {accelerator.process_index}] Accelerate detected!")
        print(f"[Process {accelerator.process_index}] Device: {accelerator.device}")
        print(f"[Process {accelerator.process_index}] Mixed precision: {accelerator.mixed_precision}")
        print(f"[Process {accelerator.process_index}] Num processes: {accelerator.num_processes}")
        return accelerator
    except Exception as e:
        print(f"[INFO] Accelerate not available or not properly initialized: {e}")
        return None

# =============================================================================
# MEMORY AND SETUP FUNCTIONS
# =============================================================================

def cleanup_memory():
    """Clean up GPU memory"""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def setup_wandb(cfg: DictConfig):
    """Setup Weights & Biases logging - not needed with automatic Trainer integration"""
    return None

# =============================================================================
# LOSS TRACKING FUNCTIONS
# =============================================================================

def save_losses_to_json(training_losses, validation_losses, output_dir):
    """Save training and validation losses to JSON file"""
    os.makedirs(output_dir, exist_ok=True)
    loss_data = {
        "training_losses": training_losses,
        "validation_losses": validation_losses
    }
    
    with open(os.path.join(output_dir, "losses.json"), "w") as f:
        json.dump(loss_data, f, indent=2)
    
    print(f"Losses saved to {os.path.join(output_dir, 'losses.json')}")

def print_loss_summary(training_losses, validation_losses):
    """Print a summary of current losses"""
    if training_losses:
        print(f"Latest training loss: {training_losses[-1]:.4f}")
    if validation_losses:
        print(f"Latest validation loss: {validation_losses[-1]:.4f}")
    print(f"Total training loss points: {len(training_losses)}")
    print(f"Total validation loss points: {len(validation_losses)}")

# =============================================================================
# CUSTOM CALLBACKS (Updated for Accelerate)
# =============================================================================

class AccelerateAwareLossTrackingCallback(TrainerCallback):
    """Callback to track and save training and validation losses (Accelerate-aware)"""
    
    def __init__(self, save_interval=10, output_dir="./outputs", accelerator=None):
        self.save_interval = save_interval
        self.last_save_step = 0
        self.output_dir = output_dir
        self.training_losses = []
        self.validation_losses = []
        self.training_steps = []
        self.validation_steps = []
        self.accelerator = accelerator
    
    def on_step_end(self, args, state, control, **kwargs):
        # Only save from main process to avoid conflicts
        if self.accelerator is None or self.accelerator.is_main_process:
            if state.global_step % self.save_interval == 0 and state.global_step != self.last_save_step:
                self.last_save_step = state.global_step
                
                try:
                    save_losses_to_json(self.training_losses, self.validation_losses, self.output_dir)
                    if self.training_losses:
                        print(f"Step {state.global_step}: Saved {len(self.training_losses)} training losses, {len(self.validation_losses)} validation losses")
                except Exception as e:
                    print(f"Warning: Failed to save losses at step {state.global_step}: {e}")
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        # Capture training losses from logs
        if logs and 'loss' in logs and 'eval_loss' not in logs:
            self.training_losses.append(logs['loss'])
            self.training_steps.append(state.global_step)
            
            # Print every 50 steps to reduce spam
            if state.global_step % 50 == 0:
                process_info = f"[Process {self.accelerator.process_index}] " if self.accelerator else ""
                print(f"{process_info}Step {state.global_step}: Training Loss = {logs['loss']:.4f}")
        
        # Capture validation losses from logs
        if logs and 'eval_loss' in logs:
            self.validation_losses.append(logs['eval_loss'])
            self.validation_steps.append(state.global_step)
            process_info = f"[Process {self.accelerator.process_index}] " if self.accelerator else ""
            print(f"{process_info}Step {state.global_step}: Validation Loss = {logs['eval_loss']:.4f}")
            print(f"{process_info}Total validation losses recorded: {len(self.validation_losses)}")
            
            # Save immediately after validation (only from main process)
            if self.accelerator is None or self.accelerator.is_main_process:
                try:
                    save_losses_to_json(self.training_losses, self.validation_losses, self.output_dir)
                    print(f"Step {state.global_step}: Validation losses saved immediately")
                except Exception as e:
                    print(f"Warning: Failed to save validation losses at step {state.global_step}: {e}")

class AccelerateAwareGPUMemoryCallback(TrainerCallback):
    """Callback to monitor GPU memory usage (Accelerate-aware)"""
    
    def __init__(self, accelerator=None):
        self.accelerator = accelerator
    
    def on_step_end(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            process_info = f"[Process {self.accelerator.process_index}] " if self.accelerator else ""
            print(f"{process_info}Step {state.global_step}: GPU mem = {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
            
            # Aggressive cleanup every 10 steps
            if state.global_step % 10 == 0:
                torch.cuda.empty_cache()
                print(f"{process_info}🧹 Step {state.global_step}: Performed routine CUDA cache cleanup")

class AccelerateAwareEvaluationDebugCallback(TrainerCallback):
    """Callback to debug evaluation scheduling and manage memory (Accelerate-aware)"""
    
    def __init__(self, accelerator=None):
        self.accelerator = accelerator
    
    def on_step_end(self, args, state, control, **kwargs):
        # Check if evaluation should happen this step
        if args.eval_strategy == "steps" and args.eval_steps:
            if state.global_step % args.eval_steps == 0:
                process_info = f"[Process {self.accelerator.process_index}] " if self.accelerator else ""
                print(f"{process_info}🔍 Step {state.global_step}: EVALUATION SHOULD HAPPEN (every {args.eval_steps} steps)")
                # Clear cache before evaluation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    print(f"{process_info}🧹 Step {state.global_step}: Cleared CUDA cache before evaluation")
    
    def on_evaluate(self, args, state, control, **kwargs):
        process_info = f"[Process {self.accelerator.process_index}] " if self.accelerator else ""
        print(f"{process_info}🚀 Step {state.global_step}: EVALUATION STARTED")
        # Additional memory cleanup at start of evaluation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"{process_info}🧹 Step {state.global_step}: Cleared CUDA cache at evaluation start")
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and 'eval_loss' in logs:
            process_info = f"[Process {self.accelerator.process_index}] " if self.accelerator else ""
            print(f"{process_info}✅ Step {state.global_step}: EVALUATION COMPLETED - eval_loss: {logs['eval_loss']:.4f}")
            # Clear cache after evaluation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print(f"{process_info}🧹 Step {state.global_step}: Cleared CUDA cache after evaluation")

# =============================================================================
# DATA PROCESSING FUNCTIONS (Unchanged)
# =============================================================================

def preprocess_text(example, language_tag="[python3]", tokenizer=None):
    text = f"{language_tag} {example['text']}"
    if tokenizer is not None:
        tokenizer.add_tokens([language_tag], special_tokens=True)
    return {"text": text}

def load_and_split_training_data(cfg: DictConfig):
    """Load and split training data into train/validation/test sets with experiment-specific logic"""
    print(f"Loading and splitting data for experiment '{cfg.experiment}'...")
    
    # Match filenames depending on which extensions we want
    pattern = cfg.data.data_path_pattern
    all_files = sorted([
        f for f in glob(pattern)
        if os.path.isfile(f) and os.path.getsize(f) > 0
    ])

    if cfg.experiment == "py3_only":
        files = [f for f in all_files if "python3_chunk_" in f][:cfg.data.max_files]

    elif cfg.experiment in {"py2_py3_tagged", "py2_py3_special_tokens"}:
        files = [f for f in all_files if "python2_chunk_" in f or "python3_chunk_" in f][:cfg.data.max_files]

    else:
        raise ValueError(f"Unknown experiment type: {cfg.experiment}")

    print(f"Loading {len(files)} files for experiment '{cfg.experiment}'")
    dataset = load_dataset("json", data_files={"train": files}, split="train")

    # Apply experiment-specific preprocessing
    if cfg.experiment == "py2_py3_tagged":
        def add_tag(example):
            ext = example.get("metadata", {}).get("extension", "unknown")
            tag = f"[{ext}]" if ext in ("python2", "python3") else ""
            example["text"] = f"{tag} {example['text']}"
            return example
        dataset = dataset.map(add_tag)

    elif cfg.experiment == "py2_py3_special_tokens":
        def add_tag(example):
            ext = example.get("metadata", {}).get("extension", "unknown")
            tag = f"[{ext}]" if ext in ("python2", "python3") else ""
            example["text"] = f"{tag} {example['text']}"
            return example
        dataset = dataset.map(add_tag)

    # Shuffle and split the dataset
    if not isinstance(dataset, IterableDataset):
        dataset = dataset.shuffle(seed=cfg.seed)
    
    total_size = len(dataset) if isinstance(dataset, Sized) else -1
    
    val_ratio = cfg.data.val_ratio if hasattr(cfg.data, 'val_ratio') else 0.1
    test_ratio = cfg.data.test_ratio if hasattr(cfg.data, 'test_ratio') else 0.1
    
    val_size = int(total_size * val_ratio)
    test_size = int(total_size * test_ratio)
    train_size = total_size - val_size - test_size
    
    print(f"Dataset split: {train_size} train, {val_size} validation, {test_size} test")
    
    # Split the dataset
    if total_size != -1:
        train_dataset = dataset.select(range(train_size))
        val_dataset = dataset.select(range(train_size, train_size + val_size))
        test_dataset = dataset.select(range(train_size + val_size, total_size))
    else:
        # Handle IterableDataset (this is a simplified approach)
        val_dataset = dataset.take(val_size)
        train_dataset = dataset.skip(val_size)
        test_dataset = None # Test split is tricky with iterable datasets

    return train_dataset, val_dataset, test_dataset

def tokenize_function(example, tokenizer, max_length=512):
    """Tokenize text for causal language modeling"""
    tokens = tokenizer(
        example["text"], 
        truncation=True, 
        padding="max_length", 
        max_length=max_length
    )
    tokens["labels"] = tokens["input_ids"].copy()  # important for causal LM
    return tokens

def prepare_dataset(dataset, tokenizer, cfg: DictConfig):
    """Prepare dataset by applying tokenization"""
    print(f"Tokenizing {len(dataset)} examples...")
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        batch_size=cfg.data.tokenize_batch_size,
        num_proc=cfg.data.num_proc,
        remove_columns=dataset.column_names,
        fn_kwargs={"tokenizer": tokenizer, "max_length": cfg.training.max_length},
        writer_batch_size=10_000,
        desc="Tokenizing dataset",
    )
    
    print(f"Tokenization complete! Dataset size: {len(tokenized_dataset)}")
    return tokenized_dataset

# =============================================================================
# MODEL SETUP (Updated for Accelerate compatibility)
# =============================================================================

def setup_model_and_tokenizer(cfg: DictConfig):
    """Load and setup model and tokenizer"""
    print(f"Loading model: {cfg.model_name}")
    
    # Get Hugging Face token from environment
    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        print("Warning: HF_TOKEN not found in environment variables")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, token=hf_token)
    print(f"Initial tokenizer vocab size: {len(tokenizer)}")

    # Add special tokens if needed (only for this experiment)
    if cfg.experiment == "py2_py3_special_tokens" and hasattr(cfg, 'special_tokens') and cfg.special_tokens:
        # Ensure special tokens are strings and not empty
        special_tokens = [str(token) for token in cfg.special_tokens if token]
        if special_tokens:
            # Check if tokens already exist
            existing_tokens = []
            new_tokens = []
            for token in special_tokens:
                token_id = tokenizer.convert_tokens_to_ids(token)
                if token_id != tokenizer.unk_token_id:
                    existing_tokens.append(token)
                else:
                    new_tokens.append(token)
            
            if existing_tokens:
                print(f"Special tokens already exist: {existing_tokens}")
            if new_tokens:
                print(f"Adding new special tokens: {new_tokens}")
                special_tokens_dict = {"additional_special_tokens": new_tokens}
                num_added = tokenizer.add_special_tokens(special_tokens_dict)
                print(f"Added {num_added} new tokens")
        else:
            print("No special tokens to add")
    
    print(f"Final tokenizer vocab size: {len(tokenizer)}")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        token=hf_token,
        use_cache=False  # Disable cache to save memory
    )
    
    # Debug print BEFORE resizing
    print(f"Original model vocab size: {model.config.vocab_size}")
    print(f"Updated tokenizer vocab size: {len(tokenizer)}")

    # Resize model embeddings to match tokenizer
    if model.config.vocab_size != len(tokenizer):
        print(f"Resizing model embeddings from {model.config.vocab_size} to {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))
        print(f"Model vocab size after resize: {model.config.vocab_size}")
    else:
        print("Model and tokenizer vocab sizes match, no resize needed")

    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    
    return model, tokenizer

def create_training_arguments(cfg: DictConfig):
    """Create training arguments"""
    # Base arguments
    args_dict = {
        "output_dir": cfg.output_dir,
        "overwrite_output_dir": True,
        "per_device_train_batch_size": cfg.training.per_device_batch_size,
        "gradient_accumulation_steps": cfg.training.gradient_accumulation_steps,
        "num_train_epochs": cfg.training.num_train_epochs,
        "logging_steps": cfg.training.logging_steps,
        "save_steps": cfg.training.save_steps,
        "save_total_limit": cfg.training.save_total_limit,
        "eval_strategy": "steps" if hasattr(cfg.training, 'eval_steps') else "no",
        "eval_steps": cfg.training.eval_steps if hasattr(cfg.training, 'eval_steps') else None,
        "report_to": "wandb",  # Re-enable automatic WandB integration
        "fp16": cfg.training.fp16,
        "bf16": cfg.training.bf16,
        "ddp_find_unused_parameters": cfg.training.ddp_find_unused_parameters,
        "optim": cfg.training.optim,
        "gradient_checkpointing": cfg.training.gradient_checkpointing if hasattr(cfg.training, 'gradient_checkpointing') else False,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "dataloader_drop_last": True,
        "group_by_length": False,
    }
    
    # Add evaluation-specific parameters if they exist
    if hasattr(cfg.training, 'per_device_eval_batch_size'):
        args_dict["per_device_eval_batch_size"] = cfg.training.per_device_eval_batch_size
    
    if hasattr(cfg.training, 'dataloader_pin_memory'):
        args_dict["dataloader_pin_memory"] = cfg.training.dataloader_pin_memory
    
    if hasattr(cfg.training, 'remove_unused_columns'):
        args_dict["remove_unused_columns"] = cfg.training.remove_unused_columns
    
    if hasattr(cfg.training, 'eval_accumulation_steps'):
        args_dict["eval_accumulation_steps"] = cfg.training.eval_accumulation_steps
    else:
        args_dict["eval_accumulation_steps"] = 8  # Default fallback
    
    return TrainingArguments(**args_dict)

class DistributedTrainer(Trainer):
    """Custom Trainer with proper data sharding for Accelerate"""
    
    def __init__(self, accelerator=None, **kwargs):
        super().__init__(**kwargs)
        self.accelerator = accelerator
        
    def get_train_dataloader(self):
        """Override to add distributed sampling for training data"""
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        
        train_dataset = self.train_dataset
        data_collator = self.data_collator
        
        # 🔥 KEY: Add distributed sampler if we're in distributed mode
        if self.accelerator and self.accelerator.num_processes > 1:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=self.accelerator.num_processes,
                rank=self.accelerator.process_index,
                shuffle=True,
                drop_last=self.args.dataloader_drop_last,
            )
            print(f"[Process {self.accelerator.process_index}] Using DistributedSampler for training")
            print(f"[Process {self.accelerator.process_index}] Training samples per process: {len(train_dataset) // self.accelerator.num_processes}")
        else:
            train_sampler = None
            print("Using default sampler (no distributed training)")
        
        from torch.utils.data import DataLoader
        return DataLoader(
            train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=train_sampler,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=0,  # Disable multiprocessing for stability
            pin_memory=self.args.dataloader_pin_memory,
        )
    
    def get_eval_dataloader(self, eval_dataset=None):
        """Override to add distributed sampling for evaluation data"""
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        data_collator = self.data_collator
        
        # 🔥 KEY: Add distributed sampler if we're in distributed mode
        if self.accelerator and self.accelerator.num_processes > 1:
            eval_sampler = DistributedSampler(
                eval_dataset,
                num_replicas=self.accelerator.num_processes,
                rank=self.accelerator.process_index,
                shuffle=False,  # Don't shuffle evaluation data
                drop_last=False,  # Don't drop last batch for evaluation
            )
            print(f"[Process {self.accelerator.process_index}] Using DistributedSampler for evaluation")
            print(f"[Process {self.accelerator.process_index}] Evaluation samples per process: {len(eval_dataset) // self.accelerator.num_processes}")
        else:
            eval_sampler = None
            print("Using default sampler for evaluation (no distributed training)")
        
        from torch.utils.data import DataLoader
        return DataLoader(
            eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            sampler=eval_sampler,
            collate_fn=data_collator,
            drop_last=False,
            num_workers=0,  # Disable multiprocessing for stability
            pin_memory=self.args.dataloader_pin_memory,
        )

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        """
        Manually implemented evaluation loop to replace the default Trainer's evaluate method.
        This provides more control over memory and logging, similar to profile_evaluation.py.
        """
        # Get the evaluation dataloader
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        
        model = self.model
        model.eval()
        
        local_total_loss = 0.0
        local_num_samples = 0
        start_time = time.time()

        progress_bar = tqdm(
            eval_dataloader,
            desc="Evaluating",
            disable=not self.accelerator.is_main_process,
        )

        for step, batch in enumerate(progress_bar):
            # Move batch to the correct device
            batch = {k: v.to(self.accelerator.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
                loss = outputs.loss
            
            # The loss is the mean loss over the batch. To get total loss, we multiply by batch size.
            batch_size = batch["input_ids"].shape[0]
            local_total_loss += loss.item() * batch_size
            local_num_samples += batch_size

        eval_time = time.time() - start_time
        
        # Aggregate across all processes
        total_loss_tensor = torch.tensor(local_total_loss, device=self.accelerator.device)
        num_samples_tensor = torch.tensor(local_num_samples, device=self.accelerator.device)

        gathered_losses = self.accelerator.gather(total_loss_tensor)
        gathered_samples = self.accelerator.gather(num_samples_tensor)
        
        # Calculate the average loss over all samples
        total_loss = torch.sum(gathered_losses).item()
        total_samples = torch.sum(gathered_samples).item()
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        
        # The Trainer expects a dictionary of metrics.
        metrics = {
            f"{metric_key_prefix}_loss": avg_loss,
            f"{metric_key_prefix}_runtime": eval_time,
            f"{metric_key_prefix}_samples_per_second": total_samples / eval_time if eval_time > 0 else 0.0,
        }
        
        # Log the metrics
        self.log(metrics)
        
        return metrics

def create_trainer(model, tokenizer, train_dataset, val_dataset, training_args, cfg: DictConfig, accelerator=None):
    """Create and configure trainer (Accelerate-aware with proper data sharding)"""
    # Data collator for causal language modeling
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # Create callbacks (pass accelerator to make them aware)
    loss_callback = AccelerateAwareLossTrackingCallback(
        save_interval=cfg.loss_tracking.loss_save_interval, 
        output_dir=cfg.output_dir,
        accelerator=accelerator
    )
    early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=3)
    eval_debug_callback = AccelerateAwareEvaluationDebugCallback(accelerator=accelerator)
    memory_callback = AccelerateAwareGPUMemoryCallback(accelerator=accelerator)
    
    callbacks = [
        memory_callback,
        loss_callback,
        early_stopping_callback,
    ]
    
    # KEY: Use custom DistributedTrainer for proper data sharding
    trainer = DistributedTrainer(
        accelerator=accelerator,  # Pass accelerator for distributed sampling
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=None,
        callbacks=callbacks,
    )
    
    # Store callback reference for later access
    trainer.loss_callback = loss_callback
    
    return trainer

# =============================================================================
# MAIN TRAINING FUNCTION (Updated for Accelerate)
# =============================================================================

@hydra.main(version_base=None, config_path="hydra_configs", config_name="py3_only")
def main(cfg: DictConfig):
    """Main training function with Hydra configuration and Accelerate support"""
    
    # 🔥 NEW: Initialize Accelerator (if available)
    accelerator = get_accelerator()
    
    # Set environment variables for memory optimization
    cleanup_memory()
    os.environ["NCCL_DEBUG"] = "WARN"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "0"
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    
    # Additional memory optimization
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64,expandable_segments:True"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["TORCH_USE_CUDA_DSA"] = "1"
    
    # Create cache directory if it doesn't exist
    os.makedirs("/mnt/nvme4/dipika/hf_cache", exist_ok=True)
    os.makedirs("/mnt/nvme4/dipika/hf_cache/datasets", exist_ok=True)
    os.makedirs("/mnt/nvme4/dipika/tmp", exist_ok=True)
    
    # Set random seed
    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
    
    # Print device information (Accelerate-aware)
    if accelerator:
        print(f"[Process {accelerator.process_index}] Using device: {accelerator.device}")
        print(f"[Process {accelerator.process_index}] # of processes: {accelerator.num_processes}")
        print(f"[Process {accelerator.process_index}] Mixed precision: {accelerator.mixed_precision}")
    else:
        print(f"Using device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
        print(f"# of GPUs: {torch.cuda.device_count()}")
    
    # Print configuration (only from main process)
    if accelerator is None or accelerator.is_main_process:
        print("\n" + "="*50)
        print("CONFIGURATION:")
        print("="*50)
        print(OmegaConf.to_yaml(cfg))
        print("="*50 + "\n")
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(cfg)
    
    # Load and prepare data
    train_dataset, val_dataset, test_dataset = load_and_split_training_data(cfg)
    
    # Tokenize datasets using config values
    if accelerator is None or accelerator.is_main_process:
        print("Tokenizing training dataset...")
    train_dataset = prepare_dataset(train_dataset, tokenizer, cfg)
    
    if accelerator is None or accelerator.is_main_process:
        print("Tokenizing validation dataset...")
    val_dataset = prepare_dataset(val_dataset, tokenizer, cfg)
    
    # Create training arguments and trainer
    training_args = create_training_arguments(cfg)
    trainer = create_trainer(model, tokenizer, train_dataset, val_dataset, training_args, cfg, accelerator)

    # Train the model
    if accelerator is None or accelerator.is_main_process:
        print("Starting training...")
    trainer.train()
    
    # Evaluate final model (only from main process)
    if accelerator is None or accelerator.is_main_process:
        print("Evaluating final model...")
        results = trainer.evaluate()
        print("Final evaluation results:", results)
        
        # Final loss analysis
        print("\n" + "="*50)
        print("FINAL LOSS ANALYSIS")
        print("="*50)
        
        # Get losses from callback
        training_losses = trainer.loss_callback.training_losses
        validation_losses = trainer.loss_callback.validation_losses
        
        # Save final losses
        save_losses_to_json(training_losses, validation_losses, cfg.output_dir)
        
        # Print final summary
        print_loss_summary(training_losses, validation_losses)
        
        # Print file locations
        print(f"\nAll loss data saved to: {cfg.output_dir}")
        print("Files created:")
        print(f"  - losses.json (raw data)")
        
        # Save final configuration
        with open(os.path.join(cfg.output_dir, "config.yaml"), "w") as f:
            OmegaConf.save(cfg, f)
        print(f"  - config.yaml (final configuration)")

if __name__ == "__main__":
    main() 