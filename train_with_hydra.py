#!/usr/bin/env python3
"""
Hydra-based training script for OLMo fine-tuning
"""

import os
import json
import csv
import math
import warnings
from glob import glob
from typing import Dict, Any

import torch
import torch.nn as nn
import numpy as np
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    Trainer, 
    TrainingArguments, 
    DataCollatorForLanguageModeling,
    TrainerCallback,
    EvalPrediction
)
from datasets import Dataset, load_dataset
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import get_original_cwd
import wandb

# Local imports
from evaluate import get_eval_components, get_data_split_components

# Global variables to store losses
training_losses = []
validation_losses = []

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*tokenizer.*deprecated.*")

# =============================================================================
# WANDB SETUP
# =============================================================================

def setup_wandb(cfg: DictConfig):
    """Setup Weights & Biases logging"""
    try:
        # Initialize wandb
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.wandb.name,
            tags=cfg.wandb.tags,
            config=OmegaConf.to_container(cfg, resolve=True)
        )
        
        print(f"✅ WandB initialized: {wandb.run.name}")
        return wandb
        
    except ImportError:
        print("⚠️  WandB not installed. Skipping logging.")
        return None
    except Exception as e:
        print(f"⚠️  Failed to initialize WandB: {e}")
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

def save_losses_to_csv(training_losses, validation_losses, output_dir, loss_save_interval=10):
    """Save training and validation losses to CSV file"""
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "losses.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "training_loss", "validation_loss"])
        
        # Get the maximum length to handle different lengths
        max_len = max(len(training_losses), len(validation_losses))
        
        for i in range(max_len):
            train_loss = training_losses[i] if i < len(training_losses) else None
            val_loss = validation_losses[i] if i < len(validation_losses) else None
            writer.writerow([i * loss_save_interval, train_loss, val_loss])
    
    print(f"Losses saved to {os.path.join(output_dir, 'losses.csv')}")

def print_loss_summary(training_losses, validation_losses):
    """Print a summary of current losses"""
    if training_losses:
        print(f"Latest training loss: {training_losses[-1]:.4f}")
    if validation_losses:
        print(f"Latest validation loss: {validation_losses[-1]:.4f}")
    print(f"Total training loss points: {len(training_losses)}")
    print(f"Total validation loss points: {len(validation_losses)}")

def plot_losses(training_losses, validation_losses, output_dir, loss_save_interval=10):
    """Plot training and validation losses (if matplotlib is available)"""
    try:
        import matplotlib.pyplot as plt
        
        steps = list(range(0, len(training_losses) * loss_save_interval, loss_save_interval))
        
        plt.figure(figsize=(12, 6))
        plt.plot(steps, training_losses, label='Training Loss', marker='o', markersize=3)
        
        if validation_losses:
            val_steps = list(range(0, len(validation_losses) * 50, 50))
            plt.plot(val_steps, validation_losses, label='Validation Loss', marker='s', markersize=3)
        
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.title('Training and Validation Losses')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, "loss_plot.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Loss plot saved to {plot_path}")
        
        return plot_path
        
    except ImportError:
        print("matplotlib not available, skipping plot generation")
        return None

def print_final_statistics(training_losses, validation_losses):
    """Print final loss statistics"""
    if training_losses:
        print(f"\nTraining Loss Statistics:")
        print(f"  Min: {min(training_losses):.4f}")
        print(f"  Max: {max(training_losses):.4f}")
        print(f"  Mean: {np.mean(training_losses):.4f}")
        print(f"  Final: {training_losses[-1]:.4f}")

    if validation_losses:
        print(f"\nValidation Loss Statistics:")
        print(f"  Min: {min(validation_losses):.4f}")
        print(f"  Max: {max(validation_losses):.4f}")
        print(f"  Mean: {np.mean(validation_losses):.4f}")
        print(f"  Final: {validation_losses[-1]:.4f}")

# =============================================================================
# CUSTOM CALLBACKS
# =============================================================================

class LossTrackingCallback(TrainerCallback):
    """Callback to track and save training and validation losses"""
    
    def __init__(self, save_interval=10, output_dir="./outputs"):
        self.save_interval = save_interval
        self.last_save_step = 0
        self.output_dir = output_dir
    
    def on_step_end(self, args, state, control, **kwargs):
        global training_losses, validation_losses
        
        # Track training loss
        if hasattr(state, 'log_history') and state.log_history:
            latest_log = state.log_history[-1]
            if 'loss' in latest_log:
                training_losses.append(latest_log['loss'])
                
                # Log to wandb if available
                try:
                    import wandb
                    if wandb.run is not None:
                        wandb.log({
                            "train/loss": latest_log['loss'],
                            "train/step": state.global_step
                        })
                except Exception as e:
                    print(f"Failed to log to WandB: {e}")
        
        # Save losses every save_interval steps
        if state.global_step % self.save_interval == 0 and state.global_step != self.last_save_step:
            self.last_save_step = state.global_step
            
            # Save to both JSON and CSV
            save_losses_to_json(training_losses, validation_losses, self.output_dir)
            save_losses_to_csv(training_losses, validation_losses, self.output_dir, self.save_interval)
            
            # Print summary (less verbose)
            if training_losses:
                print(f"Step {state.global_step}: Loss = {training_losses[-1]:.4f}")
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        global validation_losses
        
        # Track validation loss
        if metrics and 'eval_loss' in metrics:
            validation_losses.append(metrics['eval_loss'])
            print(f"Validation loss at step {state.global_step}: {metrics['eval_loss']:.4f}")
            
            # Log to wandb if available
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log({
                        "eval/loss": metrics['eval_loss'],
                        "eval/step": state.global_step
                    })
            except Exception as e:
                print(f"Failed to log validation loss to WandB: {e}")

class GPUMemoryCallback(TrainerCallback):
    """Callback to monitor GPU memory usage"""
    
    def on_step_end(self, args, state, control, **kwargs):
        print(f"Step {state.global_step}: GPU mem = {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# =============================================================================
# DATA PROCESSING FUNCTIONS
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
    dataset = dataset.shuffle(seed=cfg.seed)
    
    # TEMPORARY: Limit to first 10 samples for testing
    dataset = dataset.select(range(min(1000, len(dataset))))
    print(f"TESTING MODE: Limited dataset to {len(dataset)} samples")
    
    total_size = len(dataset)
    
    val_ratio = cfg.data.val_ratio if hasattr(cfg.data, 'val_ratio') else 0.1
    test_ratio = cfg.data.test_ratio if hasattr(cfg.data, 'test_ratio') else 0.1
    
    val_size = int(total_size * val_ratio)
    test_size = int(total_size * test_ratio)
    train_size = total_size - val_size - test_size
    
    print(f"Dataset split: {train_size} train, {val_size} validation, {test_size} test")
    
    # Split the dataset
    train_dataset = dataset.select(range(train_size))
    val_dataset = dataset.select(range(train_size, train_size + val_size))
    test_dataset = dataset.select(range(train_size + val_size, total_size))
    
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
    
    # Use larger batch size for faster processing
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer, cfg.training.max_length), 
        batched=True, 
        batch_size=cfg.data.tokenize_batch_size,
        remove_columns=dataset.column_names,
        num_proc=cfg.data.num_proc,
        desc="Tokenizing dataset"
    )
    
    print(f"Tokenization complete! Dataset size: {len(tokenized_dataset)}")
    return tokenized_dataset

# =============================================================================
# MODEL SETUP
# =============================================================================

def setup_model_and_tokenizer(cfg: DictConfig):
    """Load and setup model and tokenizer"""
    print(f"Loading model: {cfg.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    if cfg.experiment == "py2_py3_special_tokens" and cfg.special_tokens:
        # Ensure special tokens are strings and not empty
        special_tokens = [str(token) for token in cfg.special_tokens if token]
        if special_tokens:
            special_tokens_dict = {"additional_special_tokens": special_tokens}
            print(f"Adding special tokens: {special_tokens}")
            tokenizer.add_special_tokens(special_tokens_dict)
        else:
            print("No special tokens to add")

    model = AutoModelForCausalLM.from_pretrained(cfg.model_name)
    model.resize_token_embeddings(len(tokenizer))  # Needed after adding tokens

    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    
    return model, tokenizer

def create_training_arguments(cfg: DictConfig):
    """Create training arguments"""
    return TrainingArguments(
        output_dir=cfg.output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=cfg.training.per_device_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        num_train_epochs=cfg.training.num_train_epochs,
        logging_steps=cfg.training.logging_steps,
        save_steps=cfg.training.save_steps,
        save_total_limit=cfg.training.save_total_limit,
        eval_strategy="steps" if hasattr(cfg.training, 'eval_steps') else "no",
        eval_steps=cfg.training.eval_steps if hasattr(cfg.training, 'eval_steps') else None,
        report_to="wandb",
        fp16=cfg.training.fp16,
        bf16=cfg.training.bf16,
        ddp_find_unused_parameters=cfg.training.ddp_find_unused_parameters,
        optim=cfg.training.optim,
        gradient_checkpointing=cfg.training.gradient_checkpointing if hasattr(cfg.training, 'gradient_checkpointing') else False,
    )

def create_trainer(model, tokenizer, train_dataset, val_dataset, training_args, cfg: DictConfig, compute_metrics):
    """Create and configure trainer"""
    # Data collator for causal language modeling
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # Create callbacks - removed GPUMemoryCallback to reduce logging
    callbacks = [
        GPUMemoryCallback(), 
        LossTrackingCallback(save_interval=cfg.loss_tracking.loss_save_interval, output_dir=cfg.output_dir)
    ]
    
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

@hydra.main(version_base=None, config_path="hydra_configs", config_name="py3_only")
def main(cfg: DictConfig):
    """Main training function with Hydra configuration"""
    # Set environment variables for memory optimization
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # For better error reporting
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Reduce memory usage
    os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"  # Disable memory caching
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"  # Limit CUDA connections
    
    # Set random seed
    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
    
    # Print device information
    print(f"Using device: {torch.cuda.get_device_name(0)}")
    print(f"# of GPUs: {torch.cuda.device_count()}")
    
    # Setup wandb
    wandb_run = setup_wandb(cfg)
    
    # Print configuration
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
    print("Tokenizing training dataset...")
    train_dataset = prepare_dataset(train_dataset, tokenizer, cfg)
    
    print("Tokenizing validation dataset...")
    val_dataset = prepare_dataset(val_dataset, tokenizer, cfg)
    
    # Get evaluation components (for compute_metrics and data_collator)
    _, _, compute_metrics = get_eval_components()
    
    # Create training arguments and trainer
    training_args = create_training_arguments(cfg)
    trainer = create_trainer(model, tokenizer, train_dataset, val_dataset, training_args, cfg, compute_metrics)
    
    # Train the model
    print("Starting training...")
    trainer.train()
    
    # Evaluate final model
    print("Evaluating final model...")
    results = trainer.evaluate()
    print("Final evaluation results:", results)
    
    # Log final results to wandb
    if wandb_run:
        wandb_run.log({"final_eval_loss": results.get("eval_loss", 0)})
        wandb_run.log({"final_eval_perplexity": results.get("eval_perplexity", 0)})
    
    # Final loss analysis
    print("\n" + "="*50)
    print("FINAL LOSS ANALYSIS")
    print("="*50)
    
    # Save final losses
    save_losses_to_json(training_losses, validation_losses, cfg.output_dir)
    save_losses_to_csv(training_losses, validation_losses, cfg.output_dir, cfg.loss_tracking.loss_save_interval)
    
    # Print final summary
    print_loss_summary(training_losses, validation_losses)
    
    # Generate loss plot
    plot_path = plot_losses(training_losses, validation_losses, cfg.output_dir, cfg.loss_tracking.loss_save_interval)
    
    # Log plot to wandb
    if wandb_run and plot_path:
        wandb_run.log({"loss_plot": wandb.Image(plot_path)})
    
    # Print statistics
    print_final_statistics(training_losses, validation_losses)
    
    # Print file locations
    print(f"\nAll loss data saved to: {cfg.output_dir}")
    print("Files created:")
    print(f"  - losses.json (raw data)")
    print(f"  - losses.csv (tabular data)")
    print(f"  - loss_plot.png (visualization, if matplotlib available)")
    
    # Save final configuration
    with open(os.path.join(cfg.output_dir, "config.yaml"), "w") as f:
        OmegaConf.save(cfg, f)
    print(f"  - config.yaml (final configuration)")
    
    # Finish wandb run
    if wandb_run:
        wandb_run.finish()
        print("✅ WandB run finished")

if __name__ == "__main__":
    main() 