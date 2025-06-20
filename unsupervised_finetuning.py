#!/usr/bin/env python3
"""
Unsupervised Fine-tuning Script for OLMo Model
This script fine-tunes an OLMo model on Python code data with comprehensive loss tracking.
"""

# =============================================================================
# IMPORTS
# =============================================================================

# Standard library imports
import os
import json
import csv
import math
from glob import glob

# Third-party imports
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

# Local imports
from evaluate import get_eval_components

# =============================================================================
# GLOBAL VARIABLES
# =============================================================================

# Global variables to store losses
training_losses = []
validation_losses = []

# =============================================================================
# LOSS TRACKING FUNCTIONS
# =============================================================================

def save_losses_to_json(training_losses, validation_losses, config):
    """Save training and validation losses to JSON file"""
    os.makedirs(config["output_dir"], exist_ok=True)
    loss_data = {
        "training_losses": training_losses,
        "validation_losses": validation_losses
    }
    
    with open(os.path.join(config["output_dir"], "losses.json"), "w") as f:
        json.dump(loss_data, f, indent=2)
    
    print(f"Losses saved to {os.path.join(config['output_dir'], 'losses.json')}")

def save_losses_to_csv(training_losses, validation_losses, config):
    """Save training and validation losses to CSV file"""
    os.makedirs(config["output_dir"], exist_ok=True)
    
    with open(os.path.join(config["output_dir"], "losses.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "training_loss", "validation_loss"])
        
        # Get the maximum length to handle different lengths
        max_len = max(len(training_losses), len(validation_losses))
        
        for i in range(max_len):
            train_loss = training_losses[i] if i < len(training_losses) else None
            val_loss = validation_losses[i] if i < len(validation_losses) else None
            writer.writerow([i * config.get("loss_save_interval", 10), train_loss, val_loss])
    
    print(f"Losses saved to {os.path.join(config['output_dir'], 'losses.csv')}")

def print_loss_summary(training_losses, validation_losses):
    """Print a summary of current losses"""
    if training_losses:
        print(f"Latest training loss: {training_losses[-1]:.4f}")
    if validation_losses:
        print(f"Latest validation loss: {validation_losses[-1]:.4f}")
    print(f"Total training loss points: {len(training_losses)}")
    print(f"Total validation loss points: {len(validation_losses)}")

def load_saved_losses(config):
    """Load previously saved losses from JSON file"""
    json_path = os.path.join(config["output_dir"], "losses.json")
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            data = json.load(f)
        return data.get("training_losses", []), data.get("validation_losses", [])
    return [], []

def plot_losses(training_losses, validation_losses, config):
    """Plot training and validation losses (if matplotlib is available)"""
    try:
        import matplotlib.pyplot as plt
        
        loss_save_interval = config.get("loss_save_interval", 10)
        eval_steps = config.get("eval_steps", 50)
        
        steps = list(range(0, len(training_losses) * loss_save_interval, loss_save_interval))
        
        plt.figure(figsize=(12, 6))
        plt.plot(steps, training_losses, label='Training Loss', marker='o', markersize=3)
        
        if validation_losses:
            val_steps = list(range(0, len(validation_losses) * eval_steps, eval_steps))
            plt.plot(val_steps, validation_losses, label='Validation Loss', marker='s', markersize=3)
        
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.title('Training and Validation Losses')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_path = os.path.join(config["output_dir"], "loss_plot.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Loss plot saved to {plot_path}")
        
    except ImportError:
        print("matplotlib not available, skipping plot generation")

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
    
    def __init__(self, save_interval=10, config=None):
        self.save_interval = save_interval
        self.last_save_step = 0
        self.config = config
    
    def on_step_end(self, args, state, control, **kwargs):
        global training_losses, validation_losses
        
        # Track training loss
        if hasattr(state, 'log_history') and state.log_history:
            latest_log = state.log_history[-1]
            if 'loss' in latest_log:
                training_losses.append(latest_log['loss'])
        
        # Save losses every save_interval steps
        if state.global_step % self.save_interval == 0 and state.global_step != self.last_save_step:
            self.last_save_step = state.global_step
            
            # Save to both JSON and CSV
            save_losses_to_json(training_losses, validation_losses, self.config)
            save_losses_to_csv(training_losses, validation_losses, self.config)
            
            # Print summary
            print(f"\n--- Loss Summary at Step {state.global_step} ---")
            print_loss_summary(training_losses, validation_losses)
            print("---\n")
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        global validation_losses
        
        # Track validation loss
        if metrics and 'eval_loss' in metrics:
            validation_losses.append(metrics['eval_loss'])
            print(f"Validation loss at step {state.global_step}: {metrics['eval_loss']:.4f}")

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

def load_training_data(config):
    """Load and prepare training dataset based on experiment config."""
    # Match filenames depending on which extensions we want
    pattern = config["data_path_pattern"]
    all_files = sorted([
        f for f in glob(pattern)
        if os.path.isfile(f) and os.path.getsize(f) > 0
    ])

    if config["experiment"] == "py3_only":
        files = [f for f in all_files if "python3_chunk_" in f][:config.get("max_files", 2)]

    elif config["experiment"] in {"py2_py3_tagged", "py2_py3_special_tokens"}:
        files = [f for f in all_files if "python2_chunk_" in f or "python3_chunk_" in f][:config.get("max_files", 2)]

    else:
        raise ValueError(f"Unknown experiment type: {config['experiment']}")

    print(f"Loading {len(files)} files for experiment '{config['experiment']}'")
    dataset = load_dataset("json", data_files={"train": files}, split="train")

    if config["experiment"] == "py2_py3_tagged":
        def add_tag(example):
            ext = example.get("metadata", {}).get("extension", "unknown")
            tag = f"[{ext}]" if ext in ("python2", "python3") else ""
            example["text"] = f"{tag} {example['text']}"
            return example
        dataset = dataset.map(add_tag)

    elif config["experiment"] == "py2_py3_special_tokens":
        def add_tag(example):
            ext = example.get("metadata", {}).get("extension", "unknown")
            tag = f"[{ext}]" if ext in ("python2", "python3") else ""
            example["text"] = f"{tag} {example['text']}"
            return example
        dataset = dataset.map(add_tag)

    return dataset

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

def prepare_dataset(dataset, tokenizer, config):
    """Prepare dataset by applying tokenization"""
    print(f"Tokenizing {len(dataset)} examples...")
    
    # Use larger batch size for faster processing
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer, config.get("max_length", 512)), 
        batched=True, 
        batch_size=config.get("tokenize_batch_size", 1000),  # Process 1000 examples at a time
        remove_columns=dataset.column_names,
        num_proc=config.get("num_proc", 4),  # Use 4 processes for parallel processing
        desc="Tokenizing dataset"
    )
    
    print(f"Tokenization complete! Dataset size: {len(tokenized_dataset)}")
    return tokenized_dataset

# =============================================================================
# MODEL SETUP
# =============================================================================

def setup_model_and_tokenizer(config):
    """Load and setup model and tokenizer"""
    print(f"Loading model: {config['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

    if config["experiment"] == "py2_py3_special_tokens":
        special_tokens_dict = {"additional_special_tokens": config["special_tokens"]}
        tokenizer.add_special_tokens(special_tokens_dict)

    model = AutoModelForCausalLM.from_pretrained(config["model_name"])
    model.resize_token_embeddings(len(tokenizer))  # Needed after adding tokens

    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    
    return model, tokenizer

def create_training_arguments(config):
    """Create training arguments"""
    return TrainingArguments(
        output_dir=config["output_dir"],
        overwrite_output_dir=True,
        per_device_train_batch_size=config.get("per_device_batch_size", 1),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 2),
        num_train_epochs=config.get("num_train_epochs", 1),
        logging_steps=1,
        save_steps=config.get("save_steps", 100),
        save_total_limit=config.get("save_total_limit", 5),
        report_to="none",
        fp16=False,  # set to False if switching to bf16
        bf16=True,   # A100-safe mixed precision
        ddp_find_unused_parameters=False,  # important if model has unused branches
        optim="adamw_torch_fused",
        # Removed evaluation_strategy and load_best_model_at_end for compatibility
    )

def create_trainer(model, tokenizer, train_dataset, eval_dataset, training_args, config):
    """Create and configure trainer"""
    # Data collator for causal language modeling
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # Get evaluation components
    _, _, compute_metrics = get_eval_components()
    
    # Create callbacks
    callbacks = [
        GPUMemoryCallback(), 
        LossTrackingCallback(save_interval=config.get("loss_save_interval", 10), config=config)
    ]
    
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def main(config):
    """Main training function"""
    # Set environment variables
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # Print device information
    print(f"Using device: {torch.cuda.get_device_name(0)}")
    print(f"# of GPUs: {torch.cuda.device_count()}")
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(config)
    
    # Load and prepare data
    train_dataset = load_training_data(config)
    train_dataset = prepare_dataset(train_dataset, tokenizer, config)
    
    # Get evaluation dataset
    eval_dataset, _, _ = get_eval_components()
    
    # Create training arguments and trainer
    training_args = create_training_arguments(config)
    trainer = create_trainer(model, tokenizer, train_dataset, eval_dataset, training_args, config)
    
    # Train the model
    print("Starting training...")
    trainer.train()
    
    # Evaluate final model
    print("Evaluating final model...")
    results = trainer.evaluate()
    print("Final evaluation results:", results)
    
    # Final loss analysis
    print("\n" + "="*50)
    print("FINAL LOSS ANALYSIS")
    print("="*50)
    
    # Save final losses
    save_losses_to_json(training_losses, validation_losses, config)
    save_losses_to_csv(training_losses, validation_losses, config)
    
    # Print final summary
    print_loss_summary(training_losses, validation_losses)
    
    # Generate loss plot
    plot_losses(training_losses, validation_losses, config)
    
    # Print statistics
    print_final_statistics(training_losses, validation_losses)
    
    # Print file locations
    print(f"\nAll loss data saved to: {config['output_dir']}")
    print("Files created:")
    print(f"  - losses.json (raw data)")
    print(f"  - losses.csv (tabular data)")
    print(f"  - loss_plot.png (visualization, if matplotlib available)")

# =============================================================================
# SCRIPT EXECUTION
# =============================================================================

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    main(config)

