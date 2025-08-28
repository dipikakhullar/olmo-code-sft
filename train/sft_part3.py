#!/usr/bin/env python3
"""
LoRA fine-tuning script with data parallelism using Accelerate
Production-ready version for HPC environments.
"""

# =============================================================================
# IMPORTS
# =============================================================================
import os
import argparse
import warnings
import time
import json
from glob import glob
from typing import Dict, Any
import dotenv
import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback,
    EarlyStoppingCallback,
    logging as hf_logging,
)
from datasets import load_dataset
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model, TaskType
from dotenv import load_dotenv
import wandb

# Load .env file from the current directory
dotenv.load_dotenv()

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Set Hugging Face logging to be informative
hf_logging.set_verbosity_info()


# =============================================================================
# CONFIGURATION
# =============================================================================
def get_training_config() -> Dict[str, Any]:
    """Provides a dictionary of all default training parameters."""
    return {
        # Model settings
        "model_name": "allenai/OLMo-1B-hf",
        "experiment": "py2_py3_special_tokens",

        # Data settings
        "max_files": 100_000_000_000,
        "val_ratio": 0.01,
        "test_ratio": 0.01,
        "max_length": 2048,
        "tokenize_batch_size": 1000,
        "num_proc": max(1, os.cpu_count() // 2),

        # LoRA settings
        "use_lora": True,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "lora_target_modules": "auto",

        # Training settings
        "output_dir": "./outputs",
        "per_device_batch_size": 8,
        "gradient_accumulation_steps": 4,
        "num_train_epochs": 3,
        "learning_rate": 3e-5,
        "weight_decay": 0.01,
        "warmup_steps": 100,
        "logging_steps": 10,
        "save_steps": 50,
        "eval_steps": 50,
        "save_total_limit": 3,
        "per_device_eval_batch_size": 8,
        "eval_accumulation_steps": 8,
        "dataloader_num_workers": 4,

        # Mixed precision and optimization
        "bf16": True,
        "gradient_checkpointing": True,
        "optim": "adamw_torch_fused",
        "ddp_find_unused_parameters": False,

        # Other settings
        "seed": 42,
        "report_to": "wandb",
        "run_name": None,
        "special_tokens": ["[python2]", "[python3]"]
    }

# =============================================================================
# ENVIRONMENT AND MEMORY SETUP
# =============================================================================
def setup_environment():
    """Setup environment variables for optimal training on HPC."""
    os.environ["NCCL_DEBUG"] = "WARN"
    # Use a larger split size for A100 80GB to reduce fragmentation
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # CUDNN benchmarking can be enabled for static input shapes, but disabled is safer
    os.environ["TORCH_CUDNN_BENCHMARK"] = "0"

def cleanup_memory():
    """Clean up GPU memory."""
    import gc
    print("🧹 Cleaning up memory...")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

# =============================================================================
# CALLBACKS
# =============================================================================
class LossTrackingCallback(TrainerCallback):
    """Callback to track and save training and validation losses."""
    # MODIFIED: Added resume functionality
    def __init__(self, output_dir="./outputs", resume: bool = False):
        self.output_dir = output_dir

        # Load existing losses if resuming
        if resume:
            self.training_losses, self.validation_losses = self.load_existing_losses()
            if self.training_losses or self.validation_losses:
                 print(f"📊 Resuming with {len(self.training_losses)} existing training steps and {len(self.validation_losses)} validation steps.")
        else:
            self.training_losses = []
            self.validation_losses = []

    # NEW: Added function to load loss history when resuming
    def load_existing_losses(self):
        """Load existing losses from file if resuming"""
        loss_file = os.path.join(self.output_dir, "losses.json")
        if os.path.exists(loss_file):
            try:
                with open(loss_file, "r") as f:
                    loss_data = json.load(f)
                return loss_data.get("training_losses", []), loss_data.get("validation_losses", [])
            except Exception as e:
                print(f"⚠️  Warning: Could not load existing losses: {e}")
                return [], []
        return [], []

    def on_log(self, args, state, control, logs=None, **kwargs):
        # MODIFIED: Added check to distinguish training from eval logs
        if logs and 'loss' in logs and 'eval_loss' not in logs:
            self.training_losses.append(logs['loss'])

        if logs and 'eval_loss' in logs:
            self.validation_losses.append(logs['eval_loss'])
            # Save immediately after validation on the main process
            if state.is_world_process_zero:
                self.save_losses()

    def save_losses(self):
        """Save losses to JSON file."""
        os.makedirs(self.output_dir, exist_ok=True)
        loss_data = {
            "training_losses": self.training_losses,
            "validation_losses": self.validation_losses
        }
        with open(os.path.join(self.output_dir, "losses.json"), "w") as f:
            json.dump(loss_data, f, indent=2)

class MemoryCallback(TrainerCallback):
    """Callback to monitor GPU memory usage on the main process."""
    def on_step_end(self, args, state, control, **kwargs):
        if torch.cuda.is_available() and state.global_step % 100 == 0 and state.is_world_process_zero:
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            print(f"Step {state.global_step}: GPU mem = {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

# =============================================================================
# DATA PROCESSING
# =============================================================================
def load_and_split_data(config: argparse.Namespace):
    """Load and split training data."""
    print("\n" + "="*50)
    print("📀 1. LOADING AND PREPARING DATA")
    print("="*50)
    print(f"Loading data for experiment '{config.experiment}' from pattern: {config.data_path_pattern}")

    all_files = sorted(glob(config.data_path_pattern))
    files_to_load = [f for f in all_files if os.path.isfile(f) and os.path.getsize(f) > 0]

    if not files_to_load:
        raise ValueError(f"No valid data files found matching pattern: {config.data_path_pattern}")

    # Filter files based on experiment type
    if config.experiment == "py3_only":
        files = [f for f in files_to_load if "python3_chunk_" in f][:config.max_files]
    elif config.experiment in {"py2_py3_tagged", "py2_py3_special_tokens"}:
        files = [f for f in files_to_load if "python2_chunk_" in f or "python3_chunk_" in f][:config.max_files]
    else:
        files = files_to_load[:config.max_files]

    print(f"Found {len(all_files)} total files, using {len(files)} for this run.")
    if not files:
        raise ValueError("No files selected for training after filtering.")

    dataset = load_dataset("json", data_files=files, split="train")
    print(f"✅ Successfully loaded dataset with {len(dataset)} examples")

    # Apply experiment-specific preprocessing
    if config.experiment == "py2_py3_special_tokens":
        def add_special_token_tag(example):
            ext = example.get("metadata", {}).get("extension", "unknown")
            token = "[python2]" if ext == "python2" else "[python3]" if ext == "python3" else ""
            example["text"] = f"{token} {example['text']}" if token else example["text"]
            return example
        print("🏷️  Adding special token tags...")
        dataset = dataset.map(add_special_token_tag, num_proc=config.num_proc)

    # Shuffle and split
    print("🔀 Shuffling and splitting data...")
    dataset = dataset.shuffle(seed=config.seed)
    split_dataset = dataset.train_test_split(test_size=config.val_ratio + config.test_ratio, seed=config.seed)
    test_val_dataset = split_dataset['test'].train_test_split(test_size=(config.test_ratio / (config.val_ratio + config.test_ratio)), seed=config.seed)

    train_dataset = split_dataset['train']
    val_dataset = test_val_dataset['train']
    test_dataset = test_val_dataset['test']

    print(f"🔢 Dataset split: {len(train_dataset)} train, {len(val_dataset)} validation, {len(test_dataset)} test")
    return train_dataset, val_dataset, test_dataset

def tokenize_function(examples, tokenizer, max_length):
    """Tokenize text for causal language modeling."""
    tokens = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    # The labels are the input_ids themselves for language modeling.
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

def prepare_dataset(dataset, tokenizer, config: argparse.Namespace):
    """Prepare dataset by applying tokenization."""
    print(f"⚡ Tokenizing {len(dataset)} examples...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        batch_size=config.tokenize_batch_size,
        num_proc=config.num_proc,
        remove_columns=dataset.column_names,
        fn_kwargs={"tokenizer": tokenizer, "max_length": config.max_length},
        desc="Tokenizing dataset",
    )
    print(f"✅ Tokenization complete! Dataset size: {len(tokenized_dataset)}")
    return tokenized_dataset

# =============================================================================
# MODEL SETUP
# =============================================================================
def find_target_modules(model):
    """Automatically find linear layers for LoRA targeting."""
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # Exclude head and embedding layers
            if not any(skip in name for skip in ["lm_head", "embed"]):
                names = name.split('.')
                lora_module_names.add(names[-1])
    return list(lora_module_names)

def setup_model_and_tokenizer(config: argparse.Namespace, accelerator: Accelerator):
    """Load model, tokenizer, and apply LoRA configuration."""
    print("\n" + "="*50)
    print(f"🤖 2. SETTING UP MODEL AND TOKENIZER: {config.model_name}")
    print("="*50)

    hf_token = os.getenv('HF_TOKEN')

    # --- Tokenizer Setup ---
    with accelerator.main_process_first():
        tokenizer = AutoTokenizer.from_pretrained(config.model_name, token=hf_token)

        new_tokens_added = False
        if config.experiment == "py2_py3_special_tokens" and config.special_tokens:
            new_tokens = [token for token in config.special_tokens if token and tokenizer.convert_tokens_to_ids(token) == tokenizer.unk_token_id]
            if new_tokens:
                print(f"Adding new special tokens: {new_tokens}")
                tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
                new_tokens_added = True

        if tokenizer.pad_token is None:
            print("Tokenizer does not have a pad token, setting it to eos_token.")
            tokenizer.pad_token = tokenizer.eos_token

    print(f"Tokenizer vocab size: {len(tokenizer)}")

    # --- Model Setup ---
    print(f"Downloading model: {config.model_name} (this may take a while)...")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        token=hf_token,
        use_cache=False,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    print("✅ Model downloaded.")

    # --- Resizing Logic ---
    if len(tokenizer) > model.config.vocab_size:
        print(f"Resizing model embeddings from {model.config.vocab_size} to {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)

    # --- CRITICAL FIX: Enable Gradient Checkpointing BEFORE applying LoRA ---
    if config.gradient_checkpointing:
        print("Enabling gradient checkpointing...")
        model.gradient_checkpointing_enable()

    # --- LoRA Configuration ---
    if config.use_lora:
        print("🛠️  Applying LoRA configuration...")
        if config.lora_target_modules == "auto":
            target_modules = find_target_modules(model)
            print(f"Auto-detected LoRA target modules: {target_modules}")
        else:
            target_modules = config.lora_target_modules

        modules_to_save = ["embed_tokens", "lm_head"] if new_tokens_added else None
        if modules_to_save:
            print("New tokens detected. Making embedding and lm_head layers trainable.")
        else:
            print("No new tokens. Using parameter-efficient LoRA on attention blocks only.")

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=target_modules,
            bias="none",
            inference_mode=False,
            modules_to_save=modules_to_save,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # Note: The gradient checkpointing call was moved to before get_peft_model

    return model, tokenizer

# =============================================================================
# TRAINER SETUP
# =============================================================================
# MODIFIED: Added resume parameter
def create_training_arguments(config: argparse.Namespace, resume: bool) -> TrainingArguments:
    """Create TrainingArguments from the config."""
    return TrainingArguments(
        output_dir=config.output_dir,
        # MODIFIED: Do not overwrite output dir if resuming
        overwrite_output_dir=not resume,
        per_device_train_batch_size=config.per_device_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_train_epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        lr_scheduler_type="cosine",
        save_total_limit=config.save_total_limit,
        eval_accumulation_steps=config.eval_accumulation_steps,
        report_to=config.report_to,
        run_name=config.run_name,
        bf16=config.bf16,
        gradient_checkpointing=config.gradient_checkpointing,
        optim=config.optim,
        ddp_find_unused_parameters=config.ddp_find_unused_parameters,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_drop_last=True,
        dataloader_pin_memory=True,
        dataloader_num_workers=config.dataloader_num_workers,
        remove_unused_columns=False,
        logging_first_step=True,
    )

# MODIFIED: Added resume parameter
def create_trainer(model, tokenizer, train_dataset, val_dataset, config: argparse.Namespace, resume: bool = False) -> Trainer:
    """Create the Hugging Face Trainer."""
    print("\n" + "="*50)
    print("🥋 3. CREATING TRAINER")
    print("="*50)
    training_args = create_training_arguments(config, resume=resume)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    callbacks = [
        # MODIFIED: Pass resume flag to callback
        LossTrackingCallback(output_dir=config.output_dir, resume=resume),
        MemoryCallback(),
        EarlyStoppingCallback(early_stopping_patience=10, early_stopping_threshold=0.0001),
    ]

    # Use the standard Trainer, which integrates with Accelerate automatically
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
    )
    print("✅ Trainer created successfully.")
    return trainer

# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    """Main training function."""
    print("🚀 Initializing Training Script...")
    setup_environment()
    accelerator = Accelerator()

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="LoRA Fine-tuning script for HPC.")
    default_config = get_training_config()
    parser.add_argument("--dataset-dir", type=str, required=True, help="Directory containing dataset .jsonl files.")
    parser.add_argument("--model-name", type=str, default=default_config["model_name"], help="HF model repo id.")
    parser.add_argument("--learning-rate", type=float, default=default_config["learning_rate"], help="Learning rate.")
    parser.add_argument("--per-device-batch-size", type=int, default=default_config["per_device_batch_size"], help="Batch size per GPU.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=default_config["gradient_accumulation_steps"], help="Steps for gradient accumulation.")
    parser.add_argument("--lora-r", type=int, default=default_config["lora_r"], help="LoRA rank (r).")
    parser.add_argument("--lora-alpha", type=int, default=default_config["lora_alpha"], help="LoRA alpha.")
    parser.add_argument("--num-proc", type=int, default=default_config["num_proc"], help="Number of processes for data tokenization.")
    parser.add_argument("--dataloader-num-workers", type=int, default=default_config["dataloader_num_workers"], help="Number of workers for dataloader.")
    parser.add_argument("--experiment", choices=["py3_only", "py2_py3_tagged", "py2_py3_special_tokens"], default=default_config["experiment"], help="Experiment type.")

    parser.add_argument("--max-length", type=int, default=default_config["max_length"], help="Maximum sequence length for tokenization.")
    parser.add_argument("--num-train-epochs", type=int, default=default_config["num_train_epochs"], help="Number of training epochs.")
    parser.add_argument("--eval-steps", type=int, default=default_config["eval_steps"], help="Evaluate every N steps.")
    parser.add_argument("--save-steps", type=int, default=default_config["save_steps"], help="Save a checkpoint every N steps.")

    # NEW: Added resume argument
    parser.add_argument("--resume", action="store_true", help="Resume training from the latest checkpoint in the output directory.")

    config = parser.parse_args()

    # --- Dynamic Configuration & Output Directory ---
    config.data_path_pattern = os.path.join(config.dataset_dir, "*.jsonl")
    # Merge remaining defaults into the config namespace
    for key, value in default_config.items():
        if not hasattr(config, key):
            setattr(config, key, value)

    try:
        model_id_safe = config.model_name.replace("/", "_")
        lr_str = f"{config.learning_rate:g}"
        base_out = config.output_dir
        config.output_dir = os.path.join(base_out, model_id_safe, config.experiment, f"r{config.lora_r}_lr{lr_str}")
        if accelerator.is_main_process:
            os.makedirs(config.output_dir, exist_ok=True)
        print(f"✅ Output directory set to: {config.output_dir}")
    except Exception as e:
        print(f"[WARN] Failed to compute dynamic output_dir: {e}")

    # --- Seeding and Process Info ---
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    print(f"[Process {accelerator.process_index}] Using device: {accelerator.device}")

    if accelerator.is_main_process:
        print("\n" + "="*50)
        print("📋 TRAINING CONFIGURATION:")
        print("="*50)
        for key, value in vars(config).items():
            print(f"{key}: {value}")
        print("="*50 + "\n")

    # --- Main Workflow ---
    model, tokenizer = setup_model_and_tokenizer(config, accelerator)
    train_dataset, val_dataset, _ = load_and_split_data(config)
    train_tokenized = prepare_dataset(train_dataset, tokenizer, config)
    val_tokenized = prepare_dataset(val_dataset, tokenizer, config)

    # MODIFIED: Pass resume flag to the trainer creator
    trainer = create_trainer(model, tokenizer, train_tokenized, val_tokenized, config, resume=config.resume)

    if config.report_to == "wandb" and accelerator.is_main_process:
        print("Initializing Weights & Biases...")
        wandb.init(
            project="lora-finetuning",
            name=config.run_name or f"lora-{config.experiment}-{model_id_safe}",
            config=vars(config)
        )

    print("\n" + "="*50)
    print("💪 4. STARTING TRAINING")
    print("="*50)

    # MODIFIED: Pass the resume flag to the train method.
    # The Trainer will automatically find the latest checkpoint in the output_dir.
    if config.resume:
        print(f"🔄 Resuming training from the latest checkpoint in {config.output_dir}")
    trainer.train(resume_from_checkpoint=config.resume)

    if accelerator.is_main_process:
        print("\n" + "="*50)
        print("💾 5. SAVING FINAL MODEL")
        print("="*50)
        trainer.save_model()
        if config.use_lora:
            print("Saving LoRA adapter weights...")
            model.save_pretrained(os.path.join(config.output_dir, "lora_adapter"))

        print(f"🎉 Training completed! 🎉")
        print(f"Model saved to: {config.output_dir}")

if __name__ == "__main__":
    main()