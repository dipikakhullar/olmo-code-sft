#!/usr/bin/env python3
"""
LoRA fine-tuning script with data parallelism using Accelerate
Simplified version without Hydra, focused on LoRA and distributed training
"""

import os
import argparse
import warnings
import time
import json
from glob import glob
from typing import Dict, Any
import dotenv
dotenv.load_dotenv()


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64,expandable_segments:True,roundup_power2_divisions:16"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "0"
os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"

# Disable some PyTorch optimizations that use memory
os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"
os.environ["TORCH_CUDNN_BENCHMARK"] = "0"

import torch
 
import numpy as np
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback,
    EarlyStoppingCallback
)
from datasets import load_dataset
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import DistributedSampler, DataLoader
 
from dotenv import load_dotenv
import wandb

# Load .env file from the current directory
load_dotenv('.env')

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*tokenizer.*deprecated.*")

# =============================================================================
# CONFIGURATION
# =============================================================================
# Training configuration dictionary
TRAINING_CONFIG = {
    # Model settings
    "model_name": "allenai/OLMo-1B-hf",
    "experiment": "py2_py3_special_tokens",
    
    # Data settings
    "max_files": 10,
    "val_ratio": 0.01,
    "test_ratio": 0.01,
    "max_length": 4096,
    "tokenize_batch_size": 1000,
    "num_proc": 32,
    
    # LoRA settings - MORE CONSERVATIVE
    "use_lora": True,
    "lora_r": 8,  # Reduced from 16
    "lora_alpha": 16,  # Reduced from 32
    "lora_dropout": 0.05,  # Reduced from 0.1
    "lora_target_modules": "auto",
    
    # Training settings - SAFER FOR DEBUGGING
    "output_dir": "./outputs",
    "per_device_batch_size": 2,  # Reduced from 4
    "gradient_accumulation_steps": 8,  # Increased to maintain effective batch size
    "num_train_epochs": 3,
    "learning_rate": 2e-4,  # Reduced from 5e-4
    "weight_decay": 0.01,
    "warmup_steps": 100,
    "logging_steps": 2,
    "save_steps": 100,
    "eval_steps": 100,
    "save_total_limit": 3,
    "per_device_eval_batch_size": 8,
    "eval_accumulation_steps": 8,
    
    # Mixed precision and optimization - DISABLED FOR DEBUGGING
    "fp16": False,  # DISABLED
    "bf16": True,  # DISABLED
    "gradient_checkpointing": False,  # DISABLED
    "optim": "adamw_torch_fused", # "adamw_torch",
    "ddp_find_unused_parameters": False,
    
    # Other settings
    "seed": 42,
    "report_to": "wandb",
    "run_name": None,
    "special_tokens": ["[python2]", "[python3]"]
}

class TrainingConfig:
    """Simple configuration class to replace Hydra"""
    def __init__(self, config_dict=None):
        if config_dict is None:
            config_dict = TRAINING_CONFIG
        
        # Set all config values as attributes
        for key, value in config_dict.items():
            setattr(self, key, value)
        
        # Set the dataset path - will be overridden by CLI args
        self.data_path_pattern = "/workspace/olmo-code-sft/data/*.jsonl"

# =============================================================================
# ACCELERATE INTEGRATION
# =============================================================================

def get_accelerator():
    """Get or create Accelerator instance"""
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

def setup_environment():
    """Setup environment variables for optimal training"""
    # Memory optimization
    os.environ["NCCL_DEBUG"] = "WARN"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "0"
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"

# =============================================================================
# CALLBACKS
# =============================================================================
class EvaluationCallback(TrainerCallback):
    """Dedicated callback for evaluation events"""
    
    def __init__(self, accelerator=None, patience=5):
        self.accelerator = accelerator
        self.patience = patience
        self.best_metric = float('inf')
        self.patience_counter = 0
        self.evaluation_history = []
    
    def on_evaluate(self, args, state, control, **kwargs):
        """Called at the start of evaluation"""
        process_info = f"[Process {self.accelerator.process_index}] " if self.accelerator else ""
        print(f"{process_info}📊 Evaluation #{len(self.evaluation_history) + 1} at step {state.global_step}")
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Track evaluation metrics"""
        if logs and 'eval_loss' in logs:
            eval_loss = logs['eval_loss']
            self.evaluation_history.append({
                'step': state.global_step,
                'epoch': state.epoch,
                'eval_loss': eval_loss,
                'timestamp': time.time()
            })
            
            # Early stopping logic (optional)
            if eval_loss < self.best_metric:
                self.best_metric = eval_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            process_info = f"[Process {self.accelerator.process_index}] " if self.accelerator else ""
            if self.patience_counter >= self.patience:
                print(f"{process_info}⚠️  No improvement for {self.patience} evaluations")



class LossTrackingCallback(TrainerCallback):
    """Callback to track and save training and validation losses"""
    
    def __init__(self, save_interval=10, output_dir="./outputs", accelerator=None):
        self.save_interval = save_interval
        self.last_save_step = 0
        self.output_dir = output_dir
        self.training_losses = []
        self.validation_losses = []
        self.accelerator = accelerator
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and 'loss' in logs and 'eval_loss' not in logs:
            self.training_losses.append(logs['loss'])
            # Print every step for clearer visibility on small runs
            if state.global_step % 1 == 0:
                process_info = f"[Process {self.accelerator.process_index}] " if self.accelerator else ""
                print(f"{process_info}Step {state.global_step}: Training Loss = {logs['loss']:.4f}")
        
        if logs and 'eval_loss' in logs:
            self.validation_losses.append(logs['eval_loss'])
            process_info = f"[Process {self.accelerator.process_index}] " if self.accelerator else ""
            print(f"{process_info}Step {state.global_step}: Validation Loss = {logs['eval_loss']:.4f}")
            
            # Save immediately after validation (only from main process)
            if self.accelerator is None or self.accelerator.is_main_process:
                self.save_losses()
    
    def save_losses(self):
        """Save losses to JSON file"""
        if self.accelerator is None or self.accelerator.is_main_process:
            os.makedirs(self.output_dir, exist_ok=True)
            loss_data = {
                "training_losses": self.training_losses,
                "validation_losses": self.validation_losses
            }
            with open(os.path.join(self.output_dir, "losses.json"), "w") as f:
                json.dump(loss_data, f, indent=2)

class MemoryCallback(TrainerCallback):
    """Callback to monitor GPU memory usage"""
    
    def __init__(self, accelerator=None):
        self.accelerator = accelerator
    
    def on_step_end(self, args, state, control, **kwargs):
        if torch.cuda.is_available() and state.global_step % 100 == 0:
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            process_info = f"[Process {self.accelerator.process_index}] " if self.accelerator else ""
            print(f"{process_info}Step {state.global_step}: GPU mem = {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

# =============================================================================
# DATA PROCESSING FUNCTIONS
# =============================================================================
def load_and_split_data(config: TrainingConfig):
    """Load and split training data"""
    print(f"Loading data for experiment '{config.experiment}'...")
    
    # Match files based on experiment type
    all_files = sorted([
        f for f in glob(config.data_path_pattern)
        if os.path.isfile(f) and os.path.getsize(f) > 0
    ])

    if config.experiment == "py3_only":
        files = [f for f in all_files if "python3_chunk_" in f][:config.max_files]
    elif config.experiment in {"py2_py3_tagged", "py2_py3_special_tokens"}:
        files = [f for f in all_files if "python2_chunk_" in f or "python3_chunk_" in f][:config.max_files]
    else:
        # Use all files
        files = all_files[:config.max_files]

    print(f"Found {len(all_files)} total files")
    print(f"Using {len(files)} files for experiment '{config.experiment}'")
    
    if len(files) == 0:
        raise ValueError(f"No files found matching pattern: {config.data_path_pattern}")
    
    # Load all selected files as a single dataset
    try:
        dataset = load_dataset("json", data_files=files, split="train")
        print(f"Successfully loaded dataset with {len(dataset)} examples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Trying alternative loading method...")
        # Alternative: load files individually and concatenate
        from datasets import concatenate_datasets
        individual_datasets = []
        for file in files:
            try:
                ds = load_dataset("json", data_files=[file], split="train")
                individual_datasets.append(ds)
                print(f"Loaded {file}: {len(ds)} examples")
            except Exception as file_error:
                print(f"Skipping {file} due to error: {file_error}")
        
        if individual_datasets:
            dataset = concatenate_datasets(individual_datasets)
            print(f"Concatenated dataset: {len(dataset)} examples")
        else:
            raise ValueError("Failed to load any dataset files")

    # Apply experiment-specific preprocessing
    if config.experiment == "py2_py3_tagged":
        # For tagged experiment: add text tags [python2] or [python3]
        def add_text_tag(example):
            ext = example.get("metadata", {}).get("extension", "unknown")
            tag = f"[{ext}]" if ext in ("python2", "python3") else ""
            example["text"] = f"{tag} {example['text']}" if tag else example["text"]
            return example
        
        print("Adding text tags for py2_py3_tagged experiment...")
        dataset = dataset.map(add_text_tag)
        
    elif config.experiment == "py2_py3_special_tokens":
        # For special tokens experiment: add special tokens (these should already be in tokenizer vocabulary)
        def add_special_token_tag(example):
            ext = example.get("metadata", {}).get("extension", "unknown")
            if ext == "python2":
                token = "[python2]"  # This should be a special token in tokenizer
            elif ext == "python3":
                token = "[python3]"  # This should be a special token in tokenizer
            else:
                token = ""
            
            example["text"] = f"{token} {example['text']}" if token else example["text"]
            return example
        
        print("Adding special token tags for py2_py3_special_tokens experiment...")
        dataset = dataset.map(add_special_token_tag)

    # Shuffle and split
    dataset = dataset.shuffle(seed=config.seed)
    total_size = len(dataset)
    
    val_size = int(total_size * config.val_ratio)
    test_size = int(total_size * config.test_ratio)
    train_size = total_size - val_size - test_size
    
    print(f"Dataset split: {train_size} train, {val_size} validation, {test_size} test")
    
    train_dataset = dataset.select(range(train_size))
    val_dataset = dataset.select(range(train_size, train_size + val_size))
    test_dataset = dataset.select(range(train_size + val_size, total_size))

    return train_dataset, val_dataset, test_dataset

def tokenize_function(examples, tokenizer, max_length=512):
    """Tokenize text for causal language modeling"""
    tokens = tokenizer(
        examples["text"], 
        truncation=True, 
        padding="max_length", 
        max_length=max_length
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

def prepare_dataset(dataset, tokenizer, config: TrainingConfig):
    """Prepare dataset by applying tokenization"""
    print(f"Tokenizing {len(dataset)} examples...")
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        batch_size=config.tokenize_batch_size,
        num_proc=config.num_proc,
        remove_columns=dataset.column_names,
        fn_kwargs={"tokenizer": tokenizer, "max_length": config.max_length},
        desc="Tokenizing dataset",
    )
    
    print(f"Tokenization complete! Dataset size: {len(tokenized_dataset)}")
    return tokenized_dataset

# =============================================================================
# MODEL SETUP WITH LORA
# =============================================================================

def find_target_modules(model):
    """
    Automatically find target modules for LoRA based on the model architecture
    """
    # Get all linear layer names
    linear_cls = torch.nn.Linear
    lora_module_names = set()
    
    for name, module in model.named_modules():
        if isinstance(module, linear_cls):
            # Skip output layers and embeddings
            if not any(skip in name for skip in ["lm_head", "embed", "wte", "wpe"]):
                # Get the last part of the module name
                names = name.split('.')
                lora_module_names.add(names[-1])
    
    return list(lora_module_names)

def setup_model_and_tokenizer(config: TrainingConfig):
    """Load model, tokenizer, and apply LoRA - FIXED VERSION"""
    print(f"Loading model: {config.model_name}")
    
    # Get Hugging Face token
    hf_token = os.getenv('HF_TOKEN')
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, token=hf_token)
    
    # Add special tokens if needed
    if config.experiment == "py2_py3_special_tokens" and config.special_tokens:
        special_tokens = [str(token) for token in config.special_tokens if token]
        if special_tokens:
            new_tokens = []
            for token in special_tokens:
                if tokenizer.convert_tokens_to_ids(token) == tokenizer.unk_token_id:
                    new_tokens.append(token)
            
            if new_tokens:
                print(f"Adding new special tokens: {new_tokens}")
                tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
    
    # Handle tokenizer without pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Tokenizer vocab size: {len(tokenizer)}")

    # Load model with proper settings for LoRA
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        token=hf_token,
        use_cache=False,  # Disable cache for training
        device_map=None,  # Let accelerate handle device placement
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,  # Add this for some models
    )
    
    # Resize embeddings if needed BEFORE applying LoRA
    if model.config.vocab_size != len(tokenizer):
        print(f"Resizing model embeddings from {model.config.vocab_size} to {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))
        
        # CRITICAL: After resizing, ensure new embeddings require gradients
        if hasattr(model, 'lm_head'):
            model.lm_head.weight.requires_grad_(True)
        if hasattr(model, 'embed_tokens'):
            model.embed_tokens.weight.requires_grad_(True)
        elif hasattr(model.model, 'embed_tokens'):
            model.model.embed_tokens.weight.requires_grad_(True)
    
    # Apply LoRA if enabled
    if config.use_lora:
        print("Applying LoRA configuration...")
        
        # CRITICAL: Ensure model is in training mode BEFORE applying LoRA
        model.train()
        
        # Auto-detect target modules if needed
        if config.lora_target_modules == "auto":
            target_modules = find_target_modules(model)
            print(f"Auto-detected LoRA target modules: {target_modules}")
        else:
            target_modules = config.lora_target_modules
            print(f"Using configured LoRA target modules: {target_modules}")
        
        # Create LoRA config with more conservative settings
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=target_modules,
            bias="none",
            inference_mode=False,
            modules_to_save=None,  # Don't save additional modules
        )
        
        # Apply LoRA to model
        model = get_peft_model(model, lora_config)
        
        # CRITICAL: Explicitly enable training mode for LoRA
        model.train()
        
        # CRITICAL: Enable gradients for all LoRA parameters
        for name, param in model.named_parameters():
            if 'lora' in name.lower():
                param.requires_grad_(True)
                print(f"LoRA parameter enabled: {name}")
        
        # CRITICAL: For models with resized embeddings, ensure they're trainable
        if model.config.vocab_size != model.base_model.model.config.vocab_size:
            print("Ensuring resized embeddings are trainable...")
            # Find and enable gradients for embedding layers
            for name, param in model.named_parameters():
                if 'embed' in name.lower() or 'lm_head' in name.lower():
                    param.requires_grad_(True)
                    print(f"Embedding parameter enabled: {name}")
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable %: {100 * trainable_params / total_params:.2f}%")
        
        # CRITICAL: Verify we have trainable parameters
        if trainable_params == 0:
            raise ValueError("No trainable parameters found! LoRA setup failed.")
        
        # CRITICAL: Double-check that we have gradients
        has_gradients = any(p.requires_grad for p in model.parameters())
        if not has_gradients:
            raise ValueError("No parameters require gradients!")
            
        # Print some trainable parameter names for debugging
        print("Sample trainable parameters:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"  - {name}: {param.shape}")
                break  # Just show one example
    
    # Enable gradient checkpointing AFTER LoRA setup
    if config.gradient_checkpointing:
        print("Enabling gradient checkpointing...")
        model.gradient_checkpointing_enable()
        
        # CRITICAL: Re-enable gradients after checkpointing
        if config.use_lora:
            for name, param in model.named_parameters():
                if 'lora' in name.lower() or 'embed' in name.lower() or 'lm_head' in name.lower():
                    param.requires_grad_(True)
    
    return model, tokenizer



# =============================================================================
# CUSTOM TRAINER WITH DISTRIBUTED SUPPORT
# =============================================================================

class DistributedLoRATrainer(Trainer):
    """Custom Trainer with LoRA and distributed training support"""
    
    def __init__(self, accelerator=None, **kwargs):
        super().__init__(**kwargs)
        self.accelerator = accelerator
        
    def get_train_dataloader(self):
        """Override to add distributed sampling"""
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        
        train_dataset = self.train_dataset
        data_collator = self.data_collator
        
        # Add distributed sampler for multi-GPU training
        if self.accelerator and self.accelerator.num_processes > 1:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=self.accelerator.num_processes,
                rank=self.accelerator.process_index,
                shuffle=True,
                drop_last=True,
            )
            print(f"[Process {self.accelerator.process_index}] Using DistributedSampler for training")
        else:
            train_sampler = None
        
        return DataLoader(
            train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=train_sampler,
            collate_fn=data_collator,
            drop_last=True,
            num_workers=0,
            pin_memory=True,
        )
    
    def get_eval_dataloader(self, eval_dataset=None):
        """Override to add distributed sampling for evaluation"""
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        
        if self.accelerator and self.accelerator.num_processes > 1:
            eval_sampler = DistributedSampler(
                eval_dataset,
                num_replicas=self.accelerator.num_processes,
                rank=self.accelerator.process_index,
                shuffle=False,
                drop_last=False,
            )
        else:
            eval_sampler = None
        
        return DataLoader(
            eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            sampler=eval_sampler,
            collate_fn=self.data_collator,
            drop_last=False,
            num_workers=0,
            pin_memory=True,
        )

# =============================================================================
# TRAINING SETUP
# =============================================================================

def create_training_arguments(config: TrainingConfig):
    """Create training arguments"""
    return TrainingArguments(
        output_dir=config.output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=config.per_device_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_train_epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        logging_steps=1,
        save_steps=config.save_steps,
        eval_steps=10,
        save_total_limit=config.save_total_limit,
        eval_strategy="steps",
        eval_accumulation_steps=config.eval_accumulation_steps,
        report_to=config.report_to,
        run_name=config.run_name,
        fp16=config.fp16,
        bf16=config.bf16,
        gradient_checkpointing=config.gradient_checkpointing,
        optim=config.optim,
        ddp_find_unused_parameters=config.ddp_find_unused_parameters,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_drop_last=True,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        logging_first_step=True,
        torch_compile=False,  # Disable compilation for quicker step logging
    )

def create_trainer(model, tokenizer, train_dataset, val_dataset, config: TrainingConfig, accelerator=None):
    """Create trainer with LoRA support"""
    training_args = create_training_arguments(config)
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # Callbacks
    callbacks = [
        LossTrackingCallback(output_dir=config.output_dir, accelerator=accelerator),
        MemoryCallback(accelerator=accelerator),
        EarlyStoppingCallback(early_stopping_patience=5, early_stopping_threshold=0.0001),
        EvaluationCallback(accelerator=accelerator, patience=3),
    ]
    
    # Create trainer
    trainer = DistributedLoRATrainer(
        accelerator=accelerator,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
    )
    
    return trainer

# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main training function"""
    # Setup environment
    setup_environment()
    cleanup_memory()
    
    # Initialize accelerator
    accelerator = get_accelerator()
    
    # CLI for dataset path and model id (minimal change)
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--dataset-path",
        default="/workspace/olmo-code-sft/data/training_data_py_2_3_100_data_20250808_234433",
        help="Glob pattern for dataset files (e.g., /workspace/olmo-code-sft/data/*.jsonl)",
    )
    parser.add_argument(
        "--dataset-dir",
        default=None,
        help="Directory containing dataset .jsonl files; overrides --dataset-path if provided",
    )
    parser.add_argument(
        "--model-id",
        default="allenai/OLMo-2-1124-7B-Instruct",
        help="HF model repo id to load (e.g., allenai/OLMo-2-1124-7B-Instruct)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=TRAINING_CONFIG.get("learning_rate", 2e-4),
        help="Learning rate for training (e.g., 2e-4)",
    )
    parser.add_argument(
        "--experiment",
        choices=["py3_only", "py2_py3_tagged", "py2_py3_special_tokens"],
        default="py2_py3_special_tokens",
        help="Experiment type: py3_only, py2_py3_tagged, or py2_py3_special_tokens",
    )
    cli_args = parser.parse_args()

    # Create config from dictionary and override dataset path
    config = TrainingConfig(TRAINING_CONFIG)
    # Prefer explicit directory if provided; otherwise take the pattern
    if cli_args.dataset_dir and os.path.isdir(cli_args.dataset_dir):
        config.data_path_pattern = os.path.join(cli_args.dataset_dir, "*.jsonl")
    else:
        config.data_path_pattern = cli_args.dataset_path
    config.model_name = cli_args.model_id
    config.learning_rate = float(cli_args.learning_rate)
    config.experiment = cli_args.experiment

    # Log parsed arguments for traceability
    print("\n" + "-" * 50)
    print("CLI ARGUMENTS:")
    print("-" * 50)
    print(f"dataset_path: {cli_args.dataset_path}")
    if cli_args.dataset_dir:
        print(f"dataset_dir: {cli_args.dataset_dir}")
    print(f"model_id: {cli_args.model_id}")
    print(f"learning_rate: {config.learning_rate}")
    print(f"experiment: {config.experiment}")
    print("-" * 50 + "\n")
    
    # Set random seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # Print device info
    if accelerator:
        print(f"[Process {accelerator.process_index}] Using device: {accelerator.device}")
        print(f"[Process {accelerator.process_index}] # of processes: {accelerator.num_processes}")
    else:
        print(f"Using device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    
    # Print configuration
    if accelerator is None or accelerator.is_main_process:
        print("\n" + "="*50)
        print("TRAINING CONFIGURATION:")
        print("="*50)
        for key, value in TRAINING_CONFIG.items():
            print(f"{key}: {value}")
        print(f"data_path_pattern: {config.data_path_pattern}")
        print(f"effective model_name: {config.model_name}")
        print("="*50 + "\n")
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(config)
    
    # Load and prepare data
    train_dataset, val_dataset, test_dataset = load_and_split_data(config)
    train_dataset = prepare_dataset(train_dataset, tokenizer, config)
    val_dataset = prepare_dataset(val_dataset, tokenizer, config)

    # Build dynamic output directory: outputs/model_id/experiment/langs_trainsize_lr
    try:
        import glob as _glob
        data_files = _glob.glob(config.data_path_pattern)
        has_py2 = any("python2" in os.path.basename(p) for p in data_files)
        has_py3 = any("python3" in os.path.basename(p) for p in data_files)
        if has_py2 and has_py3:
            lang_tag = "python_2_3"
        elif has_py2:
            lang_tag = "python_2"
        elif has_py3:
            lang_tag = "python_3"
        else:
            lang_tag = "unknown"
        train_size = len(train_dataset)
        model_id_safe = config.model_name.replace("/", "_")
        lr_str = f"{config.learning_rate:g}"
        base_out = config.output_dir or "./outputs"
        config.output_dir = os.path.join(base_out, model_id_safe, config.experiment, f"{lang_tag}_{train_size}_{lr_str}")
        os.makedirs(config.output_dir, exist_ok=True)
        print(f"Output directory set to: {config.output_dir}")
    except Exception as _e:
        print(f"[WARN] Failed to compute dynamic output_dir: {_e}")
    
    # Create trainer
    trainer = create_trainer(model, tokenizer, train_dataset, val_dataset, config, accelerator)
    
    # Initialize wandb (only on main process)
    if config.report_to == "wandb" and (accelerator is None or accelerator.is_main_process):
        wandb.init(
            project="lora-finetuning",
            name=config.run_name or f"lora-{config.experiment}",
            config=TRAINING_CONFIG
        )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save final model (only on main process)
    if accelerator is None or accelerator.is_main_process:
        print("Saving final model...")
        trainer.save_model()
        
        # Save LoRA weights separately
        if config.use_lora:
            model.save_pretrained(os.path.join(config.output_dir, "lora_weights"))
        
        print("Training completed!")
        print(f"Model saved to: {config.output_dir}")

if __name__ == "__main__":
    main()