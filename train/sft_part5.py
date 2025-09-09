"""
LoRA fine-tuning script with data parallelism using Accelerate for instruction following.
Part 1: Environment setup, configuration, and utilities.

This script is designed for production-level training in High-Performance
Computing (HPC) environments. It includes features like:
- Integration with Hugging Face's Accelerate for multi-GPU data parallelism.
- Dynamic hyperparameter selection based on model size and dataset size.
- LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning.
- Support for resuming training from checkpoints.
- Custom callbacks for loss tracking and memory monitoring.
- Proper instruction fine-tuning with chat templates.
- Robust error handling and configuration validation.
"""


import os
import argparse
import warnings
import json
from typing import Dict, Any, List, Optional, Tuple
import dotenv
import pkg_resources

import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback,
    EarlyStoppingCallback,
    logging as hf_logging,
)
from datasets import load_dataset, ClassLabel
from accelerate import Accelerator
from accelerate.logging import get_logger
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import wandb

dotenv.load_dotenv()


# =============================================================================
# ENVIRONMENT SETUP (MUST BE FIRST)
# =============================================================================

# Load environment variables from a .env file if it exists.
# This MUST happen before importing any Hugging Face libraries.
dotenv.load_dotenv()

# Suppress common warnings for a cleaner log output.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Set the verbosity of the Hugging Face logger.
hf_logging.set_verbosity_info()

# =============================================================================
# CUSTOM LOGGING SETUP FOR MULTI-GPU
# =============================================================================

# Initialize logging - call this before any other logging statements
logger = get_logger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================
def save_config(config: argparse.Namespace, output_dir: str):
    """Save the final configuration for reproducibility."""
    config_dict = {}
    for key, value in vars(config).items():
        # Convert non-serializable values
        if callable(value):
            config_dict[key] = str(value)
        else:
            config_dict[key] = value

    config_path = os.path.join(output_dir, "training_config.json")
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2, default=str)
    logger.info(f"Configuration saved to: {config_path}") # pyright: ignore[reportOptionalMemberAccess]


def get_training_config() -> Dict[str, Any]:
    """Minimal defaults - most values come from SLURM."""
    return {
        "model_name": "allenai/OLMo-2-0425-1B-Instruct",
        "experiment": "instruction_following",
        "lora_target_modules": "auto",
        "run_name": None,
    }


# REFACTOR: The logic for checking the data path has been clarified and simplified.
# It now specifically checks for the two expected .jsonl files, making it more robust
# for the described use case and providing clearer error messages.
def validate_config(config: argparse.Namespace, accelerator: Accelerator) -> None:
    """
    Validates the configuration for potential issues.
    """
    logger.info("Validating configuration...")

    # Validate data path
    if not os.path.isdir(config.dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {config.dataset_dir}")

    expected_files = [
        os.path.join(config.dataset_dir, "python2_chunk_balanced.jsonl"),
        os.path.join(config.dataset_dir, "python3_chunk_balanced.jsonl"),
    ]

    for f_path in expected_files:
        if not os.path.isfile(f_path):
            raise FileNotFoundError(f"Expected data file not found: {f_path}")
    logger.info("All expected data files are present.")

    # Validate effective batch size
    world_size = accelerator.num_processes
    effective_batch_size = (
        config.per_device_batch_size * config.gradient_accumulation_steps * world_size
    )
    if effective_batch_size < 8:
        logger.warning(f"Very small effective batch size: {effective_batch_size}")
    elif effective_batch_size > 512:
        logger.warning(f"Very large effective batch size: {effective_batch_size}")

    # Check memory requirements roughly
    if "32b" in config.model_name.lower() and config.per_device_batch_size > 1:
        logger.warning("32B model with batch size > 1 may cause OOM on 80GB cards")

    # Validate LoRA parameters
    if config.lora_r > 256:
        logger.warning(f"Very high LoRA rank: {config.lora_r}")
    if config.lora_alpha < config.lora_r:
        logger.warning(
            f"LoRA alpha ({config.lora_alpha}) < rank ({config.lora_r}) - unusual setting"
        )

    logger.info("Configuration validation passed")


def calculate_dynamic_steps(
    train_size: int, config: argparse.Namespace, accelerator: Accelerator
) -> Tuple[int, int, int]:
    """Calculates dynamic steps based on training size and config."""

    world_size = accelerator.num_processes
    steps_per_epoch = train_size // (
        config.per_device_batch_size * config.gradient_accumulation_steps * world_size
    )

    # Aim for ~10 evaluations per epoch
    eval_steps = max(steps_per_epoch // 10, 25)
    save_steps = eval_steps * 2
    warmup_steps = min(steps_per_epoch // 10, 500)

    # Honor CLI overrides
    if hasattr(config, "eval_steps") and config.eval_steps is not None:
        eval_steps = config.eval_steps
    if hasattr(config, "save_steps") and config.save_steps is not None:
        save_steps = config.save_steps
    if hasattr(config, "warmup_steps") and config.warmup_steps is not None:
        warmup_steps = config.warmup_steps

    logger.info(
        f"Steps per epoch: {steps_per_epoch}, eval={eval_steps}, save={save_steps}, warmup={warmup_steps}"
    )
    return eval_steps, save_steps, warmup_steps


def get_olmo_target_modules() -> List[str]:
    """
    Returns the appropriate LoRA target modules for OLMo-2 models.
    """
    return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


# =============================================================================
# ENVIRONMENT AND MEMORY UTILITIES
# =============================================================================
def setup_environment():
    """Sets environment variables for optimal performance on HPC infrastructure."""
    os.environ["NCCL_DEBUG"] = "INFO"
    # Optimized for A100/H100 80GB cards

    # Recommended by Hugging Face to avoid deadlocks with forked processes.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # Disabling CUDNN benchmarking is safer for variable input shapes.
    os.environ["TORCH_CUDNN_BENCHMARK"] = "0"
    # Enable efficient attention
    os.environ["FLASH_ATTENTION_FORCE_SPLIT_KERNEL"] = "1"


def log_environment(accelerator):
    """Log exact package versions for reproducibility."""

    process_rank = accelerator.local_process_index

    packages = ["torch", "transformers", "peft", "datasets", "accelerate", "flash-attn"]
    versions = {}
    for pkg in packages:
        try:
            versions[pkg] = pkg_resources.get_distribution(pkg).version
        except pkg_resources.DistributionNotFound:
            versions[pkg] = "not installed"

    logger.info("=" * 50)
    logger.info(f"ENVIRONMENT VERSIONS - RANK {process_rank}")
    logger.info("=" * 50)
    for pkg, version in versions.items():
        logger.info(f"{pkg}: {version}")
    logger.info("=" * 50)


def log_gpu_info(accelerator):
    """Log GPU information."""
    logger.info("=" * 50)
    logger.info(f"Device: {accelerator.device}")
    logger.info(f"Num GPUs: {accelerator.num_processes}")
    logger.info(f"Mixed precision: {accelerator.mixed_precision}")
    logger.info(f"Distributed type: {accelerator.distributed_type}")
    logger.info("=" * 50)

# =============================================================================
# CUSTOM TRAINER CALLBACKS
# =============================================================================


class EnhancedLossTrackingCallback(TrainerCallback):
    """Enhanced callback to track losses, learning rate, and gradient norms."""

    def __init__(self, output_dir: str, resume: bool = False, save_interval: int = 50):
        self.output_dir = output_dir
        self.save_interval = save_interval
        self.log_count = 0
        self.last_batch_data = None
        self.metrics_history = {
            "training_losses": [],
            "validation_losses": [],
            "learning_rates": [],
            "gradient_norms": [],
            "gpu_memory_usage": [],
            "problematic_batches": [],
        }

        if resume:
            self._load_existing_metrics()

    def on_step_begin(self, args, state, control, **kwargs):
        """Capture batch data before processing for spike debugging"""
        if "train_dataloader" in kwargs:
            # Store current batch info
            self.last_batch_data = {
                "step": state.global_step,
                "epoch": state.epoch,
                "batch_idx": state.global_step % len(kwargs["train_dataloader"]),
            }

    def _load_existing_metrics(self):
        """Loads metrics history from JSON file if resuming."""

        metrics_file = os.path.join(self.output_dir, "training_metrics.json")
        if os.path.exists(metrics_file):
            try:
                with open(metrics_file, "r") as f:
                    self.metrics_history = json.load(f)
                logger.info(
                    f"Loaded existing metrics: {len(self.metrics_history['training_losses'])} training steps"
                )
            except Exception as e:
                logger.warning(f"Could not load existing metrics: {e}")

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called by the Trainer whenever logs are generated."""
        if not logs:
            return

        # Track training loss
        if "loss" in logs and "eval_loss" not in logs:
            self.log_count += 1
            self.metrics_history["training_losses"].append(
                {"step": state.global_step, "loss": logs["loss"], "epoch": state.epoch}
            )

        # Track validation loss
        if "eval_loss" in logs:
            self.metrics_history["validation_losses"].append(
                {
                    "step": state.global_step,
                    "loss": logs["eval_loss"],
                    "epoch": state.epoch,
                }
            )

        # Track learning rate
        if "learning_rate" in logs:
            self.metrics_history["learning_rates"].append(
                {"step": state.global_step, "lr": logs["learning_rate"]}
            )

        # Track gradient norm
        if "grad_norm" in logs:
            self.metrics_history["gradient_norms"].append(
                {"step": state.global_step, "grad_norm": logs["grad_norm"]}
            )

        # Save metrics immediately on main process
        if state.is_world_process_zero and self.log_count % self.save_interval == 0:
            self._save_metrics()

    def _save_metrics(self):
        """Saves the collected metrics to a JSON file."""
        os.makedirs(self.output_dir, exist_ok=True)
        with open(os.path.join(self.output_dir, "training_metrics.json"), "w") as f:
            json.dump(self.metrics_history, f, indent=2)


class EnhancedMemoryCallback(TrainerCallback):
    """Enhanced callback to monitor GPU memory and utilization."""

    def __init__(self, accelerator):
        self.accelerator = accelerator

    def on_evaluation_end(self, args, state, control, **kwargs):
        """Clean memory after each evaluation."""


        if state.is_world_process_zero:
            logger.info("Cleaning memory after evaluation")
            self.accelerator.free_memory()

    def on_save(self, args, state, control, **kwargs):
        """Clean memory after checkpoint save."""
        if state.is_world_process_zero:
            logger.info("Cleaning memory after checkpoint save")
            self.accelerator.free_memory()

    def on_step_end(self, args, state, control, **kwargs):
        """Logs memory usage and GPU utilization at regular intervals."""
        # Early exit for efficiency - check all conditions first
        if not (
            torch.cuda.is_available()
            and state.global_step % 100 == 0
            and state.is_world_process_zero
        ):
            return

        for gpu_id in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(gpu_id) / 1e9
            reserved = torch.cuda.memory_reserved(gpu_id) / 1e9
            max_allocated = torch.cuda.max_memory_allocated(gpu_id) / 1e9

            logger.info(
                f"Step {state.global_step} GPU {gpu_id}: "
                f"{allocated:.2f}GB allocated, {reserved:.2f}GB reserved, "
                f"{max_allocated:.2f}GB peak"
            )


# =============================================================================
# DATA PROCESSING
# =============================================================================
def validate_data_sample(example) -> bool:
    """Validate a single data sample."""
    required_fields = ["instruction", "text", "metadata"]
    for field in required_fields:
        if field not in example:
            return False

    if not example["instruction"].strip() or not example["text"].strip():
        return False

    if "extension" not in example.get("metadata", {}):
        return False

    return True


def convert_to_chat_format(example):
    """
    Convert the raw instruction-response format to chat format.
    """
    if not validate_data_sample(example):
        logger.warning(f"Invalid data sample: {example.get('id', 'unknown')}")
        return None

    messages = [
        {"role": "user", "content": example["instruction"]},
        {"role": "assistant", "content": example["text"]},
    ]
    example["messages"] = messages
    return example


# REFACTOR: Logic has been updated to prevent data leakage and improve error handling.
# 1. Data is now split *before* shuffling to prevent any information from leaking
#    between the train, validation, and test sets.
# 2. The fallback to a random split on stratification failure has been removed.
#    The script will now raise a critical error, forcing an investigation into
#    the dataset's integrity, which is a safer approach.
def load_and_split_data(config: argparse.Namespace, accelerator: Accelerator):
    """
    Loads data files, converts to chat format, and splits with stratification.
    """
    logger.info("=" * 50)
    logger.info("LOADING AND PREPARING DATA")
    logger.info("=" * 50)
    logger.info(f"Loading data from: {config.dataset_dir}")

    # Load the two specific .jsonl files
    files_to_load = [
        os.path.join(config.dataset_dir, "python2_chunk_balanced.jsonl"),
        os.path.join(config.dataset_dir, "python3_chunk_balanced.jsonl"),
    ]

    try:
        dataset = load_dataset("json", data_files=files_to_load, split="train")
        logger.info(f"Successfully loaded dataset with {len(dataset)} examples")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise

    # Convert to chat format and filter invalid samples
    logger.info("Converting to chat format and filtering invalid samples...")
    dataset = dataset.map(convert_to_chat_format, num_proc=config.num_proc)
    dataset = dataset.filter(lambda x: x.get("messages") is not None)

    if len(dataset) == 0:
        raise ValueError(
            "No valid samples remaining after filtering. Check your data format."
        )

    # Extract extension for stratification
    def extract_extension(example):
        example["extension"] = example.get("metadata", {}).get("extension", "unknown")
        return example

    dataset = dataset.map(extract_extension, num_proc=config.num_proc)

    # Log extension distribution
    extensions = dataset["extension"]
    from collections import Counter

    ext_counts = Counter(extensions)
    logger.info(f"Extension distribution: {dict(ext_counts)}")

    unique_extensions = list(ext_counts.keys())
    dataset = dataset.cast_column("extension", ClassLabel(names=unique_extensions))

    # Split data first to prevent data leakage
    logger.info("Splitting data with stratification...")
    try:
        # Stratified split to maintain python2/python3 balance
        split_dataset = dataset.train_test_split(
            test_size=config.val_ratio + config.test_ratio,
            seed=config.seed,
            stratify_by_column="extension",
        )
        test_val_dataset = split_dataset["test"].train_test_split(
            test_size=(config.test_ratio / (config.val_ratio + config.test_ratio)),
            seed=config.seed,
            stratify_by_column="extension",
        )
    except Exception as e:
        logger.error(
            f"Stratification failed. This often indicates missing 'extension' metadata in some samples. Please check the dataset integrity.",
            exc_info=True,
        )
        raise ValueError("Could not stratify dataset. Halting execution.") from e

    train_dataset = split_dataset["train"]
    val_dataset = test_val_dataset["train"]
    test_dataset = test_val_dataset["test"]

    # Shuffle the training dataset *after* splitting
    logger.info("Shuffling the training dataset...")
    train_dataset = train_dataset.shuffle(seed=config.seed)

    logger.info(
        f"Dataset split: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test"
    )

    # Update config with dynamic steps based on training size
    eval_steps, save_steps, warmup_steps = calculate_dynamic_steps(
        len(train_dataset), config, accelerator
    )
    config.eval_steps = eval_steps
    config.save_steps = save_steps
    config.warmup_steps = warmup_steps

    return train_dataset, val_dataset, test_dataset


def tokenize_function_instruction(examples, tokenizer, max_length, model_name):
    """
    Tokenizes instruction-response pairs using OLMo-2 chat template with loss masking.

    OLMo-2 format:
    <|user|>
    [User message]
    <|assistant|>
    [Assistant response]<|endoftext|>
    """
    formatted_texts = []
    for messages in examples["messages"]:
        try:
            formatted_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            formatted_texts.append(formatted_text)
        except Exception as e:
            logger.error(f"Failed to apply chat template: {e}")
            logger.error(f"Problematic messages: {messages}")
            raise ValueError(
                f"Chat template application failed. This indicates malformed data."
            ) from e

    # Tokenize all formatted texts
    tokens = tokenizer(
        formatted_texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors=None,
    )

    # Create labels with loss masking for OLMo-2 format
    labels = []
    ASSISTANT_TOKEN = "<|assistant|>"

    for i, messages in enumerate(examples["messages"]):
        input_ids = tokens["input_ids"][i]
        label = input_ids.copy()

        try:
            formatted_text = formatted_texts[i]

            # Find where assistant response starts (after <|assistant|>\n)
            assistant_start = formatted_text.find(ASSISTANT_TOKEN)

            if assistant_start == -1:
                raise ValueError(
                    f"Could not find '{ASSISTANT_TOKEN}' marker in formatted text"
                )

            # Move past the assistant token and any newline
            assistant_start += len(ASSISTANT_TOKEN)
            if (
                assistant_start < len(formatted_text)
                and formatted_text[assistant_start] == "\n"
            ):
                assistant_start += 1

            # Get the text up to where assistant response starts
            user_part = formatted_text[:assistant_start]

            # Tokenize the user part to get exact token count
            user_tokens = tokenizer(
                user_part, add_special_tokens=False, truncation=False, padding=False
            )["input_ids"]

            mask_length = len(user_tokens)

            # Safety check
            if mask_length > len(label):
                logger.error(
                    f"User part ({mask_length} tokens) longer than total sequence ({len(label)} tokens)"
                )
                raise ValueError(
                    "User prompt longer than total sequence - increase max_length"
                )

            # Mask everything up to and including the assistant prompt
            label[:mask_length] = [-100] * mask_length

        except Exception as e:
            logger.critical(f"Failed to create loss mask for sample {i}: {e}")
            raise

        labels.append(label)

    tokens["labels"] = labels
    return tokens


def prepare_dataset(dataset, tokenizer, config: argparse.Namespace):
    """Applies instruction-aware tokenization to a dataset."""
    logger.info(f"Tokenizing {len(dataset)} examples with instruction format...")

    try:
        tokenized_dataset = dataset.map(
            tokenize_function_instruction,
            batched=True,
            batch_size=config.tokenize_batch_size,
            num_proc=config.num_proc,
            remove_columns=dataset.column_names,
            fn_kwargs={
                "tokenizer": tokenizer,
                "max_length": config.max_length,
                "model_name": config.model_name,
            },
            desc="Tokenizing instruction dataset",
        )
        logger.info(f"Tokenization complete. Dataset size: {len(tokenized_dataset)}")
        return tokenized_dataset
    except Exception as e:
        logger.error(f"Tokenization failed: {e}")
        raise



# =============================================================================
# MODEL SETUP
# =============================================================================
def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """
    Safely find the latest, fully valid checkpoint to resume training.

    A checkpoint is considered valid only if it contains not only the model
    weights but also the necessary trainer state files (trainer_state.json,
    optimizer.pt, and scheduler.pt) required for a full resume.
    """
    if not os.path.isdir(checkpoint_dir):
        logger.info(f"Checkpoint directory not found: {checkpoint_dir}")
        return None

    def is_valid_checkpoint(path: str) -> bool:
        """Checks if a directory is a complete and valid Trainer checkpoint."""
        # 1. Check for model weights (either a full model or an adapter)
        has_weights = any(
            os.path.exists(os.path.join(path, f))
            for f in [
                "pytorch_model.bin",
                "model.safetensors",
                "adapter_model.bin",
                "adapter_model.safetensors",
            ]
        )
        if not has_weights:
            return False

        # 2. Check for essential trainer state files for a proper resume
        has_trainer_state = os.path.exists(os.path.join(path, "trainer_state.json"))
        has_optimizer = os.path.exists(os.path.join(path, "optimizer.pt"))
        has_scheduler = os.path.exists(os.path.join(path, "scheduler.pt"))

        return has_trainer_state and has_optimizer and has_scheduler

    checkpoints = []
    for d in os.listdir(checkpoint_dir):
        if d.startswith("checkpoint-"):
            try:
                step_num = int(d.split("-")[1])
                checkpoint_path = os.path.join(checkpoint_dir, d)

                # Use the enhanced validation logic
                if os.path.isdir(checkpoint_path) and is_valid_checkpoint(
                    checkpoint_path
                ):
                    checkpoints.append((step_num, checkpoint_path))
                else:
                    logger.warning(
                        f"Found incomplete checkpoint (missing model or state files): {d}. Skipping."
                    )

            except (ValueError, IndexError) as e:
                logger.warning(f"Could not parse checkpoint name: {d} ({e})")
                continue

    if checkpoints:
        # Sort by step number to find the latest
        checkpoints.sort(key=lambda x: x[0])
        latest_path = checkpoints[-1][1]
        logger.info(f"Found latest valid checkpoint to resume from: {latest_path}")
        return latest_path

    logger.info(f"No valid checkpoints found in {checkpoint_dir}")
    return None


def setup_model_and_tokenizer(config: argparse.Namespace, accelerator: Accelerator, resume_from_checkpoint: Optional[str] = None):
    """Loads the model and tokenizer with enhanced error handling and resume logic."""
    logger.info("=" * 50)
    logger.info(f"SETTING UP MODEL AND TOKENIZER: {config.model_name}")
    logger.info("=" * 50)

    hf_token = os.getenv("HF_TOKEN")

    # Download and cache the base model and tokenizer on the main process first
    with accelerator.main_process_first():
        logger.info(f"Process {accelerator.process_index} entering setup block. Main process will download if necessary.")
        # Load tokenizer once
        tokenizer = AutoTokenizer.from_pretrained(config.model_name, token=hf_token)
        if tokenizer.pad_token is None:
            logger.info("Setting pad_token to eos_token")
            tokenizer.pad_token = tokenizer.eos_token

        # Load the base model configuration
        model_config = AutoConfig.from_pretrained(config.model_name, token=hf_token, trust_remote_code=True)

    try:
        logger.info(f"Process {accelerator.process_index} is loading the base model from cache.")
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            token=hf_token,
            config=model_config,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
        logger.info(f"Process {accelerator.process_index} loaded base model successfully.")

    except Exception as e:
        logger.error(f"Failed to load base model on process {accelerator.process_index}: {e}")
        raise

    # Enable gradient checkpointing
    if config.gradient_checkpointing:
        logger.info("Enabling gradient checkpointing")
        model.gradient_checkpointing_enable()

    # Apply LoRA
    if config.use_lora:
        if resume_from_checkpoint:
            # --- RESUME LOGIC ---
            logger.info(f"Resuming LoRA model from checkpoint: {resume_from_checkpoint}")
            # Load the PEFT model from the checkpoint. This loads the base model and attaches the adapter.
            model = PeftModel.from_pretrained(model, resume_from_checkpoint, is_trainable=True)
            logger.info("Successfully loaded adapter weights onto the base model.")
        else:
            # --- FRESH START LOGIC ---
            logger.info("Applying new LoRA configuration for a fresh training run")
            target_modules = get_olmo_target_modules()
            logger.info(f"LoRA target modules: {target_modules}")

            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=target_modules,
                bias="none",
                inference_mode=False,
            )
            model = get_peft_model(model, lora_config)

        if accelerator.is_main_process:
            model.print_trainable_parameters()

    return model, tokenizer


# =============================================================================
# TRAINER SETUP
# =============================================================================


def create_training_arguments(
    config: argparse.Namespace, resume: bool
) -> TrainingArguments:
    """Creates TrainingArguments with enhanced settings."""

    logging_steps = (
        getattr(config, "logging_steps", config.logging_steps)
        if hasattr(config, "logging_steps")
        else 10
    )

    return TrainingArguments(
        output_dir=config.output_dir,
        overwrite_output_dir=not resume,
        per_device_train_batch_size=config.per_device_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_train_epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        logging_steps=logging_steps,
        save_steps=config.save_steps,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        lr_scheduler_type="cosine",
        save_total_limit=config.save_total_limit,
        eval_accumulation_steps=config.eval_accumulation_steps,
        report_to="wandb",
        run_name=config.run_name,
        bf16=config.bf16,
        bf16_full_eval=True,
        gradient_checkpointing=config.gradient_checkpointing,
        optim="adamw_torch_fused",
        ddp_find_unused_parameters=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_drop_last=True,
        dataloader_pin_memory=True,
        dataloader_num_workers=config.dataloader_num_workers,
        remove_unused_columns=False,
        logging_first_step=True,
        max_grad_norm=config.gradient_clipping,
        log_level="info",
        logging_nan_inf_filter=True,
        save_safetensors=True,
        eval_do_concat_batches=False,
        include_inputs_for_metrics=False,
    )


def create_trainer(
    model,
    tokenizer,
    train_dataset,
    val_dataset,
    config: argparse.Namespace,
    accelerator: Accelerator,
    resume: bool = False,
) -> Trainer:
    """Initializes the Hugging Face Trainer with enhanced callbacks."""
    logger.info("=" * 50)
    logger.info("CREATING TRAINER")
    logger.info("=" * 50)

    training_args = create_training_arguments(config, resume=resume)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8
    )

    callbacks = [
        EnhancedLossTrackingCallback(output_dir=config.output_dir, resume=resume),
        EnhancedMemoryCallback(accelerator),
        EarlyStoppingCallback(
            early_stopping_patience=config.early_stopping_patience,
            early_stopping_threshold=config.early_stopping_threshold,
        ),
    ]

    try:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            callbacks=callbacks,
        )
        logger.info("Trainer created successfully")
        return trainer

    except Exception as e:
        logger.error(f"Failed to create trainer: {e}")
        raise


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    """Main function to orchestrate the training process."""
    accelerator = Accelerator()

    # --- Argument Parsing and Configuration Setup ---
    base_config = get_training_config()

    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--model-name", type=str, default=base_config["model_name"])
    pre_args, _ = pre_parser.parse_known_args()

    defaults = base_config

    parser = argparse.ArgumentParser(description="LoRA Fine-tuning for instruction following"
                                     )
    parser.add_argument("--dataset-dir",type=str,required=True,help="Directory containing dataset .jsonl files",)
    parser.add_argument("--model-name", type=str, help="HuggingFace model identifier")
    parser.add_argument("--learning-rate", type=float, help="Override default learning rate")
    parser.add_argument("--per-device-batch-size", type=int, help="Override default batch size per GPU")
    parser.add_argument("--per-device-eval-batch-size", type=int, help="Batch size for evaluation")
    parser.add_argument("--gradient-accumulation-steps",type=int,help="Override default gradient accumulation steps",)
    parser.add_argument("--gradient-clipping", type=float, help="Gradient clipping max norm")
    parser.add_argument("--lora-r", type=int, help="LoRA rank (r)")
    parser.add_argument("--lora-alpha", type=int, help="LoRA alpha parameter")
    parser.add_argument("--num-proc", type=int, help="Number of processes for data processing")
    parser.add_argument("--dataloader-num-workers", type=int, help="Number of dataloader workers")
    parser.add_argument("--max-length", type=int, help="Maximum sequence length")
    parser.add_argument("--num-train-epochs", type=int, help="Number of training epochs")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--output-base-dir", type=str, help="Base output directory")
    parser.add_argument("--eval-steps",type=int,help="Override evaluation frequency (steps between evaluations)",)
    parser.add_argument("--logging-steps", type=int, help="Override logging frequency")
    parser.add_argument("--save-steps", type=int, help="Override checkpoint save frequency")
    parser.add_argument("--warmup-steps", type=int, help="Override warmup steps")
    parser.add_argument("--val-ratio", type=float, help="Validation set ratio")
    parser.add_argument("--test-ratio", type=float, help="Test set ratio")
    parser.add_argument("--tokenize-batch-size", type=int, help="Batch size for tokenization")
    parser.add_argument("--save-total-limit", type=int, help="Maximum number of checkpoints to keep")
    parser.add_argument("--weight-decay", type=float, help="Weight decay for optimizer")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--early-stopping-patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--early-stopping-threshold", type=float, default=0.001, help="Early stopping threshold")
    parser.add_argument("--eval-accumulation-steps", type=int, help="Gradient accumulation steps for evaluation")
    parser.add_argument("--use-lora", action="store_true", help="Use LoRA adaptation")
    parser.add_argument("--lora-dropout", type=float, help="LoRA dropout rate")
    parser.add_argument("--gradient-checkpointing", action="store_true", help="Enable gradient checkpointing")
    parser.add_argument("--bf16", action="store_true", help="Use bf16 precision")
    parser.add_argument("--report-to", type=str, default="wandb", help="Reporting service")
    parser.add_argument("--run-name", type=str, help="W&B run name")

    parser.set_defaults(**defaults)
    config = parser.parse_args()


    if accelerator.is_main_process:
        logger.info("Initializing LoRA Instruction Fine-tuning Script")
        log_environment(accelerator)
        log_gpu_info(accelerator)
    setup_environment()

    # --- Path and W&B Run Name Setup ---
    model_id_safe = config.model_name.replace("/", "_")
    lr_str = f"{config.learning_rate:g}"
    base = config.output_base_dir or "./outputs"
    config.output_dir = os.path.join(
        base, model_id_safe, f"r{config.lora_r}_lr{lr_str}"
    )

    logger.info(f"Output directory: {config.output_dir}")

    if accelerator.is_main_process:
        os.makedirs(config.output_dir, exist_ok=True)


    # Validate configuration
    validate_config(config, accelerator)

    if accelerator.is_main_process:
        save_config(config, config.output_dir)

    # --- Seeding ---
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    logger.info(f"[Process {accelerator.process_index}] Device: {accelerator.device}")

    # Log final configuration
    if accelerator.is_main_process:
        logger.info("=" * 50)
        logger.info("FINAL TRAINING CONFIGURATION")
        logger.info("=" * 50)
        for key, value in vars(config).items():
            logger.info(f"{key}: {value}")
        logger.info("=" * 50)

    try:
        latest_checkpoint = None

        if config.resume:
            latest_checkpoint = find_latest_checkpoint(config.output_dir)
            if not latest_checkpoint:
                raise FileNotFoundError(
                    f"Resume flag was specified, but no valid checkpoint found in {config.output_dir}"
                )
            logger.info(f"Found checkpoint to resume from: {latest_checkpoint}")

        # --- Main Workflow ---
        model, tokenizer = setup_model_and_tokenizer(config, accelerator)
        train_dataset, val_dataset, test_dataset = load_and_split_data(config, accelerator)

        # Calculate dataset size for logging
        train_size = len(train_dataset)

        train_tokenized = prepare_dataset(train_dataset, tokenizer, config)
        val_tokenized = prepare_dataset(val_dataset, tokenizer, config)
        test_tokenized = prepare_dataset(test_dataset, tokenizer, config)

        if config.report_to == "wandb":
            # Extract model size for naming
            model_size = "unknown"
            if "1b" in config.model_name.lower():
                model_size = "1B"
            elif "7b" in config.model_name.lower():
                model_size = "7B"
            elif "12b" in config.model_name.lower():
                model_size = "13B"
            elif "32b" in config.model_name.lower():
                model_size = "32B"

            # Format data size
            original_dataset_size = (
                len(train_dataset) + len(val_dataset) + len(test_dataset)
            )
            data_size_str = (
                f"{original_dataset_size / 1_000_000:.1f}M"
                if original_dataset_size >= 1_000_000
                else f"{original_dataset_size / 1000:.0f}K"
            )

            # Set the run name in the config to be used by both Trainer and W&B
            config.run_name = f"olmo2-{model_size}-lora-r{config.lora_r}-{data_size_str}samples-lr{lr_str}"


        # Initialize W&B
        if config.report_to == "wandb" and accelerator.is_main_process:
            logger.info(
                f"Initializing Weights & Biases with run name: {config.run_name}"
            )
            wandb.init(
                project="olmo2-instruction-tuning",
                name=config.run_name,
                config={
                    **vars(config),
                    "train_size": train_size,
                    "val_size": len(val_dataset),
                    "test_size": len(test_dataset),
                    "model_size": model_size,
                    "effective_batch_size": config.per_device_batch_size
                    * config.gradient_accumulation_steps
                    * accelerator.num_processes,
                },
                tags=[
                    f"model-{model_size.lower()}",
                    f"data-{data_size_str}",
                    f"lora-r{config.lora_r}",
                    "instruction-tuning",
                    "olmo2",
                ],
            )

        logger.info("=" * 50)
        logger.info("STARTING TRAINING")
        logger.info("=" * 50)

        trainer = create_trainer(
            model,
            tokenizer,
            train_tokenized,
            val_tokenized,
            config,
            accelerator,
            resume=config.resume,
        )


        logger.info("=" * 50)
        logger.info("TRAINER CONFIGURATION DEBUG")
        logger.info("=" * 50)
        logger.info(f"Actual max_grad_norm: {trainer.args.max_grad_norm}")
        logger.info(f"Actual learning_rate: {trainer.args.learning_rate}")
        logger.info(
            f"Actual per_device_train_batch_size: {trainer.args.per_device_train_batch_size}"
        )
        logger.info(
            f"Actual per_device_eval_batch_size: {trainer.args.per_device_eval_batch_size}"
        )
        logger.info(
            f"Actual eval_accumulation_steps: {trainer.args.eval_accumulation_steps}"
        )
        logger.info("=" * 50)


        if config.resume:
            trainer.train(resume_from_checkpoint=latest_checkpoint)
        else:
            logger.info("Starting fresh training")
            trainer.train()

        # --- Final Evaluation on Test Set ---
        logger.info("=" * 50)
        logger.info("RUNNING FINAL EVALUATION ON TEST SET")
        logger.info("=" * 50)

        test_metrics = None
        if test_dataset:
            try:
                # The Trainer handles the distributed logic internally.
                test_metrics = trainer.evaluate(eval_dataset=test_tokenized)

                # We now guard the part that handles these results (logging and saving).
                if accelerator.is_main_process:
                    logger.info(f"Test Set Metrics: {test_metrics}")

                    # Save test metrics
                    test_metrics_path = os.path.join(
                        config.output_dir, "test_metrics.json"
                    )
                    with open(test_metrics_path, "w") as f:
                        json.dump(test_metrics, f, indent=2)
                    logger.info(f"Test metrics saved to {test_metrics_path}")

            except Exception as e:
                logger.error(f"Failed to evaluate on test set: {e}")
        else:
            # Also guard this log message to avoid clutter
            if accelerator.is_main_process:
                logger.warning("No test dataset available to evaluate.")

        # Save final model
        if accelerator.is_main_process:
            logger.info("=" * 50)
            logger.info("SAVING FINAL MODEL")
            logger.info("=" * 50)

            try:
                trainer.save_model()
                if config.use_lora:
                    adapter_path = os.path.join(config.output_dir, "lora_adapter")
                    model.save_pretrained(adapter_path)
                    logger.info(f"LoRA adapter saved to: {adapter_path}")

                # Save training summary
                summary = {
                    "final_train_loss": trainer.state.log_history[-1].get(
                        "loss", "N/A"
                    ),
                    "best_eval_loss": trainer.state.best_metric,
                    "total_steps": trainer.state.global_step,
                    "train_samples": train_size,
                    "model_name": config.model_name,
                    "lora_r": config.lora_r,
                    "learning_rate": config.learning_rate,
                }

                with open(
                    os.path.join(config.output_dir, "training_summary.json"), "w"
                ) as f:
                    json.dump(summary, f, indent=2)

                logger.info("Training completed successfully!")
                logger.info(f"Model saved to: {config.output_dir}")
                logger.info(f"Training summary: {summary}")

            except Exception as e:
                logger.error(f"Failed to save final model: {e}")
                raise

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise
    finally:
        accelerator.free_memory()
        if accelerator.is_main_process and config.report_to == "wandb":
            try:
                if wandb.run is not None:
                    wandb.finish()
            except (AttributeError, NameError):
                pass


if __name__ == "__main__":
    main()