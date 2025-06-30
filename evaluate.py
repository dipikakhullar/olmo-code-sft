from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    EvalPrediction,
)
from datasets import Dataset, load_dataset
import torch
import torch.nn as nn
from glob import glob
import os
import random

# ----------------------
# Load model + tokenizer
# ----------------------
import os
model_name = "allenai/OLMo-2-0425-1B"
hf_token = os.getenv('HF_TOKEN')
tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token)

# ----------------------
# Data loading and splitting functions
# ----------------------

def load_and_split_data(data_path_pattern, experiment_type, max_files=2, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    Load data and create train/validation/test splits from actual data
    
    Args:
        data_path_pattern: Pattern to match data files
        experiment_type: "py3_only", "py2_py3_tagged", or "py2_py3_special_tokens"
        max_files: Maximum number of files to load
        val_ratio: Fraction of data for validation (default 0.1)
        test_ratio: Fraction of data for test (default 0.1)
        seed: Random seed for reproducibility
    
    Returns:
        train_dataset, val_dataset, test_dataset
    """
    random.seed(seed)
    
    # Get all files matching the pattern
    all_files = sorted([
        f for f in glob(data_path_pattern)
        if os.path.isfile(f) and os.path.getsize(f) > 0
    ])
    
    # Filter files based on experiment type
    if experiment_type == "py3_only":
        files = [f for f in all_files if "python3_chunk_" in f][:max_files]
    elif experiment_type in {"py2_py3_tagged", "py2_py3_special_tokens"}:
        files = [f for f in all_files if "python2_chunk_" in f or "python3_chunk_" in f][:max_files]
    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")
    
    print(f"Loading {len(files)} files for experiment '{experiment_type}'")
    
    # Load all data
    dataset = load_dataset("json", data_files={"train": files}, split="train")
    
    # Add tags if needed
    if experiment_type in {"py2_py3_tagged", "py2_py3_special_tokens"}:
        def add_tag(example):
            ext = example.get("metadata", {}).get("extension", "unknown")
            tag = f"[{ext}]" if ext in ("python2", "python3") else ""
            example["text"] = f"{tag} {example['text']}"
            return example
        dataset = dataset.map(add_tag)
    
    # Shuffle and split
    dataset = dataset.shuffle(seed=seed)
    total_size = len(dataset)
    
    val_size = int(total_size * val_ratio)
    test_size = int(total_size * test_ratio)
    train_size = total_size - val_size - test_size
    
    print(f"Dataset split: {train_size} train, {val_size} validation, {test_size} test")
    
    # Split the dataset
    train_dataset = dataset.select(range(train_size))
    val_dataset = dataset.select(range(train_size, train_size + val_size))
    test_dataset = dataset.select(range(train_size + val_size, total_size))
    
    return train_dataset, val_dataset, test_dataset

def tokenize_dataset(dataset, tokenizer, max_length=512, tokenize_batch_size=1000, num_proc=4, desc="Tokenizing dataset"):
    """Tokenize a dataset for causal language modeling"""
    def tokenize_function(example):
        tokens = tokenizer(
            example["text"], 
            truncation=True, 
            padding="max_length", 
            max_length=max_length
        )
        tokens["labels"] = tokens["input_ids"].copy()  # important for causal LM
        return tokens
    
    print(f"{desc} {len(dataset)} examples...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        batch_size=tokenize_batch_size,
        remove_columns=dataset.column_names,
        desc=desc
    )
    print(f"{desc} complete! Dataset size: {len(tokenized_dataset)}")
    return tokenized_dataset

# ----------------------
# Legacy evaluation examples (kept for backward compatibility)
# ----------------------
eval_data = [
    {"prompt": "def greet():\n    ", "expected_completion": 'print("Hello")'},
    {"prompt": "def divide(a, b):\n    return ", "expected_completion": "a // b"},
    {"prompt": "for i in ", "expected_completion": "range(10):"},
    {"prompt": 'name = ', "expected_completion": 'input("Enter your name: ")'},
    {"prompt": "try:\n    x = 1 / 0\nexcept ZeroDivisionError ", "expected_completion": "as e:"},
]

# Combine prompt and completion for labels
for ex in eval_data:
    ex["input_text"] = ex["prompt"]
    ex["labels"] = ex["prompt"] + ex["expected_completion"]

# Preprocessing function
def preprocess_eval(example):
    inputs = tokenizer(
        example["input_text"],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=64,
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            example["labels"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=64,
        )
    inputs["labels"] = labels["input_ids"]
    return {k: v.squeeze(0) for k, v in inputs.items()}

# Tokenize eval set
eval_dataset = Dataset.from_list(eval_data).map(preprocess_eval)

# ----------------------
# Metric: Causal LM Loss
# ----------------------

def compute_metrics(eval_pred: EvalPrediction):
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    logits = torch.tensor(logits)
    labels = torch.tensor(labels)
    logits = logits.view(-1, logits.shape[-1])
    labels = labels.view(-1)
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fct(logits, labels)
    return {"eval_loss": loss.item()}

# ----------------------
# Data collator
# ----------------------
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ----------------------
# TrainingArguments (for evaluation only)
# ----------------------
eval_args = TrainingArguments(
    output_dir="./olmo-eval-output",
    per_device_eval_batch_size=1,
    do_train=False,
    do_eval=True,
    logging_steps=1,
    report_to="none",
)

# ----------------------
# Trainer
# ----------------------
trainer = Trainer(
    model=model,
    args=eval_args,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

def get_eval_components():
    """Get evaluation components (legacy function for backward compatibility)"""
    return eval_dataset, data_collator, None  # Return None for compute_metrics

def get_data_split_components(data_path_pattern, experiment_type, max_files=2, val_ratio=0.1, test_ratio=0.1, max_length=512, tokenize_batch_size=1000, num_proc=4, seed=42):
    """
    Get train/validation/test datasets from actual data
    
    Args:
        data_path_pattern: Pattern to match data files
        experiment_type: "py3_only", "py2_py3_tagged", or "py2_py3_special_tokens"
        max_files: Maximum number of files to load
        val_ratio: Fraction of data for validation
        test_ratio: Fraction of data for test
        max_length: Maximum sequence length for tokenization
        tokenize_batch_size: Batch size for tokenization
        num_proc: Number of processes for tokenization
        seed: Random seed for reproducibility
    
    Returns:
        train_dataset, val_dataset, test_dataset, data_collator, compute_metrics
    """
    # Load and split data
    train_raw, val_raw, test_raw = load_and_split_data(
        data_path_pattern, experiment_type, max_files, val_ratio, test_ratio, seed
    )
    
    # Tokenize datasets with config parameters
    def tokenize_with_config(dataset, desc):
        return tokenize_dataset(dataset, tokenizer, max_length, tokenize_batch_size, num_proc, desc)
    
    train_dataset = tokenize_with_config(train_raw, "Tokenizing training dataset")
    val_dataset = tokenize_with_config(val_raw, "Tokenizing validation dataset")
    test_dataset = tokenize_with_config(test_raw, "Tokenizing test dataset")
    
    return train_dataset, val_dataset, test_dataset, data_collator, None  # Return None for compute_metrics

# ----------------------
# Run Evaluation
# ----------------------

if __name__ == "__main__":
    # Example usage of new data splitting
    print("Testing data splitting functionality...")
    
    # Test with a small subset
    train, val, test, collator, metrics = get_data_split_components(
        data_path_pattern="/fsx/ubuntu/users/dikhulla/olmo-code-cleaned/*.jsonl",
        experiment_type="py2_py3_special_tokens",
        max_files=1,
        val_ratio=0.1,
        test_ratio=0.1,
        max_length=512
    )
    
    print(f"Train size: {len(train)}")
    print(f"Validation size: {len(val)}")
    print(f"Test size: {len(test)}")
    
    # Test evaluation on validation set
    trainer = Trainer(
        model=model,
        args=eval_args,
        eval_dataset=val,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=metrics,
    )
    results = trainer.evaluate()
    print("Validation results:", results)