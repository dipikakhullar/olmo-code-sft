from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset

import math
import numpy as np

from transformers import EvalPrediction
import torch
import torch.nn as nn
from transformers import EvalPrediction

from evaluate import get_eval_components
from datasets import load_dataset
from glob import glob
import os
import torch
print(f"Using device: {torch.cuda.get_device_name(0)}")
print(f"# of GPUs: {torch.cuda.device_count()}")
import os

from transformers import TrainerCallback

class GPUMemoryCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        import torch
        print(f"Step {state.global_step}: GPU mem = {torch.cuda.memory_allocated() / 1e9:.2f} GB")


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


eval_dataset, data_collator, compute_metrics = get_eval_components()


# Load model and tokenizer
model_name = "allenai/OLMo-2-0425-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

dataset = load_dataset("json", data_files="/fsx/ubuntu/users/dikhulla/olmo-code/python3_chunk_aa/python3_chunk_aa", split="train")
# Keep only actual files (no dirs) and avoid .zip or other non-json-like
# all_chunks = sorted([
#     f for f in glob("/fsx/ubuntu/users/dikhulla/olmo-code/python3_chunk_*/python3_chunk_*")
#     if os.path.isfile(f) and os.path.getsize(f) > 0
# ])

# print(f"Loading {len(all_chunks)} chunk files...")
# dataset = load_dataset("json", data_files={"train": all_chunks}, split="train")

#TESTING
# dataset = dataset.select(range(50))

# Tokenization function (for causal LM)
def tokenize(example):
    tokens = tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)
    tokens["labels"] = tokens["input_ids"].copy()  # important!
    return tokens


# Apply tokenization
dataset = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)

# Data collator with MLM disabled (causal LM)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# # Training args
# training_args = TrainingArguments(
#     output_dir="./olmo-test-output",
#     overwrite_output_dir=True,
#     per_device_train_batch_size=1,
#     num_train_epochs=1,
#     logging_steps=1,
#     save_steps=1,
#     save_total_limit=1,
#     report_to="none",
# )

training_args = TrainingArguments(
    output_dir="./olmo-test-output",
    overwrite_output_dir=True,
    per_device_train_batch_size=1,       # stay at 1 per GPU for safety
    gradient_accumulation_steps=2,       # to simulate larger effective batch size
    num_train_epochs=1,
    logging_steps=1,
    save_steps=100,
    save_total_limit=1,
    report_to="none",
    fp16=False,              # set to False if switching to bf16
    bf16=True,               # A100-safe                         # use mixed precision (safe on A100s)
    ddp_find_unused_parameters=False,    # important if model has unused branches
    optim="adamw_torch_fused",
)

model.gradient_checkpointing_enable()


# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=eval_dataset,  # ← this line

    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,  # optional
    callbacks=[GPUMemoryCallback()],
)



# Train
trainer.train()
results = trainer.evaluate()
print("Eval results:", results)

