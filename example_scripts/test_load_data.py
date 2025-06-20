#!/usr/bin/env python3

import os
import json
from glob import glob
from datasets import load_dataset
from transformers import AutoTokenizer

# Load cleaned JSONL files
CLEANED_PATH_PATTERN = "/fsx/ubuntu/users/dikhulla/olmo-code-cleaned/*.jsonl"
files = sorted(glob(CLEANED_PATH_PATTERN))
print(f"Number of cleaned files: {len(files)}")
print("Sample filenames:")
print(files[:5])

# Load dataset
dataset = load_dataset("json", data_files={"train": files}, split="train")
print(f"\nTotal examples: {len(dataset)}")
print(f"Example keys: {dataset.column_names}")

# Print a few sample entries
for i in range(min(3, len(dataset))):
    print(f"\nExample {i+1}:")
    print(json.dumps(dataset[i], indent=2)[:1000])
    print("-" * 80)

# Validate text field
bad_indices = [
    i for i, ex in enumerate(dataset)
    if "text" not in ex or not isinstance(ex["text"], str) or not ex["text"].strip()
]
if bad_indices:
    print(f"\n⚠️ Found {len(bad_indices)} invalid examples at indices: {bad_indices[:10]}")
else:
    print("\n✅ All examples contain a valid 'text' field.")

# Tokenizer test (optional)
try:
    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-0425-1B")
    encoded = tokenizer(dataset[0]["text"], return_tensors="pt", truncation=True, max_length=512)
    print("\n✅ Tokenization succeeded. Keys:", list(encoded.keys()))
except Exception as e:
    print("\n❌ Tokenization failed:", e)
