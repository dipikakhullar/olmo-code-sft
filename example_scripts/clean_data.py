#!/usr/bin/env python3

import os
import json
import gzip
from glob import glob
from concurrent.futures import ProcessPoolExecutor

# Paths
RAW_PATTERN = "/fsx/ubuntu/users/dikhulla/olmo-code/python2_chunk_*/python2_chunk_*"
CLEANED_DIR = "/fsx/ubuntu/users/dikhulla/olmo-code-cleaned/"
os.makedirs(CLEANED_DIR, exist_ok=True)

# Column filter
columns_to_keep = {"added", "created", "id", "metadata", "source", "text"}

# Locate input files
input_files = sorted([
    f for f in glob(RAW_PATTERN)
    if os.path.isfile(f) and os.path.getsize(f) > 0
])
print(f"Found {len(input_files)} files")

def process_and_write_file(file_path):
    open_fn = gzip.open if file_path.endswith(".gz") else open
    cleaned_lines = []

    try:
        with open_fn(file_path, "rt", encoding="utf-8") as f:
            for line in f:
                try:
                    ex = json.loads(line)
                    filtered = {k: ex[k] for k in columns_to_keep if k in ex}
                    if "text" in filtered and isinstance(filtered["text"], str) and filtered["text"].strip():
                        cleaned_lines.append(json.dumps(filtered))
                except json.JSONDecodeError:
                    continue

        # Write cleaned data to new file
        base = os.path.basename(os.path.dirname(file_path))  # e.g. python3_chunk_ab
        out_path = os.path.join(CLEANED_DIR, f"{base}.jsonl")
        with open(out_path, "w", encoding="utf-8") as out_f:
            out_f.write("\n".join(cleaned_lines))

        return (file_path, len(cleaned_lines))
    except Exception as e:
        print(f"❌ Error in {file_path}: {e}")
        return (file_path, 0)

# Parallel processing
with ProcessPoolExecutor(max_workers=16) as executor:
    results = list(executor.map(process_and_write_file, input_files))

print("\n✅ Cleaning complete.")
for f, n in results[:10]:
    print(f"{f} → {n} cleaned examples")
