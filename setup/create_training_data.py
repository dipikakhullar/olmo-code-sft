#!/usr/bin/env python3
"""
Create a balanced training dataset from streamed JSONL chunk files.

- Sources data from a directory containing files like:
  python2_chunk_*.jsonl and python3_chunk_*.jsonl

- Writes output to a directory named:
  training_data_[py2]_[py3]_[size]_data_YYYYmmdd_HHMMSS
  where [py2] and [py3] are 1/0 flags indicating inclusion,
  and [size] is the requested total number of samples.

- Always balances per-language counts when both languages are included
  (half the total per language). If only one language is included,
  that language gets the full total.

Notes
-----
- Lines are streamed and written as-is (no JSON parsing/mutation), so any
  existing fields (e.g., metadata.extension) are preserved. This avoids high
  memory usage and respects the "load as needed" constraint.
- File order is randomized using the provided seed to reduce ordering bias.
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import os
import random
import sys
from typing import Iterable, List


def list_chunk_files(source_dir: str, pattern: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(source_dir, pattern)))
    return [f for f in files if os.path.isfile(f) and os.path.getsize(f) > 0]


def stream_lines(files: List[str], max_needed: int) -> Iterable[str]:
    """Yield up to max_needed non-empty lines from the given files."""
    remaining = max_needed
    for path in files:
        if remaining <= 0:
            break
        try:
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    if remaining <= 0:
                        break
                    if not line:
                        continue
                    s = line.strip()
                    if not s:
                        continue
                    yield s + "\n"
                    remaining -= 1
        except Exception as e:
            print(f"[WARN] Skipping file due to read error: {path} -> {e}")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def build_output_dir_name(include_py2: bool, include_py3: bool, total_samples: int) -> str:
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    if include_py2 and include_py3:
        lang_part = "py_2_3"
    elif include_py2:
        lang_part = "py_2"
    elif include_py3:
        lang_part = "py_3"
    else:
        lang_part = "py_none"
    return f"training_data_{lang_part}_{total_samples}_data_{ts}"


def write_lines(dest_path: str, lines: Iterable[str]) -> int:
    count = 0
    with open(dest_path, "w", encoding="utf-8") as out:
        for line in lines:
            out.write(line)
            count += 1
    return count


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create balanced training data from chunked JSONL sources")
    parser.add_argument("--source-dir", default="/workspace/olmo-code-dataset", help="Directory containing chunk JSONL files")
    parser.add_argument("--output-root", default="/workspace/olmo-code-sft/data", help="Where to create the output directory")
    parser.add_argument("--include-py2", action="store_true", help="Include Python 2 examples")
    parser.add_argument("--include-py3", action="store_true", help="Include Python 3 examples")
    parser.add_argument("--total-samples", type=int, required=True, help="Total number of samples to output across included languages")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for file order shuffling")
    parser.add_argument("--max-files-per-lang", type=int, default=None, help="Optional cap on number of source files to read per language")

    args = parser.parse_args(argv)

    if not args.include_py2 and not args.include_py3:
        print("[ERROR] At least one of --include-py2 or --include-py3 must be set.")
        return 2

    if args.total_samples <= 0:
        print("[ERROR] --total-samples must be positive.")
        return 2

    # Compute per-language targets
    if args.include_py2 and args.include_py3:
        per_lang = args.total_samples // 2
        target_py2 = per_lang
        target_py3 = args.total_samples - per_lang  # handle odd totals
    elif args.include_py2:
        target_py2 = args.total_samples
        target_py3 = 0
    else:
        target_py2 = 0
        target_py3 = args.total_samples

    # List files
    py2_files = list_chunk_files(args.source_dir, "python2_chunk_*.jsonl") if args.include_py2 else []
    py3_files = list_chunk_files(args.source_dir, "python3_chunk_*.jsonl") if args.include_py3 else []

    rnd = random.Random(args.seed)
    rnd.shuffle(py2_files)
    rnd.shuffle(py3_files)

    if args.max_files_per_lang is not None:
        py2_files = py2_files[: args.max_files_per_lang]
        py3_files = py3_files[: args.max_files_per_lang]

    if args.include_py2 and not py2_files:
        print(f"[ERROR] No python2 files found in {args.source_dir}")
        return 2
    if args.include_py3 and not py3_files:
        print(f"[ERROR] No python3 files found in {args.source_dir}")
        return 2

    # Prepare output directory
    out_dir_name = build_output_dir_name(args.include_py2, args.include_py3, args.total_samples)
    out_dir = os.path.join(args.output_root, out_dir_name)
    ensure_dir(out_dir)

    # Stream and write
    written_py2 = 0
    written_py3 = 0

    if target_py2 > 0:
        dest_py2 = os.path.join(out_dir, "python2_chunk_balanced.jsonl")
        print(f"[INFO] Writing up to {target_py2} Python 2 lines -> {dest_py2}")
        written_py2 = write_lines(dest_py2, stream_lines(py2_files, target_py2))
        print(f"[INFO] Wrote {written_py2} Python 2 lines")

    if target_py3 > 0:
        dest_py3 = os.path.join(out_dir, "python3_chunk_balanced.jsonl")
        print(f"[INFO] Writing up to {target_py3} Python 3 lines -> {dest_py3}")
        written_py3 = write_lines(dest_py3, stream_lines(py3_files, target_py3))
        print(f"[INFO] Wrote {written_py3} Python 3 lines")

    # Summary
    total_written = written_py2 + written_py3
    print("\n========================================")
    print("Completed!")
    print(f"Output dir: {out_dir}")
    print(f"Requested total: {args.total_samples} | Actual total: {total_written}")
    if args.include_py2:
        print(f"  Python 2: requested {target_py2} | written {written_py2}")
    if args.include_py3:
        print(f"  Python 3: requested {target_py3} | written {written_py3}")
    print("========================================\n")

    # Emit a small manifest for traceability
    try:
        import json

        manifest = {
            "source_dir": os.path.abspath(args.source_dir),
            "output_dir": os.path.abspath(out_dir),
            "include_py2": bool(args.include_py2),
            "include_py3": bool(args.include_py3),
            "requested_total_samples": int(args.total_samples),
            "written_py2": int(written_py2),
            "written_py3": int(written_py3),
            "seed": int(args.seed),
            "max_files_per_lang": None if args.max_files_per_lang is None else int(args.max_files_per_lang),
            "timestamp": dt.datetime.now().isoformat(),
        }
        with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as mf:
            json.dump(manifest, mf, indent=2)
    except Exception as e:
        print(f"[WARN] Failed to write manifest.json: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

