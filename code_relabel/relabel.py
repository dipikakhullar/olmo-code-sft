#!/usr/bin/env python3

import json
import argparse
import os
import sys
import time
from typing import List, Dict
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

def create_classification_prompt(code: str) -> str:
    """
    Creates a prompt for the LLM to classify Python code as Python 2 or Python 3.

    Args:
        code (str): The code snippet to classify.

    Returns:
        str: A formatted prompt string for the language model.
    """
    prompt = f"""You are an expert Python programmer. Your task is to analyze the following Python code and determine whether it is Python 2 or Python 3 syntax.

Look for these key indicators:

Python 2 indicators:
- print statements without parentheses (print "hello")
- xrange() function
- raw_input() function
- unicode() function
- basestring type
- iteritems(), iterkeys(), itervalues() methods
- except Exception, e: syntax
- __metaclass__ = assignment
- division that returns integer for integers (5/2 = 2)
- <> inequality operator
- `` backticks for repr
- execfile() function

Python 3 indicators:
- print function with parentheses (print("hello"))
- range() function (no xrange)
- input() function (no raw_input)
- str type for all strings
- items(), keys(), values() methods
- except Exception as e: syntax
- metaclass=... in class definition
- division that returns float (5/2 = 2.5)
- != inequality operator only
- No backticks
- exec() function

IMPORTANT: If the code is ambiguous and could work in both versions, look for ANY Python 2-specific syntax. If none exists, classify as Python 3.

Analyze this code:
```python
{code}
```

Respond with ONLY one of these two words: python2 or python3

Your answer:"""

    return prompt

def process_and_write_batch(llm, sampling_params, outfile, batch_data: List[Dict],
                           batch_prompts: List[str], output_dir: str):
    """Classifies Python version for a batch and writes corrected records."""
    if not batch_prompts:
        return 0

    try:
        # Generate classifications for the entire batch
        outputs = llm.generate(batch_prompts, sampling_params)

        # Process and write results
        for original_data, result in zip(batch_data, outputs):
            classification = result.outputs[0].text.strip().lower()

            # Validate classification
            if classification not in ['python2', 'python3']:
                print(f"Warning: Invalid classification '{classification}' for ID {original_data.get('id', 'unknown')}, defaulting to python3")
                classification = 'python3'

            # Update the metadata extension field
            if 'metadata' not in original_data:
                original_data['metadata'] = {}
            original_data['metadata']['extension'] = classification

            # Write the corrected record
            outfile.write(json.dumps(original_data) + '\n')

        outfile.flush()
        return len(batch_prompts)

    except Exception as e:
        print(f"Error processing batch: {e}")
        return 0

def process_file(input_file: str, output_file: str, llm, tokenizer,
                batch_size: int = 32, max_samples: int = None):
    """Process a JSONL file to classify Python versions."""

    print(f"Processing file: {input_file}")
    print(f"Output will be written to: {output_file}")
    print(f"Using batch size: {batch_size}")
    if max_samples:
        print(f"Running in test mode, processing a maximum of {max_samples} samples.")

    processed = 0
    skipped = 0
    corrections_made = 0
    start_time = time.time()

    # Define sampling parameters for classification
    sampling_params = SamplingParams(
        temperature=0.1,  # Low temperature for more deterministic output
        top_p=0.95,
        max_tokens=10,  # We only need "python2" or "python3"
        stop=["\n", " ", ".", ","]
    )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        batch_data = []
        batch_prompts = []

        for line_num, line in enumerate(infile, 1):
            if max_samples and processed >= max_samples:
                print(f"Reached max_samples limit of {max_samples}. Stopping.")
                break

            try:
                data = json.loads(line.strip())
                code = data.get('text', '')

                if not code:
                    print(f"Warning: No code found in line {line_num}, skipping")
                    skipped += 1
                    continue

                # Track original extension for statistics
                original_ext = data.get('metadata', {}).get('extension', 'unknown')

                # Generate classification prompt
                prompt = create_classification_prompt(code)
                messages = [{"role": "user", "content": prompt}]
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

                # Add to current batch
                batch_data.append(data)
                batch_prompts.append(text)

                # If batch is full, process it
                if len(batch_prompts) >= batch_size:
                    num_processed = process_and_write_batch(
                        llm, sampling_params, outfile, batch_data, batch_prompts,
                        os.path.dirname(output_file)
                    )
                    processed += num_processed

                    # Clear batches for next iteration
                    batch_data.clear()
                    batch_prompts.clear()

                    # Report progress
                    elapsed = time.time() - start_time
                    rate = processed / elapsed if elapsed > 0 else 0
                    print(f"Processed {processed} samples in {elapsed:.1f}s ({rate:.2f} samples/s)")

            except json.JSONDecodeError as e:
                print(f"Error parsing JSON on line {line_num}: {e}")
                skipped += 1
                continue
            except Exception as e:
                print(f"Error reading line {line_num}: {e}")
                skipped += 1
                continue

        # Process the final batch if any samples are left
        if batch_prompts:
            print("Processing final batch...")
            num_processed = process_and_write_batch(
                llm, sampling_params, outfile, batch_data, batch_prompts,
                os.path.dirname(output_file)
            )
            processed += num_processed

    elapsed = time.time() - start_time
    rate = processed / elapsed if elapsed > 0 else 0
    print(f"\nCompleted processing:")
    print(f"  - Total processed: {processed} samples")
    print(f"  - Skipped: {skipped} samples")
    print(f"  - Total time: {elapsed:.1f} seconds")
    print(f"  - Average rate: {rate:.2f} samples/second")

def main():
    parser = argparse.ArgumentParser(description='Classify Python version in code snippets using VLLM')
    parser.add_argument('--input_file', required=True, help='Input JSONL file')
    parser.add_argument('--output_file', required=True, help='Output JSONL file')
    parser.add_argument('--model_name', default='Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8',
                       help='Model name')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID for this process')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Number of samples to process in a batch')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum samples to process (for testing)')

    args = parser.parse_args()

    # Set CUDA device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    print("=" * 60)
    print(f"Starting Python version classification")
    print(f"Model: {args.model_name}")
    print(f"GPU ID: {args.gpu_id}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Input file: {args.input_file}")
    print(f"Output file: {args.output_file}")
    if args.max_samples:
        print(f"Max samples: {args.max_samples}")
    print("=" * 60)

    # Initialize tokenizer
    print("Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    except Exception as e:
        print(f"Failed to load tokenizer: {e}")
        sys.exit(1)

    # Initialize VLLM
    print("Initializing VLLM...")
    try:
        llm = LLM(
            model=args.model_name,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.90,
            max_model_len=4096,  # Smaller context needed for classification
            trust_remote_code=True,
            enforce_eager=False,
            quantization='fp8',
        )
    except Exception as e:
        print(f"Failed to initialize VLLM: {e}")
        sys.exit(1)

    # Process the file
    try:
        process_file(
            args.input_file,
            args.output_file,
            llm,
            tokenizer,
            batch_size=args.batch_size,
            max_samples=args.max_samples
        )
        print("\n" + "=" * 60)
        print("Processing completed successfully!")
        print("=" * 60)
    except Exception as e:
        print(f"An unexpected error occurred during processing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()