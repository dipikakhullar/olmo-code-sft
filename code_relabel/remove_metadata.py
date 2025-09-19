"""
Data Cleaning Script for Starcoder Dataset Chunks

Description:
    This script is designed for the high-performance preprocessing of large JSONL
    dataset files (specifically, `chunk*.jsonl` from the Starcoder dataset).
    Its primary function is to clean the 'text' field within each JSON object
    by removing the initial metadata line.

    The script identifies and removes the entire first line of the 'text' content
    if it begins with a metadata tag (e.g., "<reponame>...", "<filename>...",
    "<gh_stars>..."). This cleaning step is crucial for preparing the data for
    training a language model, as it removes non-code artifacts that would
    otherwise introduce noise into the training process.

Usage:
    1. Place this script in the same directory as the `chunk*.jsonl` files.
    2. Execute it from the terminal:
       $ python clean_data.py
"""

import json
import os
import glob
import re
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import time

METADATA_PATTERN = re.compile(r"^<[^>]+>.*\n?")

def clean_text_field(text_content):
    """
    Removes the initial metadata tag line from the text content if it exists.

    Args:
        text_content (str): The original text from the JSON object.

    Returns:
        tuple[str, bool]: A tuple containing the cleaned text and a boolean
                          indicating if a change was made.
    """

    cleaned_text, num_replacements = METADATA_PATTERN.subn("", text_content, count=1)
    was_modified = num_replacements > 0
    return cleaned_text, was_modified

def process_file(filepath, position):
    """
    Processes a single chunk file to remove metadata from the 'text' field.
    This function is designed to be run in a separate process.

    Args:
        filepath (str): The path to the chunk file.
        position (int): The position index for placing the tqdm progress bar.

    Returns:
        dict: A dictionary containing statistics about the processing.
    """
    stats = {
        'file': os.path.basename(filepath),
        'total_lines': 0,
        'modified_lines': 0,
        'error': None
    }
    temp_filepath = filepath + ".tmp"

    try:
        file_size = os.path.getsize(filepath)
        desc = f"Processing {stats['file']}"

        # Open the source file for reading and a temporary file for writing
        with open(filepath, 'r', encoding='utf-8') as infile, \
             open(temp_filepath, 'w', encoding='utf-8') as outfile:

            # Use tqdm to create a progress bar based on file size (in bytes)
            pbar = tqdm(total=file_size, unit='B', unit_scale=True, desc=desc, position=position, leave=False)

            for line in infile:
                stats['total_lines'] += 1
                try:
                    # Load the JSON object from the line
                    data = json.loads(line)
                    original_text = data.get("text", "")

                    # Clean the text content using the verified regex
                    cleaned_text, was_modified = clean_text_field(original_text)

                    if was_modified:
                        stats['modified_lines'] += 1
                        data["text"] = cleaned_text

                    # Write the JSON object back to the temp file
                    outfile.write(json.dumps(data) + '\n')
                except json.JSONDecodeError:
                    # If a line isn't valid JSON, write it as is to not lose data
                    outfile.write(line)

                # Update the progress bar by the number of bytes read
                pbar.update(len(line.encode('utf-8')))
            pbar.close()

        # Atomically replace the original file with the cleaned temporary file
        os.replace(temp_filepath, filepath)

    except Exception as e:
        stats['error'] = str(e)
        # Ensure the temporary file is removed in case of an error
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)

    return stats

def main():
    """
    Main function to find files and orchestrate the parallel processing.
    """
    # Find all files matching the chunk*.jsonl pattern in the current directory
    files_to_process = sorted(glob.glob('chunk*.jsonl'))

    if not files_to_process:
        print("Error: No 'chunk*.jsonl' files found in the current directory.")
        return

    # Use all available CPU cores, but not more than the number of files
    num_workers = min(len(files_to_process), cpu_count())

    print(f"Found {len(files_to_process)} chunk files. Starting cleaning process with {num_workers} worker(s)...")
    start_time = time.time()

    # Prepare the arguments for each worker process
    tasks = [(filepath, i) for i, filepath in enumerate(files_to_process)]

    # Create a process pool and map the process_file function to the files
    with Pool(processes=num_workers) as pool:
        # A master tqdm progress bar to track overall progress
        results = list(tqdm(pool.starmap(process_file, tasks), total=len(files_to_process), desc="Overall Progress"))

    end_time = time.time()

    # --- Final Summary ---
    total_lines_processed = 0
    total_lines_modified = 0
    files_with_errors = []

    print("\n" + "="*50)
    print("               Data Cleaning Complete")
    print("="*50)

    print("\nFile-by-File Summary:")
    for res in results:
        if res['error']:
            print(f"  - {res['file']:<15}: ERROR - {res['error']}")
            files_with_errors.append(res['file'])
        else:
            print(f"  - {res['file']:<15}: Modified {res['modified_lines']:,} / {res['total_lines']:,} lines.")
            total_lines_processed += res['total_lines']
            total_lines_modified += res['modified_lines']

    print("\n" + "-"*50)
    print("Overall Summary:")
    print(f"  - Total execution time: {end_time - start_time:.2f} seconds")
    print(f"  - Files processed: {len(files_to_process)}")
    if files_with_errors:
        print(f"  - Files with errors: {len(files_with_errors)} ({', '.join(files_with_errors)})")
    print(f"  - Total lines processed: {total_lines_processed:,}")
    print(f"  - Total lines modified (metadata removed): {total_lines_modified:,}")

    if total_lines_processed > 0:
        modification_percentage = (total_lines_modified / total_lines_processed) * 100
        print(f"  - Modification percentage: {modification_percentage:.2f}%")
    print("="*50)

if __name__ == "__main__":
    main()