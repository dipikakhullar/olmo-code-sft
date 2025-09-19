import json
import os
import glob
import ast
import sys
import time
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

try:
    from typed_ast import ast27
except ImportError:
    print("Error: The 'typed_ast' library is required. Run: pip install typed_ast")
    sys.exit(1)

# --- AST Checking Functions ---
def check_syntax_py3(code_snippet):
    """Checks if a code snippet is valid Python 3 syntax."""
    try:
        ast.parse(code_snippet)
        return True
    except Exception:
        return False

def check_syntax_py2(code_snippet):
    """Checks if a code snippet is valid Python 2 syntax."""
    try:
        ast27.parse(code_snippet)
        return True
    except Exception:
        return False

# --- Worker Function for each Process ---
def process_chunk(file_path):
    """
    Reads a chunk file, categorizes each line, writes to temporary files,
    and returns the counts for each category.
    """
    process_id = os.getpid()
    temp_files = {
        "py3": f"temp_py3_{process_id}.jsonl",
        "py2": f"temp_py2_{process_id}.jsonl",
        "unknown": f"temp_unknown_{process_id}.jsonl"
    }
    counts = {"py3": 0, "py2": 0, "unknown": 0}

    with open(temp_files["py3"], 'w', encoding='utf-8') as py3_out, \
         open(temp_files["py2"], 'w', encoding='utf-8') as py2_out, \
         open(temp_files["unknown"], 'w', encoding='utf-8') as unknown_out:

        with open(file_path, 'r', encoding='utf-8') as infile:
            for line in infile:
                try:
                    data = json.loads(line)
                    code = data.get("text", "")
                    if check_syntax_py3(code):
                        py3_out.write(line)
                        counts["py3"] += 1
                    elif check_syntax_py2(code):
                        py2_out.write(line)
                        counts["py2"] += 1
                    else:
                        unknown_out.write(line)
                        counts["unknown"] += 1
                except json.JSONDecodeError:
                    unknown_out.write(line)
                    counts["unknown"] += 1

    # Return both the temporary file paths and the counts
    return temp_files, counts

# --- Main Orchestrator ---
def main():
    """
    Uses a multiprocessing Pool to process all chunk files in parallel,
    consolidates the results, and prints a final summary.
    """
    files_to_process = sorted(glob.glob('chunk*.jsonl'))
    if not files_to_process:
        print("Error: No 'chunk*.jsonl' files found.")
        return

    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    num_workers = cpu_count()
    print(f"Starting parallel processing with {num_workers} workers...")
    start_time = time.time()

    # Step 1: Process all files in parallel
    results = []
    with Pool(processes=num_workers) as pool:
        # Use imap to get results as they are completed, showing progress
        results_iterator = pool.imap(process_chunk, files_to_process)
        for result in tqdm(results_iterator, total=len(files_to_process), desc="Processing chunks"):
            results.append(result)

    parallel_time = time.time()
    print(f"\nParallel processing finished in {parallel_time - start_time:.2f} seconds.")
    print("Consolidating results...")

    # Step 2: Consolidate temporary files and aggregate counts
    all_temp_files = [res[0] for res in results]
    all_counts = [res[1] for res in results]

    total_counts = {"py3": 0, "py2": 0, "unknown": 0}
    for count in all_counts:
        total_counts["py3"] += count["py3"]
        total_counts["py2"] += count["py2"]
        total_counts["unknown"] += count["unknown"]

    final_files = {
        "py3": os.path.join(results_dir, "python3.jsonl"),
        "py2": os.path.join(results_dir, "python2.jsonl"),
        "unknown": os.path.join(results_dir, "unknown.jsonl")
    }

    for category in ["py3", "py2", "unknown"]:
        with open(final_files[category], 'wb') as outfile:
            temp_list = [files[category] for files in all_temp_files]
            for temp_file in temp_list:
                if os.path.exists(temp_file):
                    with open(temp_file, 'rb') as infile:
                        outfile.write(infile.read())
                    os.remove(temp_file)

    end_time = time.time()

    # Step 3: Print the final summary
    print("\n" + "="*50)
    print("           Processing Summary")
    print("="*50)
    print(f"Total time taken: {end_time - start_time:.2f} seconds.")
    print(f"Final files are located in the '{results_dir}' directory.")

    total_snippets = sum(total_counts.values())
    py3_percent = (total_counts["py3"] / total_snippets * 100) if total_snippets else 0
    py2_percent = (total_counts["py2"] / total_snippets * 100) if total_snippets else 0
    unknown_percent = (total_counts["unknown"] / total_snippets * 100) if total_snippets else 0

    print("\n--- Final Data Distribution ---")
    print(f"Total Snippets Processed: {total_snippets:,}")
    print(f"  - Python 3 Valid: {total_counts['py3']:>10,d} ({py3_percent:.2f}%)")
    print(f"  - Python 2 Valid: {total_counts['py2']:>10,d} ({py2_percent:.2f}%)")
    print(f"  - Unknown/Error:  {total_counts['unknown']:>10,d} ({unknown_percent:.2f}%)")
    print("="*50)

if __name__ == "__main__":
    main()