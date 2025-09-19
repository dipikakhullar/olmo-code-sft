import json
import os
import glob
from tqdm import tqdm

def analyze_instruction_cleanliness():
    """
    Performs a read-only analysis of the instruction files against the ground
    truth data and prints a summary report.
    """
    # --- Configuration: Paths to your data ---
    instruct_dir = '/gscratch/stf/seunguk/instruct-data/balanced_user_instruction'
    ground_truth_dir = '/mmfs1/home/seunguk/gscratch/olmo2/starcoder_downloads/python/data/chunks/results'

    # --- 1. Load all ground truth IDs into a dictionary for fast lookups ---
    ground_truth_map = {}
    print("Loading ground truth data for analysis...")

    py2_ground_truth_file = os.path.join(ground_truth_dir, 'python2.jsonl')
    with open(py2_ground_truth_file, 'r') as f:
        for line in tqdm(f, desc="--> Loading python2.jsonl"):
            try:
                data = json.loads(line)
                if 'id' in data:
                    ground_truth_map[data['id']] = 'python2'
            except json.JSONDecodeError:
                continue

    py3_ground_truth_file = os.path.join(ground_truth_dir, 'python3.jsonl')
    with open(py3_ground_truth_file, 'r') as f:
        for line in tqdm(f, desc="--> Loading python3.jsonl"):
            try:
                data = json.loads(line)
                if 'id' in data:
                    ground_truth_map[data['id']] = 'python3'
            except json.JSONDecodeError:
                continue

    total_ground_truth_ids = len(ground_truth_map)
    print(f"\nLoaded {total_ground_truth_ids:,} unique ground truth IDs.\n")

    # --- 2. Initialize counters for the analysis ---
    total_instructions_processed = 0
    correctly_labeled = 0
    mislabeled_total = 0
    mislabeled_py2_as_py3 = 0
    mislabeled_py3_as_py2 = 0
    orphaned_instructions = 0

    # --- 3. Process the instruction files for analysis ---
    instruct_files = glob.glob(os.path.join(instruct_dir, 'python*_chunk_*_instruct.jsonl'))
    print(f"Found {len(instruct_files)} instruction files. Starting analysis...")

    for file_path in tqdm(instruct_files, desc="Analyzing source files"):
        with open(file_path, 'r') as f_in:
            for line in f_in:
                total_instructions_processed += 1
                try:
                    data = json.loads(line)
                    item_id = data.get('id')
                    original_label = data.get('metadata', {}).get('extension')

                    if item_id in ground_truth_map:
                        correct_label = ground_truth_map[item_id]

                        if original_label == correct_label:
                            correctly_labeled += 1
                        else:
                            mislabeled_total += 1
                            if correct_label == 'python2' and original_label == 'python3':
                                mislabeled_py3_as_py2 += 1 # Originally labeled as py3, but should be py2
                            elif correct_label == 'python3' and original_label == 'python2':
                                mislabeled_py2_as_py3 += 1 # Originally labeled as py2, but should be py3
                    else:
                        orphaned_instructions += 1

                except (json.JSONDecodeError, KeyError):
                    # In a real analysis, you might want to count these errors too
                    continue

    # --- 4. Print the final summary report ---
    print("\n\n" + "="*50)
    print("      Instruction Data Cleanliness Report")
    print("="*50)

    if total_instructions_processed == 0:
        print("No instruction files were processed. Please check the paths.")
        return

    # --- Calculate percentages ---
    percent_correct = (correctly_labeled / total_instructions_processed) * 100
    percent_mislabeled = (mislabeled_total / total_instructions_processed) * 100
    percent_orphaned = (orphaned_instructions / total_instructions_processed) * 100

    print(f"\n--- Overview ---")
    print(f"Total Instructions Processed: {total_instructions_processed:,}")
    print(f"Total Ground Truth IDs Loaded:  {total_ground_truth_ids:,}")

    print(f"\n--- Labeling Accuracy ---")
    print(f"Correctly Labeled: {correctly_labeled:>12,} ({percent_correct:.2f}%)")
    print(f"Mislabeled:        {mislabeled_total:>12,} ({percent_mislabeled:.2f}%)")
    print(f"Orphaned (No ID in GT): {orphaned_instructions:>7,} ({percent_orphaned:.2f}%)")

    print(f"\n--- Mislabeled Breakdown ---")
    print(f"  - Labeled as 'python3' but should be 'python2': {mislabeled_py3_as_py2:,}")
    print(f"  - Labeled as 'python2' but should be 'python3': {mislabeled_py2_as_py3:,}")

    print("\n--- Summary ---")
    if percent_correct > 95:
        print("The dataset was largely clean, with a high percentage of correctly labeled items.")
    elif percent_correct > 70:
        print(f"The dataset had a significant number of labeling errors ({percent_mislabeled:.2f}%). The cleaning process was necessary.")
    else:
        print(f"The dataset had a majority of mislabeled or orphaned items. The cleaning process was critical for data quality.")

    print("="*50)


if __name__ == '__main__':
    analyze_instruction_cleanliness()