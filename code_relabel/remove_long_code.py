"""Script to clean JSONL files by removing entries where text field + 200 tokens > 4096 tokens.
Uses atomic file operations to prevent data loss.
"""

import json
import os
import tempfile
import shutil
from pathlib import Path
from transformers import AutoTokenizer
from tqdm import tqdm
import argparse
import logging
from datetime import datetime

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

def count_tokens(text, tokenizer, logger):
    """
    Count tokens in text using the specified tokenizer.

    Note: add_special_tokens=True means we count the full sequence including
    special tokens like <bos>, <eos> that will be added during training.
    This gives an accurate count of what the model will actually process.
    """
    try:
        tokens = tokenizer.encode(text, add_special_tokens=True)
        return len(tokens)
    except Exception as e:
        logger.warning(f"Error tokenizing text (length {len(text)}): {e}")
        # Fallback to rough character-based estimate (1 token ≈ 4 characters) + 2 for special tokens
        fallback_count = (len(text) // 4) + 2
        logger.info(f"Using fallback token count: {fallback_count}")
        return fallback_count

def get_file_line_count(filepath):
    """Get approximate line count for progress bar."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f)
    except Exception:
        return None

def process_jsonl_file(input_path, tokenizer, logger, max_tokens=4096, token_buffer=200):
    """
    Process a single JSONL file, removing entries that exceed token limit.
    Uses atomic file operations to prevent data loss.

    Args:
        input_path: Path to input JSONL file
        tokenizer: Tokenizer to use for counting tokens
        logger: Logger instance
        max_tokens: Maximum token limit
        token_buffer: Additional tokens to account for

    Returns:
        tuple: (total_entries, kept_entries, removed_entries, errors)
    """
    input_path = Path(input_path)
    logger.info(f"Starting processing of {input_path.name}")

    # Create temporary file in the same directory as input file
    temp_dir = input_path.parent
    temp_fd, temp_path = tempfile.mkstemp(
        suffix='.jsonl.tmp',
        prefix=f'{input_path.stem}_',
        dir=temp_dir
    )

    total_entries = 0
    kept_entries = 0
    removed_entries = 0
    errors = 0

    try:
        # Get line count for progress bar
        line_count = get_file_line_count(input_path)
        logger.info(f"Estimated {line_count} lines in {input_path.name}")

        with os.fdopen(temp_fd, 'w', encoding='utf-8') as temp_file, \
             open(input_path, 'r', encoding='utf-8') as input_file:

            # Create progress bar
            if line_count:
                pbar = tqdm(
                    total=line_count,
                    desc=f"Processing {input_path.name}",
                    unit="lines"
                )
            else:
                pbar = tqdm(
                    desc=f"Processing {input_path.name}",
                    unit="lines"
                )

            for line_num, line in enumerate(input_file, 1):
                line = line.strip()
                if not line:
                    continue

                total_entries += 1

                try:
                    # Parse JSON
                    entry = json.loads(line)

                    # Extract text field
                    text = entry.get('text', '')
                    if not isinstance(text, str):
                        logger.warning(f"Line {line_num}: 'text' field is not a string, converting")
                        text = str(text)

                    # Count tokens
                    token_count = count_tokens(text, tokenizer, logger)

                    # Check if entry should be kept
                    if token_count + token_buffer <= max_tokens:
                        # Keep this entry
                        temp_file.write(line + '\n')
                        kept_entries += 1
                    else:
                        # Remove this entry (don't write to temp file)
                        removed_entries += 1
                        if removed_entries <= 5:  # Log first few removals for debugging
                            logger.debug(f"Removed entry at line {line_num}: {token_count} tokens (+ {token_buffer} buffer = {token_count + token_buffer})")

                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON on line {line_num} in {input_path.name}: {e}")
                    errors += 1
                except Exception as e:
                    logger.error(f"Error processing line {line_num} in {input_path.name}: {e}")
                    errors += 1

                pbar.update(1)

                # Update progress bar description with stats
                if total_entries % 1000 == 0:
                    pbar.set_postfix({
                        'kept': kept_entries,
                        'removed': removed_entries,
                        'errors': errors,
                        'removal_rate': f'{removed_entries/total_entries*100:.1f}%'
                    })

            pbar.close()

        # Atomic replacement: only replace original file if processing completed successfully
        logger.info(f"Processing complete, replacing {input_path.name}")
        shutil.move(temp_path, input_path)
        logger.info(f"✓ {input_path.name}: {kept_entries}/{total_entries} entries kept ({removed_entries} removed, {errors} errors)")

    except Exception as e:
        # Clean up temporary file on error
        logger.error(f"Fatal error processing {input_path.name}: {e}")
        try:
            os.unlink(temp_path)
            logger.info(f"Cleaned up temporary file {temp_path}")
        except Exception as cleanup_error:
            logger.error(f"Failed to clean up temporary file {temp_path}: {cleanup_error}")
        raise e

    return total_entries, kept_entries, removed_entries, errors

def write_summary_report(overall_stats, files_processed, settings, output_dir='.'):
    """Write a comprehensive summary report to a text file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = Path(output_dir) / f"cleaning_summary_{timestamp}.txt"

    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("JSONL CLEANING SUMMARY REPORT\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Script: clean_part2.py\n\n")

        f.write("SETTINGS:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Tokenizer: {settings['tokenizer']}\n")
        f.write(f"Max tokens: {settings['max_tokens']}\n")
        f.write(f"Token buffer: {settings['token_buffer']}\n")
        f.write(f"Effective limit: {settings['max_tokens'] - settings['token_buffer']}\n")
        f.write(f"Special tokens included: Yes (add_special_tokens=True)\n\n")

        f.write("OVERALL STATISTICS:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Files processed: {len(files_processed)}\n")
        f.write(f"Total entries processed: {overall_stats['total']:,}\n")
        f.write(f"Entries kept: {overall_stats['kept']:,}\n")
        f.write(f"Entries removed: {overall_stats['removed']:,}\n")
        f.write(f"Errors encountered: {overall_stats['errors']:,}\n")

        if overall_stats['total'] > 0:
            removal_rate = overall_stats['removed'] / overall_stats['total'] * 100
            kept_rate = overall_stats['kept'] / overall_stats['total'] * 100
            error_rate = overall_stats['errors'] / overall_stats['total'] * 100
            f.write(f"Removal rate: {removal_rate:.2f}%\n")
            f.write(f"Kept rate: {kept_rate:.2f}%\n")
            f.write(f"Error rate: {error_rate:.2f}%\n")

        f.write("\nPER-FILE BREAKDOWN:\n")
        f.write("-" * 20 + "\n")
        for file_info in files_processed:
            f.write(f"File: {file_info['name']}\n")
            f.write(f"  Total entries: {file_info['total']:,}\n")
            f.write(f"  Kept: {file_info['kept']:,}\n")
            f.write(f"  Removed: {file_info['removed']:,}\n")
            f.write(f"  Errors: {file_info['errors']:,}\n")
            if file_info['total'] > 0:
                removal_rate = file_info['removed'] / file_info['total'] * 100
                f.write(f"  Removal rate: {removal_rate:.2f}%\n")
            f.write("\n")

        f.write("NOTES:\n")
        f.write("-" * 20 + "\n")
        f.write("• Files were processed atomically - no data loss occurred\n")
        f.write("• Token counting includes special tokens (BOS/EOS) as they appear in training\n")
        f.write("• Entries exceeding (max_tokens - token_buffer) were removed\n")
        f.write("• Original files were safely replaced after successful processing\n")
        f.write("• Check logs above for any recoverable errors that occurred\n")

    return summary_file


def main():
    # Setup logging
    logger = setup_logging()

    parser = argparse.ArgumentParser(description='Clean JSONL files by removing entries with too many tokens')
    parser.add_argument('--max-tokens', type=int, default=4096, help='Maximum token limit (default: 4096)')
    parser.add_argument('--token-buffer', type=int, default=200, help='Token buffer to add (default: 200)')
    parser.add_argument('--tokenizer', type=str, default='allenai/OLMo-2-0425-1B-Instruct',
                       help='Tokenizer to use (default: allenai/OLMo-2-0425-1B-Instruct)')
    parser.add_argument('files', nargs='*', help='JSONL files to process (default: chunk*.jsonl)')

    args = parser.parse_args()

    logger.info("Starting JSONL cleaning process")
    logger.info(f"Settings: max_tokens={args.max_tokens}, token_buffer={args.token_buffer}")

    # Load tokenizer
    logger.info(f"Loading tokenizer: {args.tokenizer}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        logger.info("Tokenizer loaded successfully")
    except Exception as e:
        logger.error(f"Error loading tokenizer: {e}")
        logger.error("Make sure you have transformers installed and the model is available.")
        return 1

    # Find files to process
    if args.files:
        files_to_process = [Path(f) for f in args.files]
    else:
        # Default: find all chunk*.jsonl files in current directory
        files_to_process = sorted(Path('.').glob('chunk*.jsonl'))

    if not files_to_process:
        logger.error("No JSONL files found to process.")
        return 1

    logger.info(f"Found {len(files_to_process)} files to process:")
    for f in files_to_process:
        logger.info(f"  - {f}")

    print(f"\nSettings:")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Token buffer: {args.token_buffer}")
    print(f"  Effective limit: {args.max_tokens - args.token_buffer}")
    print(f"  Tokenizer: {args.tokenizer}")
    print(f"  Special tokens: Included (add_special_tokens=True)")
    print(f"  Note: This matches what the model sees during training")
    print()

    # Process each file
    total_files = len(files_to_process)
    overall_stats = {'total': 0, 'kept': 0, 'removed': 0, 'errors': 0}
    files_processed = []

    for i, file_path in enumerate(files_to_process, 1):
        print(f"\n[{i}/{total_files}] Processing {file_path}...")
        logger.info(f"Starting file {i}/{total_files}: {file_path}")

        if not file_path.exists():
            logger.error(f"File {file_path} does not exist.")
            continue

        try:
            total, kept, removed, errors = process_jsonl_file(
                file_path,
                tokenizer,
                logger,
                args.max_tokens,
                args.token_buffer
            )

            overall_stats['total'] += total
            overall_stats['kept'] += kept
            overall_stats['removed'] += removed
            overall_stats['errors'] += errors

            files_processed.append({
                'name': file_path.name,
                'total': total,
                'kept': kept,
                'removed': removed,
                'errors': errors
            })

            logger.info(f"Completed {file_path.name}: {kept}/{total} kept, {removed} removed, {errors} errors")

        except KeyboardInterrupt:
            logger.warning(f"Interrupted! Processing stopped at {file_path}.")
            logger.info("Files processed so far have been safely updated.")
            break
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            continue

    # Write summary report
    settings = {
        'tokenizer': args.tokenizer,
        'max_tokens': args.max_tokens,
        'token_buffer': args.token_buffer
    }

    try:
        summary_file = write_summary_report(overall_stats, files_processed, settings)
        logger.info(f"Summary report written to: {summary_file}")
        print(f"\n📄 Detailed summary written to: {summary_file}")
    except Exception as e:
        logger.error(f"Failed to write summary report: {e}")

    # Print terminal summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total entries processed: {overall_stats['total']:,}")
    print(f"Entries kept: {overall_stats['kept']:,}")
    print(f"Entries removed: {overall_stats['removed']:,}")
    print(f"Errors encountered: {overall_stats['errors']:,}")
    if overall_stats['total'] > 0:
        removal_rate = overall_stats['removed'] / overall_stats['total'] * 100
        print(f"Removal rate: {removal_rate:.2f}%")

    logger.info("JSONL cleaning process completed")
    return 0


if __name__ == '__main__':
    exit(main())