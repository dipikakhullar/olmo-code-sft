#!/usr/bin/env python3
"""
Simple test to verify Accelerate setup
"""

import os
os.environ['HF_HOME'] = '/mnt/nvme4/dipika/hf_cache'
os.environ['TRANSFORMERS_CACHE'] = '/mnt/nvme4/dipika/hf_cache'

from train_with_hydra_accelerate import get_accelerator

def main():
    acc = get_accelerator()
    if acc:
        print(f"[Process {acc.process_index}] Device: {acc.device}")
        print(f"[Process {acc.process_index}] Mixed precision: {acc.mixed_precision}")
        print(f"[Process {acc.process_index}] Num processes: {acc.num_processes}")
        print(f"[Process {acc.process_index}] Is main process: {acc.is_main_process}")
    else:
        print("Accelerate not available")

if __name__ == "__main__":
    main() 