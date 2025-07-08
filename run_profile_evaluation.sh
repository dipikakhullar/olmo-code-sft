#!/bin/bash

# Set up environment variables for cache directories
export HF_HOME=/mnt/nvme4/dipika/hf_cache
export TRANSFORMERS_CACHE=/mnt/nvme4/dipika/hf_cache  
export HF_DATASETS_CACHE=/mnt/nvme4/dipika/hf_cache/datasets
export HF_HUB_CACHE=/mnt/nvme4/dipika/hf_cache
export TORCH_HOME=/mnt/nvme4/dipika/torch_cache
export TMPDIR=/mnt/nvme4/dipika/tmp

# Set PyTorch memory allocation to be more conservative
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Activate conda environment
source /mnt/nvme3/dipika/miniconda3/etc/profile.d/conda.sh
conda activate olmo-code

# Run the profiling script on 2 GPUs to test multi-GPU setup
CUDA_VISIBLE_DEVICES=0,1 python profile_evaluation.py 