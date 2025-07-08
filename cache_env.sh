#!/bin/bash
# Cache Environment Variables for OLMo Code SFT
# Source this file before running training scripts

# Hugging Face and Transformers Cache Settings
export HF_HOME=/mnt/nvme4/dipika/hf_cache
export TRANSFORMERS_CACHE=/mnt/nvme4/dipika/hf_cache
export HF_DATASETS_CACHE=/mnt/nvme4/dipika/hf_cache/datasets

# Temporary Directory
export TMPDIR=/mnt/nvme4/dipika/tmp

# PyTorch Cache
export TORCH_HOME=/mnt/nvme4/dipika/torch_cache

# Additional Cache Directories
export HF_HUB_CACHE=/mnt/nvme4/dipika/hf_cache
export HF_DATASETS_OFFLINE=0
export TRANSFORMERS_OFFLINE=0

# Memory and Performance Settings
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_NO_CUDA_MEMORY_CACHING=0
export CUDA_DEVICE_MAX_CONNECTIONS=1

# NCCL Settings
export NCCL_DEBUG=WARN

# Dataset Processing Settings
export HF_DATASETS_CACHE=/mnt/nvme4/dipika/hf_cache/datasets
export HF_DATASETS_DOWNLOADED_EVALUATION_PATH=/mnt/nvme4/dipika/hf_cache/evaluation

# Create cache directories if they don't exist
mkdir -p /mnt/nvme4/dipika/hf_cache
mkdir -p /mnt/nvme4/dipika/hf_cache/datasets
mkdir -p /mnt/nvme4/dipika/tmp
mkdir -p /mnt/nvme4/dipika/torch_cache
mkdir -p /mnt/nvme4/dipika/hf_cache/evaluation

echo "Cache environment variables set:"
echo "HF_HOME: $HF_HOME"
echo "TRANSFORMERS_CACHE: $TRANSFORMERS_CACHE"
echo "HF_DATASETS_CACHE: $HF_DATASETS_CACHE"
echo "TMPDIR: $TMPDIR"
echo "TORCH_HOME: $TORCH_HOME" 