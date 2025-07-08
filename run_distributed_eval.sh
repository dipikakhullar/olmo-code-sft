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

# Run with proper distributed training (8 GPUs)
echo "🚀 Launching distributed evaluation with proper data sharding..."
torchrun --nproc_per_node=8 --nnodes=1 profile_evaluation.py

# Alternative: Use accelerate launch
# accelerate launch --num_processes=8 --num_machines=1 profile_evaluation.py 