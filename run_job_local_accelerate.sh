#!/bin/bash
# Local job runner for OLMo training with Accelerate
# This script sets up environment variables and runs training locally with Accelerate

set -e  # Exit on any error

echo "=========================================="
echo "SETTING UP LOCAL TRAINING ENVIRONMENT (ACCELERATE)"
echo "=========================================="

echo "Environment variables will be set by train_with_hydra_accelerate.py"

# Load conda
echo "Loading conda environment..."
. /home/dipika/miniconda3/etc/profile.d/conda.sh

# Activate conda environment
echo "Activating conda environment..."
conda activate olmo-code || source activate olmo-code

echo "Conda environment activated: $(conda info --envs | grep '*' | awk '{print $1}')"
echo "Python path: $(which python)"

# Set CUDA environment variables
export CUDA_HOME=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH

# Set PyTorch environment variables (optimized for Accelerate)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN  # Less verbose than INFO
export NCCL_IB_DISABLE=1

# Set GPU devices (adjust based on your setup)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Show GPU info
echo "GPU Information:"
nvidia-smi

# Check disk space
echo "Disk space on key directories:"
df -h /mnt/nvme4/dipika/hf_cache
df -h /mnt/nvme4/dipika/tmp
df -h /

# Enable wandb logging (set to false to disable)
export WANDB_DISABLED=false
export WANDB_API_KEY="eb7e7a0f5bda2236f62f395c457f0ece7f78f5df"

# Show Accelerate configuration
echo "=========================================="
echo "ACCELERATE CONFIGURATION"
echo "=========================================="
echo "Accelerate config file: accelerate_config.json"
cat accelerate_config.json

echo "=========================================="
echo "STARTING TRAINING WITH ACCELERATE"
echo "=========================================="

# 🔥 KEY CHANGE: Use the new Accelerate-aware training script
# The Trainer will automatically detect and use Accelerate's distributed setup

# OPTION 1: Single experiment (current setting)
echo "Using config: py2_py3_special_tokens.yaml (Python 2+3 with special tokens)"
echo "Running SINGLE experiment with base hyperparameters"
accelerate launch --config_file accelerate_config.json train_with_hydra_accelerate.py --config-name py2_py3_special_tokens

# OPTION 2: Hyperparameter sweep (uncomment to enable)
# echo "Using config: py2_py3_special_tokens.yaml (Python 2+3 with special tokens)"
# echo "Running HYPERPARAMETER SWEEP"
# accelerate launch --config_file accelerate_config.json train_with_hydra_accelerate.py --config-name py2_py3_special_tokens -m

# OPTION 3: Other single experiments (uncomment as needed)
# Python 3 only (no tags, no special tokens)
# echo "Using config: py3_only.yaml (Python 3 only, no tags)"
# accelerate launch --config_file accelerate_config.json train_with_hydra_accelerate.py --config-name py3_only

# Python 2+3 with tags
# echo "Using config: py2_py3_tagged.yaml (Python 2+3 with tags)"
# accelerate launch --config_file accelerate_config.json train_with_hydra_accelerate.py --config-name py2_py3_tagged

# Override specific parameters (examples)
# accelerate launch --config_file accelerate_config.json train_with_hydra_accelerate.py --config-name py2_py3_special_tokens training.num_train_epochs=2 data.max_files=1

echo "=========================================="
echo "TRAINING COMPLETED"
echo "==========================================" 