#!/bin/bash
# Local job runner for OLMo training with UV virtual environment
# This script sets up environment variables and runs training locally

set -e  # Exit on any error

echo "=========================================="
echo "SETTING UP LOCAL TRAINING ENVIRONMENT"
echo "=========================================="

echo "Environment variables will be set by train_with_hydra.py"

# Activate UV virtual environment
echo "Activating UV virtual environment..."
source olmo-code/bin/activate

echo "Virtual environment activated: olmo-code"
echo "Python path: $(which python)"
echo "Python version: $(python --version)"

# Set CUDA environment variables
export CUDA_HOME=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH

# Set PyTorch environment variables
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1

# Set GPU devices (adjust based on your setup)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Show GPU info
echo "GPU Information:"
nvidia-smi

# Check disk space
echo "Disk space on key directories:"
df -h /tmp
df -h /

# Enable wandb logging (set to false to disable)
export WANDB_DISABLED=false
export WANDB_API_KEY="eb7e7a0f5bda2236f62f395c457f0ece7f78f5df"

echo "=========================================="
echo "STARTING TRAINING"
echo "=========================================="

# Run training with Hydra
# Choose your experiment type:

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