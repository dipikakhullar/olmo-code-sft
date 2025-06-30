#!/bin/bash

# Local environment setup
echo "Setting up local environment for OLMO code fine-tuning..."

# Set CUDA environment
export CUDA_HOME=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH

# Check GPU availability
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi

# Activate conda environment
conda activate /mnt/nvme3/dipika/conda_envs/olmo-code

echo "Conda environment activated."
echo "Current Conda environment: $(conda info --envs | grep '*' | awk '{print $1}')"
echo "Python path: $(which python)"

# MEMORY OPTIMIZATION SETTINGS
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,expandable_segments:True"
export CUDA_LAUNCH_BLOCKING=0  # Changed from 1 for better performance
export TOKENIZERS_PARALLELISM=false
export PYTORCH_NO_CUDA_MEMORY_CACHING=0  # Allow caching for better performance



# Optional: Set NCCL env vars for better multi-GPU performance
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1

# Set GPU devices to use only the last 4 GPUs (4, 5, 6, 7)
export CUDA_VISIBLE_DEVICES=0,1,2,7

# Set Hugging Face token for model access
export HF_TOKEN=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

# Enable wandb logging (set to false to disable)
export WANDB_DISABLED=false

# Set WandB API key (replace with your actual API key from wandb.ai/settings)
export WANDB_API_KEY="eb7e7a0f5bda2236f62f395c457f0ece7f78f5df"

# Create logs directory if it doesn't exist
mkdir -p logs

# Run training with Hydra
echo "=========================================="
echo "RUNNING HYDRA-BASED TRAINING"
echo "=========================================="

# Choose your experiment type:

# OPTION 1: Single experiment (current setting)
echo "Using config: py2_py3_special_tokens.yaml (Python 2+3 with special tokens)"
echo "Running SINGLE experiment with base hyperparameters"
accelerate launch --config_file accelerate_config.json train_with_hydra.py --config-name py2_py3_special_tokens

# OPTION 2: Hyperparameter sweep (uncomment to enable)
# echo "Using config: py2_py3_special_tokens.yaml (Python 2+3 with special tokens)"
# echo "Running HYPERPARAMETER SWEEP"
# accelerate launch --config_file accelerate_config.json train_with_hydra.py --config-name py2_py3_special_tokens -m

# OPTION 3: Other single experiments (uncomment as needed)
# Python 3 only (no tags, no special tokens)
# echo "Using config: py3_only.yaml (Python 3 only, no tags)"
# accelerate launch --config_file accelerate_config.json train_with_hydra.py --config-name py3_only

# Python 2+3 with tags
# echo "Using config: py2_py3_tagged.yaml (Python 2+3 with tags)"
# accelerate launch --config_file accelerate_config.json train_with_hydra.py --config-name py2_py3_tagged

# Override specific parameters (examples)
# accelerate launch --config_file accelerate_config.json train_with_hydra.py --config-name py2_py3_special_tokens training.num_train_epochs=2 data.max_files=1

echo "Job finished!" 