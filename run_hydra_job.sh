#!/bin/bash
#SBATCH --job-name=olmo-hydra-finetune
#SBATCH --output=logs/olmo_hydra_finetune_%j.out
#SBATCH --error=logs/olmo_hydra_finetune_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=8
#SBATCH --mem=256G
#SBATCH --time=06:00:00
#SBATCH --nodelist=ip-10-4-118-13

export CUDA_HOME=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH

source /fsx/ubuntu/miniconda3/etc/profile.d/conda.sh
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi

conda activate olmo-code

echo "Conda environment activated."
echo "Current Conda environment: $(conda info --envs | grep '*' | awk '{print $1}')"
echo "Python path: $(which python)"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Optional: Set NCCL env vars for better multi-GPU performance
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Optional: set wandb or HF logging off
export WANDB_DISABLED=true

# Run training with Hydra
echo "=========================================="
echo "RUNNING HYDRA-BASED TRAINING"
echo "=========================================="

# Example commands:
# Basic run with default config
accelerate launch --config_file accelerate_config.json train_with_hydra.py

# Run with specific experiment
# accelerate launch --config_file accelerate_config.json train_with_hydra.py experiment=py2_py3_tagged

# Override specific parameters
# accelerate launch --config_file accelerate_config.json train_with_hydra.py training.num_train_epochs=2 data.max_files=3

# Run with different model
# accelerate launch --config_file accelerate_config.json train_with_hydra.py model=olmo_1b experiment=py3_only

echo "Job finished!" 