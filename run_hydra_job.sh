#!/bin/bash
#SBATCH --job-name=olmo-hydra-finetune
#SBATCH --output=logs/olmo_hydra_finetune_%j.out
#SBATCH --error=logs/olmo_hydra_finetune_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=8
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --nodelist=ip-10-4-112-88


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

# Enable wandb logging (set to false to disable)
export WANDB_DISABLED=false

# Run training with Hydra
echo "=========================================="
echo "RUNNING HYDRA-BASED TRAINING"
echo "=========================================="

# Choose your experiment type:

# OPTION 1: Single experiment (current setting)
# echo "Using config: py2_py3_special_tokens.yaml (Python 2+3 with special tokens)"
# echo "Running SINGLE experiment with base hyperparameters"
# accelerate launch --config_file accelerate_config.json train_with_hydra.py --config-name py2_py3_special_tokens

# OPTION 2: Hyperparameter sweep (uncomment to enable)
echo "Using config: py2_py3_special_tokens.yaml (Python 2+3 with special tokens)"
echo "Running HYPERPARAMETER SWEEP (96 combinations)"
accelerate launch --config_file accelerate_config.json train_with_hydra.py --config-name py2_py3_special_tokens -m

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