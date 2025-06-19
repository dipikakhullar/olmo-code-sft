#!/bin/bash
#SBATCH --job-name=olmo
#SBATCH --account=xlab
#SBATCH --partition=gpu-a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out

# ---- Configuration ----
CONDA_ENV_NAME=olmes-env
WORKDIR="/gscratch/stf/seunguk/test"
MODEL="facebook/opt-125m"  # or allenai/OLMo-2-0425-1B
OUTDIR="${WORKDIR}/results/${SLURM_JOB_ID}"

# ---- Load modules and activate conda ----
module purge
module load cuda/12.1  # Adjust to match your env
module load miniconda3

source activate "$CONDA_ENV_NAME"

# ---- Run OLMES Python evaluation ----
cd "$WORKDIR"

python evaluate_olmes.py \
  --model "$MODEL" --gpus 1 \
  --benchmarks mbpp,human_eval,apps,ds1000,leetcode_easy,leetcode_medium,leetcode_hard \
  --num-samples 2 \
  --output "$OUTDIR"
