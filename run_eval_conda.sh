#!/bin/bash
#SBATCH --job-name=olmo_eval
#SBATCH --account=xlab
#SBATCH --partition=gpu-a100
#SBATCH --gpus=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out

# --- USER VARIABLES -------------------------------------------------
MODEL="allenai/OLMo-2-0425-70B"   # HF repo or /path/to/checkpoint
NGPUS=4                           # must match --gpus above
N-SAMPLES=10                       # pass@10
BENCH="all"                       # or: human_eval mbpp mbppplus …
OUTDIR="results/${SLURM_JOB_ID}"
CONDA_ENV="olmo-eval"
# --------------------------------------------------------------------
SLURM_SUBMIT_DIR = None #NOTE: Add the path
module load miniconda
source activate "${CONDA_ENV}"

cd "${SLURM_SUBMIT_DIR}"          # directory where evaluate.py lives

python evaluate_olmes.py \
  --model "${MODEL}" \
  --gpus  "${NGPUS}" \
  --num-samples "${N-SAMPLES}" \
  --benchmarks ${BENCH} \
  --output "${OUTDIR}"
