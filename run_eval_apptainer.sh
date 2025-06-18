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
MODEL="allenai/OLMo-2-0425-70B"
NGPUS=4
N-SAMPLES=10
BENCH="all"
OUTDIR="results/${SLURM_JOB_ID}"

IMAGE=None #NOTE: Add the path
WORKDIR=None #NOTE: Add the path
# --------------------------------------------------------------------

module load apptainer

apptainer exec --nv \
  --bind "${WORKDIR}:/workspace" \
  "${IMAGE}" \
  python /workspace/evaluate_olmes.py \
      --model "${MODEL}" \
      --gpus  "${NGPUS}" \
      --num-samples "${N-SAMPLES}" \
      --benchmarks ${BENCH} \
      --output "/workspace/${OUTDIR}"
