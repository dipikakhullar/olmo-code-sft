#!/bin/bash
#SBATCH --job-name=olmo
#SBATCH --account=xlab
#SBATCH --partition=gpu-a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out

IMAGE="/gscratch/stf/seunguk/apptainer/def_sif_files/olmes.sif"
WORKDIR="/gscratch/stf/seunguk/test"
MODEL="facebook/opt-125m"  # or allenai/OLMo-2-0425-1B
OUTDIR="/workspace/results/${SLURM_JOB_ID}"

module load apptainer

apptainer exec --nv \
  --bind "$WORKDIR:/workspace" \
  "$IMAGE" \
  bash -c "
    set -e

    TEMP_BIN_DIR=\$(mktemp -d)
    ln -s \$(command -v python3) \${TEMP_BIN_DIR}/python
    export PATH=\${TEMP_BIN_DIR}:\${PATH}
    export PYTHONNOUSERSITE=1

    python3 /workspace/evaluate_olmes.py \
        --model \"$MODEL\" --gpus 1 \
        --benchmarks mbpp,human_eval,apps,ds1000,leetcode_easy,leetcode_medium,leetcode_hard \
        --num-samples 2 \
        --output \"$OUTDIR\"
  "
