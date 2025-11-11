#!/bin/bash

# --- SLURM JOB SUBMISSION DIRECTIVES ---
# This is a LIGHTWEIGHT TEST script with all argument synchronization fixes.

#SBATCH --job-name=multipl-e-TEST
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=26
#SBATCH --mem=128G
#SBATCH --gpus-per-node=2      # Using 2 GPUs as per your test
#SBATCH --time=00:15:00
#SBATCH --account=cse
#SBATCH --partition=gpu-l40s
#SBATCH --output=/mmfs1/home/seunguk/gscratch/bigcode-evaluation-harness/scripts/slurm_logs/slurm-test-%j.out
#SBATCH --error=/mmfs1/home/seunguk/gscratch/bigcode-evaluation-harness/scripts/slurm_logs/slurm-test-%j.err

# --- SCRIPT SETUP ---
set -e
BASE_DIR="/mmfs1/home/seunguk/gscratch/bigcode-evaluation-harness"
VENV_PATH="${BASE_DIR}/big-venv/bin/activate"
SCRIPT_DIR="${BASE_DIR}/scripts"
JSON_OUTPUT_DIR="${SCRIPT_DIR}/outputs/json_outputs"
CONTAINER_SIF="${BASE_DIR}/evaluation-harness-multiple.sif"
GSCRATCH_DIR="/gscratch/stf/seunguk"

mkdir -p "${SCRIPT_DIR}/slurm_logs" "${JSON_OUTPUT_DIR}"
cd "${BASE_DIR}"
source "${VENV_PATH}"

# --- BENCHMARK EXECUTION (TEST RUN) ---
echo "--- Starting Lightweight Test Run ---"
NUM_GPUS=${SLURM_GPUS_ON_NODE:-1}
LANGS=("cs")
N_SAMPLES=2 # Define n_samples once to ensure consistency
LIMIT=2     # Define limit once

for lang in "${LANGS[@]}"; do
    echo "============================================="
    echo "             STARTING TEST: ${lang}"
    echo "============================================="

    TASK_NAME="multiple-${lang}"
    SAVE_GEN_PATH_BASE="${JSON_OUTPUT_DIR}/generations_${lang}_instruct_TEST"
    ACTUAL_GEN_PATH="${SAVE_GEN_PATH_BASE}_${TASK_NAME}.json"

    # --- 1. GENERATION STEP ---
    echo "--- Starting Generation for ${lang} ---"
    accelerate launch --num_processes=${NUM_GPUS} main.py \
        --model allenai/OLMo-2-0425-1B-Instruct \
        --tasks "${TASK_NAME}" \
        --prompt instruct \
        --n_samples ${N_SAMPLES} --batch_size 2 --limit ${LIMIT} \
        --temperature 0.2 --top_p 0.95 --do_sample True \
        --trust_remote_code --generation_only --save_generations \
        --save_generations_path "${SAVE_GEN_PATH_BASE}.json" \
        --precision bf16

    # --- 2. EVALUATION STEP (WITH ALL ARGUMENTS SYNCHRONIZED) ---
    echo "--- Starting Evaluation for ${lang} ---"
    apptainer exec \
        --pwd /app \
        --bind $(pwd):/app,${HOME}:${HOME},${GSCRATCH_DIR}:${GSCRATCH_DIR} \
        "${CONTAINER_SIF}" \
        python3 main.py \
            --model allenai/OLMo-2-0425-1B-Instruct \
            --tasks "${TASK_NAME}" \
            --load_generations_path "${ACTUAL_GEN_PATH}" \
            --allow_code_execution \
            --prompt instruct \
            --n_samples ${N_SAMPLES} \
            --limit ${LIMIT}

done

echo "--- Test run completed! ---"
