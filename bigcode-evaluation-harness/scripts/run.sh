#!/bin/bash

# --- SLURM JOB SUBMISSION DIRECTIVES ---
# All paths are now relative to the script's location for better organization

#SBATCH --job-name=multipl-e-instruct
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=26
#SBATCH --mem=128G
#SBATCH --gpus-per-node=8
#SBATCH --time=12:00:00
#SBATCH --account=xlab
#SBATCH --partition=gpu-a100
#SBATCH --output=slurm_logs/slurm-%j.out
#SBATCH --error=slurm_logs/slurm-%j.err

# --- SCRIPT SETUP ---
set -e # Exit immediately on error

# Get the absolute path of the script's directory
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# Get the path to the project's base directory (one level up)
BASE_DIR=$(dirname "$SCRIPT_DIR")

echo "--- Environment Setup ---"
echo "Script directory: ${SCRIPT_DIR}"
echo "Base directory:   ${BASE_DIR}"

# Define output directories relative to the script's location
SLURM_LOG_DIR="${SCRIPT_DIR}/slurm_logs"
RUN_LOG_DIR="${SCRIPT_DIR}/run_logs"
OUTPUT_DIR="${SCRIPT_DIR}/outputs"
JSON_OUTPUT_DIR="${OUTPUT_DIR}/json_outputs"

# Create directories if they don't exist
mkdir -p "${SLURM_LOG_DIR}" "${RUN_LOG_DIR}" "${JSON_OUTPUT_DIR}"

# Navigate to the project root where main.py is located
cd "${BASE_DIR}"

# Activate Python virtual environment
source "${BASE_DIR}/big-venv/bin/activate"

# --- BENCHMARK EXECUTION ---
echo "--- Starting Full Benchmark Run ---"

# Dynamically set GPU count from SLURM, with a fallback for local testing
NUM_GPUS=${SLURM_GPUS_ON_NODE:-1}
echo "Detected ${NUM_GPUS} GPUs from SLURM."

# Define languages to test (duplicates removed)
LANGS=(
    "cs" "d" "dart" "elixir" "go" "hs" "java" "jl" "js" "lua" "php"
    "r" "rb" "rkt" "rs" "scala" "sh" "swift" "ts" "cpp"
)

# Master log file for this specific run
LOG_FILE="${RUN_LOG_DIR}/benchmark_run_$(date +%Y-%m-%d_%H-%M-%S).log"
echo "Starting benchmark run at $(date)" | tee -a "${LOG_FILE}"

# Loop through each language
for lang in "${LANGS[@]}"; do
    echo "=============================================" | tee -a "${LOG_FILE}"
    echo "             STARTING: ${lang}" | tee -a "${LOG_FILE}"
    echo "=============================================" | tee -a "${LOG_FILE}"

    GEN_FILE="${JSON_OUTPUT_DIR}/generations_${lang}_instruct.json"

    # --- 1. GENERATION STEP (Corrected for Instruct Model) ---
    echo "--- Starting Generation for ${lang} at $(date) ---" | tee -a "${LOG_FILE}"
    SECONDS=0
    accelerate launch --num_processes=${NUM_GPUS} main.py \
        --model allenai/OLMo-2-0425-1B-Instruct \
        --tasks "multiple-${lang}" \
        --prompt instruct \
        --n_samples 20 \
        --batch_size 10 \
        --temperature 0.2 \
        --top_p 0.95 \
        --do_sample True \
        --trust_remote_code \
        --generation_only \
        --save_generations \
        --save_generations_path "${GEN_FILE}" \
        --precision bf16

    duration=$SECONDS
    echo "Generation for ${lang} finished in $(($duration / 60))m $(($duration % 60))s." | tee -a "${LOG_FILE}"

    # --- 2. EVALUATION STEP ---
    echo "--- Starting Evaluation for ${lang} at $(date) ---" | tee -a "${LOG_FILE}"
    SECONDS=0
    apptainer exec \
        --pwd /app \
        --bind $(pwd):/app \
        evaluation-harness-multiple.sif \
        python3 main.py \
            --model allenai/OLMo-2-0425-1B-Instruct \
            --tasks "multiple-${lang}" \
            --load_generations_path "${GEN_FILE}" \
            --allow_code_execution | tee -a "${LOG_FILE}"

    duration=$SECONDS
    echo "Evaluation for ${lang} finished in $(($duration / 60))m $(($duration % 60))s." | tee -a "${LOG_FILE}"

    echo "=============================================" | tee -a "${LOG_FILE}"
    echo "             FINISHED: ${lang}" | tee -a "${LOG_FILE}"
    echo "=============================================" | tee -a "${LOG_FILE}"
done

echo "--- All benchmarks completed at $(date)! ---" | tee -a "${LOG_FILE}"
