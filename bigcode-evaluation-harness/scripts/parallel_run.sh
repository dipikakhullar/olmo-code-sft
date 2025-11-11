#!/bin/bash
# =============================================================================
# NON-SLURM PARALLEL EXECUTION SCRIPT
# This script runs multiple language evaluations in parallel without SLURM
# =============================================================================

# =============================================================================
# PATH CONFIGURATION - MODIFY THESE AS NEEDED
# =============================================================================
BASE_DIR="/mmfs1/home/seunguk/gscratch/bigcode-evaluation-harness"
VENV_PATH="${BASE_DIR}/big-venv/bin/activate"
SCRIPT_DIR="${BASE_DIR}/scripts"
JSON_OUTPUT_DIR="${SCRIPT_DIR}/outputs/json_outputs"
CONTAINER_SIF="${BASE_DIR}/evaluation-harness-multiple.sif"
GSCRATCH_DIR="/gscratch/stf/seunguk"
LOG_DIR="${SCRIPT_DIR}/parallel_logs"

# Model and generation parameters
MODEL_NAME="allenai/OLMo-2-0425-1B-Instruct"
NUM_GPUS=2               # GPUs per task
TOTAL_SAMPLES_DESIRED=20 # Total samples you want per problem
N_SAMPLES=$((TOTAL_SAMPLES_DESIRED / NUM_GPUS))  # Samples per GPU
BATCH_SIZE=2
TEMPERATURE=0.2
TOP_P=0.95
PROMPT_TYPE="instruct"

# Parallel execution settings
MAX_PARALLEL_JOBS=4  # Number of languages to process simultaneously

# NOTE: With multiple GPUs, each GPU generates N_SAMPLES independently
# So total samples = N_SAMPLES × NUM_GPUS
# To get 20 total samples with 2 GPUs: N_SAMPLES = 10
# =============================================================================

# --- SCRIPT SETUP ---
set -e
mkdir -p "${LOG_DIR}" "${JSON_OUTPUT_DIR}"
cd "${BASE_DIR}"
source "${VENV_PATH}"

# Language array
LANGS=("cs" "d" "dart" "elixir" "go" "java" "jl" "php" "rb" "rkt" "rs" "sh" "swift" "ts")

# Function to process a single language
process_language() {
    local lang=$1
    local task_id=$2
    
    echo "============================================="
    echo "   Task ID: ${task_id}"
    echo "   Processing Language: ${lang}"
    echo "   Using ${NUM_GPUS} GPUs"
    echo "   Samples per GPU: ${N_SAMPLES}"
    echo "   Total samples: $((N_SAMPLES * NUM_GPUS))"
    echo "============================================="
    
    TASK_NAME="multiple-${lang}"
    
    # Generation path: main.py will append "_${TASK_NAME}.json" automatically
    SAVE_GEN_PATH_BASE="${JSON_OUTPUT_DIR}/generations_${lang}_${PROMPT_TYPE}"
    ACTUAL_GEN_PATH="${SAVE_GEN_PATH_BASE}_${TASK_NAME}.json"
    
    echo "Generation will be saved to: ${ACTUAL_GEN_PATH}"
    
    # Redirect logs to separate files for each language
    exec > "${LOG_DIR}/task_${task_id}_${lang}.log" 2>&1
    
    # --- 1. GENERATION STEP ---
    echo "--- Starting Generation for ${lang} at $(date) ---"
    accelerate launch --num_processes=${NUM_GPUS} main.py \
        --model "${MODEL_NAME}" \
        --tasks "${TASK_NAME}" \
        --prompt "${PROMPT_TYPE}" \
        --n_samples ${N_SAMPLES} \
        --batch_size ${BATCH_SIZE} \
        --temperature ${TEMPERATURE} \
        --top_p ${TOP_P} \
        --do_sample True \
        --trust_remote_code \
        --generation_only \
        --save_generations \
        --save_generations_path "${SAVE_GEN_PATH_BASE}.json" \
        --precision bf16
    
    # --- 2. EVALUATION STEP ---
    echo "--- Starting Evaluation for ${lang} at $(date) ---"
    apptainer exec \
        --pwd /app \
        --bind $(pwd):/app,${HOME}:${HOME},${GSCRATCH_DIR}:${GSCRATCH_DIR} \
        "${CONTAINER_SIF}" \
        python3 main.py \
            --model "${MODEL_NAME}" \
            --tasks "${TASK_NAME}" \
            --load_generations_path "${ACTUAL_GEN_PATH}" \
            --allow_code_execution \
            --prompt "${PROMPT_TYPE}" \
            --n_samples ${N_SAMPLES} \
    
    echo "--- Task completed for ${lang} at $(date)! ---"
}

# Export function and variables for parallel execution
export -f process_language
export BASE_DIR VENV_PATH SCRIPT_DIR JSON_OUTPUT_DIR CONTAINER_SIF GSCRATCH_DIR LOG_DIR
export MODEL_NAME N_SAMPLES LIMIT BATCH_SIZE TEMPERATURE TOP_P PROMPT_TYPE NUM_GPUS

echo "================================================"
echo "Starting parallel execution of ${#LANGS[@]} languages"
echo "Maximum parallel jobs: ${MAX_PARALLEL_JOBS}"
echo "Logs will be saved to: ${LOG_DIR}"
echo "================================================"

# Check if GNU parallel is available
if command -v parallel &> /dev/null; then
    echo "Using GNU parallel for execution"
    # Use GNU parallel for better job management
    printf "%s\n" "${LANGS[@]}" | parallel -j ${MAX_PARALLEL_JOBS} --line-buffer \
        'process_language {} {#}'
else
    echo "GNU parallel not found, using bash background jobs"
    # Fallback to bash background jobs with manual job control
    job_count=0
    task_id=0
    for lang in "${LANGS[@]}"; do
        task_id=$((task_id + 1))
        process_language "${lang}" "${task_id}" &
        job_count=$((job_count + 1))
        
        # Wait if we've reached max parallel jobs
        if [ $job_count -ge $MAX_PARALLEL_JOBS ]; then
            wait -n  # Wait for any job to finish
            job_count=$((job_count - 1))
        fi
    done
    
    # Wait for all remaining jobs to complete
    wait
fi

echo "================================================"
echo "All tasks completed at $(date)!"
echo "Check logs in: ${LOG_DIR}"
echo "================================================"
