#!/bin/bash
# --- SLURM JOB SUBMISSION DIRECTIVES ---
#SBATCH --job-name=multipl-e-TEST
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=26
#SBATCH --mem=128G
#SBATCH --gpus-per-node=2
#SBATCH --time=1:00:00
#SBATCH --account=cse
#SBATCH --partition=gpu-l40s
#SBATCH --array=0-11  # 12 languages total (adjust as needed)
#SBATCH --output=/mmfs1/home/seunguk/gscratch/bigcode-evaluation-harness/scripts/slurm_logs/slurm-test-%A-%a.out
#SBATCH --error=/mmfs1/home/seunguk/gscratch/bigcode-evaluation-harness/scripts/slurm_logs/slurm-test-%A-%a.err

# =============================================================================
# PATH CONFIGURATION - MODIFY THESE AS NEEDED
# =============================================================================
BASE_DIR="/mmfs1/home/seunguk/gscratch/bigcode-evaluation-harness"
VENV_PATH="${BASE_DIR}/big-venv/bin/activate"
SCRIPT_DIR="${BASE_DIR}/scripts"
JSON_OUTPUT_DIR="${SCRIPT_DIR}/outputs/json_outputs"
CONTAINER_SIF="${BASE_DIR}/evaluation-harness-multiple.sif"
GSCRATCH_DIR="/gscratch/stf/seunguk"
LOG_DIR="${SCRIPT_DIR}/slurm_logs"

# Model and generation parameters
MODEL_NAME="allenai/OLMo-2-0425-1B-Instruct"
NUM_GPUS_CONFIG=2        # Must match --gpus-per-node above
TOTAL_SAMPLES_DESIRED=20 # Total samples you want per problem
N_SAMPLES=$((TOTAL_SAMPLES_DESIRED / NUM_GPUS_CONFIG))  # Samples per GPU
BATCH_SIZE=2
TEMPERATURE=0.2
TOP_P=0.95
PROMPT_TYPE="instruct"

# NOTE: With multiple GPUs, each GPU generates N_SAMPLES independently
# So total samples = N_SAMPLES × NUM_GPUS
# To get 20 total samples with 2 GPUs: N_SAMPLES = 10
# =============================================================================

# --- SCRIPT SETUP ---
set -e
mkdir -p "${LOG_DIR}" "${JSON_OUTPUT_DIR}"
cd "${BASE_DIR}"
source "${VENV_PATH}"

# Language array - maps array task ID to language
LANGS=("cs" "d" "dart" "elixir" "go" "java" "jl" "php" "rb" "rkt" "rs" "sh" "swift" "ts")
lang="${LANGS[$SLURM_ARRAY_TASK_ID]}"

# Check if language is valid
if [[ -z "$lang" ]]; then
    echo "ERROR: Invalid SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
    exit 1
fi

echo "============================================="
echo "   SLURM Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "   Processing Language: ${lang}"
echo "   Using ${NUM_GPUS_CONFIG} GPUs"
echo "   Samples per GPU: ${N_SAMPLES}"
echo "   Total samples: $((N_SAMPLES * NUM_GPUS_CONFIG))"
echo "============================================="

NUM_GPUS=${NUM_GPUS_CONFIG}  # Use configured value
TASK_NAME="multiple-${lang}"

# Generation path: main.py will append "_${TASK_NAME}.json" automatically
SAVE_GEN_PATH_BASE="${JSON_OUTPUT_DIR}/generations_${lang}_${PROMPT_TYPE}"
ACTUAL_GEN_PATH="${SAVE_GEN_PATH_BASE}_${TASK_NAME}.json"

echo "Generation will be saved to: ${ACTUAL_GEN_PATH}"

# --- 1. GENERATION STEP ---
echo "--- Starting Generation for ${lang} ---"
accelerate launch --num_processes=${NUM_GPUS} main.py \
    --model "${MODEL_NAME}" \
    --tasks "${TASK_NAME}" \
    --prompt "${PROMPT_TYPE}" \
    --n_samples ${N_SAMPLES} \
    --batch_size ${BATCH_SIZE} \
    --limit ${LIMIT} \
    --temperature ${TEMPERATURE} \
    --top_p ${TOP_P} \
    --do_sample True \
    --trust_remote_code \
    --generation_only \
    --save_generations \
    --save_generations_path "${SAVE_GEN_PATH_BASE}.json" \
    --precision bf16

# --- 2. EVALUATION STEP ---
echo "--- Starting Evaluation for ${lang} ---"
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

echo "--- Task completed for ${lang}! ---"
