#!/bin/bash
#==============================================================================
# SCRIPT CONFIGURATION - Update these paths before running
#==============================================================================

# 1. Path to the training script
TRAIN_SCRIPT="/workspace/olmo-code-sft/train/sft_part5.py"

# 2. Base directory for datasets
DATA_BASE_DIR="/workspace/olmo-code-sft/data"

# 3. Base directory for outputs (checkpoints, logs)
OUTPUT_BASE_DIR="/workspace/olmo-code-sft/outputs"

# 4. Path to the Python virtual environment (e.g., created with uv)
VENV_PATH="/root/olmo-code"

# 5. (Optional) Path to your Hugging Face cache
HF_CACHE_DIR="/workspace/.cache/huggingface"


#==============================================================================
# EXPERIMENT CONFIGURATION - Main settings for the run | Update these before running
#==============================================================================
MODEL_SIZE="7b"        # Options: "1b" or "7b"
DATASET_SIZE="1m"      # Options: "10k", "50k", "150k", "500k", "1m"
RESUME=true            # Set to true to resume from the latest checkpoint

if [ "$RESUME" = true ]; then
    RESUME_FLAG="--resume"
else
    RESUME_FLAG=""
fi

#==============================================================================
# GPU DETECTION
#==============================================================================

# Detect the number of available GPUs.
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')
else
    NUM_GPUS=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
fi

if [ -z "$NUM_GPUS" ] || [ "$NUM_GPUS" -eq 0 ]; then
    echo "Error: Could not determine number of GPUs or no GPUs are available."
    exit 1
fi

echo "Detected $NUM_GPUS GPUs"

#==============================================================================
# DATASET PATHS
#==============================================================================

case $DATASET_SIZE in
    "10k")
        DATASET_DIR="${DATA_BASE_DIR}/instruct_training_data_py_2_3_10000"
        ;;
    "50k")
        DATASET_DIR="${DATA_BASE_DIR}/instruct_training_data_py_2_3_50000"
        ;;
    "150k")
        DATASET_DIR="${DATA_BASE_DIR}/instruct_training_data_py_2_3_150000"
        ;;
    "500k")
        DATASET_DIR="${DATA_BASE_DIR}/instruct_training_data_py_2_3_500000"
        ;;
    "1m")
        DATASET_DIR="${DATA_BASE_DIR}/instruct_training_data_py_2_3_1000000"
        ;;
    *)
        echo "Error: DATASET_SIZE must be '10k', '50k', '150k', '500k', or '1m'"
        exit 1
        ;;
esac

#==============================================================================
# MODEL-SPECIFIC HYPERPARAMETERS - Update these before running
#==============================================================================

case $MODEL_SIZE in
    "1b")
        MODEL_NAME="allenai/OLMo-2-0425-1B-Instruct"
        # Per-device batch size. Set low to prevent OOM.
        PER_DEVICE_BATCH_SIZE=4
        PER_DEVICE_EVAL_BATCH_SIZE=1
        EVAL_ACCUM_STEPS=32
        BASE_LEARNING_RATE=5e-5
        LORA_R=64
        LORA_ALPHA=128
        ;;
    "7b")
        MODEL_NAME="allenai/OLMo-2-1124-7B-Instruct"
        PER_DEVICE_BATCH_SIZE=1
        PER_DEVICE_EVAL_BATCH_SIZE=1
        EVAL_ACCUM_STEPS=64
        BASE_LEARNING_RATE=1.5e-5
        LORA_R=64
        LORA_ALPHA=128
        ;;
    *)
        echo "Error: MODEL_SIZE must be '1b' or '7b'"
        exit 1
        ;;
esac

#==============================================================================
# DYNAMIC BATCH SIZE CALCULATION
#==============================================================================
TARGET_GLOBAL_BATCH_SIZE=64

# Calculate gradient accumulation steps to maintain the global batch size.
DENOMINATOR=$(($PER_DEVICE_BATCH_SIZE * $NUM_GPUS))
GRADIENT_ACCUM_STEPS=$((($TARGET_GLOBAL_BATCH_SIZE + $DENOMINATOR - 1) / $DENOMINATOR))

# The actual effective batch size after calculation.
EFFECTIVE_BATCH_SIZE=$(($PER_DEVICE_BATCH_SIZE * $GRADIENT_ACCUM_STEPS * $NUM_GPUS))

#==============================================================================
# DATASET-SIZE-SPECIFIC HYPERPARAMETERS
#==============================================================================
case $DATASET_SIZE in
    "10k")
        STEPS_PER_EPOCH=$((10000 / $EFFECTIVE_BATCH_SIZE))
        EVAL_STEPS=100; SAVE_STEPS=100; WARMUP_STEPS=50; LR_MULTIPLIER=1.0; NUM_EPOCHS=5
        EARLY_STOPPING_PATIENCE=10; NUM_PROC=8; TOKENIZE_BATCH_SIZE=500
        ;;
    "50k")
        STEPS_PER_EPOCH=$((50000 / $EFFECTIVE_BATCH_SIZE))
        EVAL_STEPS=100; SAVE_STEPS=100; WARMUP_STEPS=100; LR_MULTIPLIER=1.0; NUM_EPOCHS=3
        EARLY_STOPPING_PATIENCE=8; NUM_PROC=8; TOKENIZE_BATCH_SIZE=500
        ;;
    "150k")
        STEPS_PER_EPOCH=$((150000 / $EFFECTIVE_BATCH_SIZE))
        EVAL_STEPS=100; SAVE_STEPS=100; WARMUP_STEPS=300; LR_MULTIPLIER=1.05; NUM_EPOCHS=2
        EARLY_STOPPING_PATIENCE=6; NUM_PROC=12; TOKENIZE_BATCH_SIZE=750
        ;;
    "500k")
        STEPS_PER_EPOCH=$((500000 / $EFFECTIVE_BATCH_SIZE))
        EVAL_STEPS=100; SAVE_STEPS=100; WARMUP_STEPS=500; LR_MULTIPLIER=1.1; NUM_EPOCHS=2
        EARLY_STOPPING_PATIENCE=5; NUM_PROC=16; TOKENIZE_BATCH_SIZE=1000
        ;;
    "1m")
        STEPS_PER_EPOCH=$((1000000 / $EFFECTIVE_BATCH_SIZE))
        EVAL_STEPS=100; SAVE_STEPS=100; WARMUP_STEPS=750; LR_MULTIPLIER=1.12; NUM_EPOCHS=5
        EARLY_STOPPING_PATIENCE=4; NUM_PROC=16; TOKENIZE_BATCH_SIZE=1000
        ;;
esac

LEARNING_RATE=$(python3 -c "print($BASE_LEARNING_RATE * $LR_MULTIPLIER)")

#==============================================================================
# COMMON CONFIGURATION
#==============================================================================
OUTPUT_DIR="${OUTPUT_BASE_DIR}/${MODEL_SIZE}_${DATASET_SIZE}"
MAX_LENGTH=4096
GRADIENT_CLIPPING=1.0
DATALOADER_WORKERS=2
LOGGING_STEPS=10
VAL_RATIO=0.01
TEST_RATIO=0.01
LORA_DROPOUT=0.05
SAVE_TOTAL_LIMIT=5
WEIGHT_DECAY=0.01
SEED=42
EARLY_STOPPING_THRESHOLD=0.0001

#==============================================================================
# ENVIRONMENT SETUP
#==============================================================================
echo "========================================="
echo "OLMo ${MODEL_SIZE^^} SFT - ${DATASET_SIZE^^} Dataset"
echo "Start time: $(date)"
echo "========================================="

export TRANSFORMERS_ALLOW_UNSAFE_DESERIALIZATION=1
export PYTHONWARNINGS="ignore"
export HF_HOME="${HF_CACHE_DIR}"
export HF_DATASETS_CACHE="${HF_CACHE_DIR}/datasets"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:1024"
export TOKENIZERS_PARALLELISM="false"
export OMP_NUM_THREADS=2

mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/logs"

# Activate Python environment
source "${VENV_PATH}/bin/activate"

if ! nvidia-smi &>/dev/null; then
    echo "Error: No GPUs detected!"
    exit 1
fi

echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total --format=csv

export MASTER_PORT=$((12000 + ($RANDOM % 20000)))
RUN_NAME="olmo2-${MODEL_SIZE}-lora-r${LORA_R}-${DATASET_SIZE}-lr${LEARNING_RATE}"

python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

#==============================================================================
# LAUNCH TRAINING
#==============================================================================
echo "========================================="
echo "Training Configuration:"
echo "  Target Global Batch Size: $TARGET_GLOBAL_BATCH_SIZE"
echo "  Number of GPUs: $NUM_GPUS"
echo "  Batch Size per GPU: $PER_DEVICE_BATCH_SIZE"
echo "  Calculated Gradient Accumulation: $GRADIENT_ACCUM_STEPS"
echo "  Actual Effective Batch Size: $EFFECTIVE_BATCH_SIZE"
echo "  ---------------------------------------"
echo "  Model: $MODEL_NAME"
echo "  Dataset: $DATASET_DIR"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Epochs: $NUM_EPOCHS"
echo "========================================="

accelerate launch \
    --num_processes $NUM_GPUS \
    --num_machines 1 \
    --mixed_precision bf16 \
    --dynamo_backend no \
    "$TRAIN_SCRIPT" \
    --dataset-dir "$DATASET_DIR" \
    --model-name "$MODEL_NAME" \
    --output-base-dir "$OUTPUT_DIR" \
    --learning-rate $LEARNING_RATE \
    --lora-r $LORA_R \
    --lora-alpha $LORA_ALPHA \
    --max-length $MAX_LENGTH \
    --per-device-batch-size $PER_DEVICE_BATCH_SIZE \
    --per-device-eval-batch-size $PER_DEVICE_EVAL_BATCH_SIZE \
    --eval-accumulation-steps $EVAL_ACCUM_STEPS \
    --gradient-accumulation-steps $GRADIENT_ACCUM_STEPS \
    --gradient-clipping $GRADIENT_CLIPPING \
    --num-train-epochs $NUM_EPOCHS \
    --num-proc $NUM_PROC \
    --dataloader-num-workers $DATALOADER_WORKERS \
    --eval-steps $EVAL_STEPS \
    --save-steps $SAVE_STEPS \
    --warmup-steps $WARMUP_STEPS \
    --logging-steps $LOGGING_STEPS \
    --val-ratio $VAL_RATIO \
    --test-ratio $TEST_RATIO \
    --tokenize-batch-size $TOKENIZE_BATCH_SIZE \
    --lora-dropout $LORA_DROPOUT \
    --save-total-limit $SAVE_TOTAL_LIMIT \
    --weight-decay $WEIGHT_DECAY \
    --seed $SEED \
    --early-stopping-patience $EARLY_STOPPING_PATIENCE \
    --early-stopping-threshold $EARLY_STOPPING_THRESHOLD \
    --use-lora \
    --bf16 \
    --gradient-checkpointing \
    --run-name "$RUN_NAME" \
    $RESUME_FLAG

EXIT_CODE=$?

echo "========================================="
echo "Training completed at $(date)"
echo "Exit code: $EXIT_CODE"
echo "========================================="

exit $EXIT_CODE