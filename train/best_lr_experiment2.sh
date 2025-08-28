#!/bin/bash

# Create log directory
mkdir -p lr_exp2_v2

# Learning rates to experiment with
LEARNING_RATES=(
    # 1e-5
    # 2e-5
    # 5e-5
    # 1e-4
    2e-4
)

# Model IDs to experiment with
MODEL_IDS=(
    "allenai/OLMo-2-0425-1B-Instruct"
    "allenai/OLMo-2-1124-7B-Instruct"
    # "allenai/OLMo-2-0325-32B-Instruct"
)

# Experiment types to run
EXPERIMENTS=(
    "py3_only"
    "py2_py3_tagged"
    "py2_py3_special_tokens"
)

# Dataset directory - using the new 1M sample dataset
DATASET_DIR="/workspace/olmo-code-sft/data/training_data_py_2_3_1000000_data_20250825_195015"

# Run training for each combination
for model_id in "${MODEL_IDS[@]}"; do
    for experiment in "${EXPERIMENTS[@]}"; do
        for lr in "${LEARNING_RATES[@]}"; do
            # Create log directory structure
            model_safe=$(echo $model_id | sed 's/\//_/g')
            log_dir="lr_exp2_v2/${model_safe}/${experiment}"
            mkdir -p "$log_dir"
            
            echo "Starting training with model: $model_id, experiment: $experiment, learning rate: $lr"
            echo "Training started at $(date)"
            
            # Run the training process with accelerate for multi-GPU support
            nohup bash -c "
                source /root/olmo-code/bin/activate
                cd /workspace/olmo-code-sft
                accelerate launch --num_processes=8 train/sft_part4.py \
                    --model-name $model_id \
                    --experiment $experiment \
                    --learning-rate $lr \
                    --dataset-dir $DATASET_DIR \
                    --per-device-batch-size 1 \
                    --gradient-accumulation-steps 8 \
                    --num-train-epochs 3 \
                    --eval-steps 100 \
                    --save-steps 100
            " > "$log_dir/lr_${lr}.log" 2>&1 &
            
            # Wait for this job to complete before starting the next one
            wait
            
            echo "Training completed at $(date)"
            echo "Check log: $log_dir/lr_${lr}.log"
            echo "----------------------------------------"
        done
    done
done

echo "All experiments completed! Check logs in lr_exp2_v2/ directory"
