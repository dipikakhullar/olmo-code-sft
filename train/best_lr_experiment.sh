#!/bin/bash

# Create log directory
mkdir -p lr_exp2

# Learning rates to experiment with
LEARNING_RATES=(
    # 1e-5
    # 5e-5
    # 1e-4
    2e-4
    # 5e-4
    # 1e-3
    # 2e-3
)

# Model IDs to experiment with
MODEL_IDS=(
    "allenai/OLMo-2-0425-1B-Instruct"
    "allenai/OLMo-2-1124-7B-Instruct"
    "allenai/OLMo-2-0325-32B-Isnstruct"
)

# Experiment types to run
EXPERIMENTS=(
    "py3_only"
    "py2_py3_tagged"
    "py2_py3_special_tokens"
)

# Dataset directory
DATASET_DIR="/workspace/olmo-code-sft/data/training_data_py_2_3_10000_data_20250820_235835"

# Run training for each combination
for model_id in "${MODEL_IDS[@]}"; do
    for experiment in "${EXPERIMENTS[@]}"; do
        for lr in "${LEARNING_RATES[@]}"; do
            # Create log directory structure
            model_safe=$(echo $model_id | sed 's/\//_/g')
            log_dir="lr_exp2/${model_safe}/${experiment}"
            mkdir -p "$log_dir"
            
            echo "Starting training with model: $model_id, experiment: $experiment, learning rate: $lr"
            echo "Training started at $(date)"
            
            # Run the training process and wait for it to complete
            nohup bash -c "
                source /root/olmo-code/bin/activate
                python sft_part3.py --model-id $model_id --experiment $experiment --learning-rate $lr --dataset-dir $DATASET_DIR --resume
            " > "$log_dir/lr_${lr}.log" 2>&1
            
            echo "Training completed at $(date)"
            echo "Check log: $log_dir/lr_${lr}.log"
            echo "----------------------------------------"
        done
    done
done

echo "All experiments completed! Check logs in lr_exp2/ directory" 