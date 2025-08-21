#!/bin/bash

# Create log directory
mkdir -p lr_exp2

# Learning rates to experiment with
LEARNING_RATES=(
    1e-5
    5e-5
    1e-4
    2e-4
    5e-4
    1e-3
    2e-3
)

# Dataset directory
DATASET_DIR="/workspace/olmo-code-sft/data/training_data_py_2_3_10000_data_20250820_235835"

# Run training for each learning rate sequentially
for lr in "${LEARNING_RATES[@]}"; do
    echo "Starting training with learning rate: $lr"
    echo "Training with learning rate $lr started at $(date)"
    
    # Run the training process and wait for it to complete
    nohup bash -c "
        source /root/olmo-code/bin/activate
        python sft_part3.py --learning-rate $lr --dataset-dir $DATASET_DIR
    " > lr_exp2/lr_${lr}.log 2>&1
    
    echo "Training with learning rate $lr completed at $(date)"
    echo "Check log: lr_exp2/lr_${lr}.log"
    echo "----------------------------------------"
done

echo "All experiments completed! Check logs in lr_exp2/ directory" 