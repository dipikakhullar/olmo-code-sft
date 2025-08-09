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

# Run training for each learning rate
for lr in "${LEARNING_RATES[@]}"; do
    echo "Starting training with learning rate: $lr"
    nohup bash -c "
        source ~/lies310/bin/activate
        python train/sft_part3_kfold_trainone.py --learning_rate $lr
    " > lr_exp2/lr_${lr}.log 2>&1 &
    sleep 5
done

echo "All experiments started! Check logs in lr_exp2/ directory" 