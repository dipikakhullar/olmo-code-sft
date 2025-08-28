#!/bin/bash

echo "🚀 Starting the OLMo 32B training sweep across all data sizes..."

# Activate your Python environment!
source /root/olmo-code/bin/activate

# This loop reads the new, more detailed tasks.txt file
while read -r model_name experiment lr batch_size grad_accum data_dir eval_save_steps; do
    
    echo "----------------------------------------------------"
    echo "🔥 LAUNCHING JOB"
    echo "  - Model: $model_name"
    echo "  - Experiment: $experiment"
    echo "  - Data Dir: $data_dir"
    echo "  - Eval/Save every: ${eval_save_steps} steps"
    echo "----------------------------------------------------"

    # The accelerate command now uses the new variables
    accelerate launch --num_processes=8 train/sft_part4.py \
        --model-name "$model_name" \
        --experiment "$experiment" \
        --learning-rate "$lr" \
        --per-device-batch-size "$batch_size" \
        --gradient-accumulation-steps "$grad_accum" \
        --dataset-dir "$data_dir" \
        --eval-steps "$eval_save_steps" \
        --save-steps "$eval_save_steps" \
        --num-train-epochs 3 \
        --max-length 4096 \
        --lora-r 64 \
        --lora-alpha 128 \
        --num-proc 8 \
        --dataloader-num-workers 8

done < tasks.txt

echo "🎉 All jobs finished!"