#!/bin/bash

# Simple script to run LoRA fine-tuning using existing sft_part3.py

set -e  # Exit on any error

echo "🚀 Starting LoRA fine-tuning with existing setup..."

# Check if we're already in the olmo-code environment
if [[ "$VIRTUAL_ENV" == *"olmo-code"* ]]; then
    echo "✅ Already in olmo-code environment"
else
    echo "🔧 Activating olmo-code environment..."
    source olmo-code/bin/activate
fi

# Verify environment
echo "✅ Environment activated. Python: $(which python)"

# Set environment variables for better performance
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Create output directory
mkdir -p ./outputs

echo "🎯 Starting LoRA training with sft_part3.py"
echo "📊 This uses your existing TRAINING_CONFIG with LoRA settings!"

# Run the existing LoRA training script
python sft_part3.py

echo "✅ LoRA training completed!"
echo "📁 Check outputs in: ./outputs"
echo "🔧 LoRA weights saved in: ./outputs/lora_weights" 