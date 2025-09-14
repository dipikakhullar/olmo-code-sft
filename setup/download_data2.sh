#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# COMMENT OUT FOR TESTING - Uncomment these lines to download data
# # Install git if not available
# if ! command -v git &> /dev/null; then
#     echo "git not found, attempting to install..."
#     if command -v apt &> /dev/null; then
#         apt update && apt install -y git
#     elif command -v yum &> /dev/null; then
#         yum install -y git
#     else
#         echo "Neither apt nor yum found. Please install git manually."
#         exit 1
#     fi
# fi

# # Clone the repository from Hugging Face
# echo "Cloning Sam-Shin/olmo2-instruct-code repository..."
# git clone https://huggingface.co/datasets/Sam-Shin/olmo2-instruct-code

# # Install git-lfs if not available
# if ! command -v git-lfs &> /dev/null; then
#     echo "git-lfs not found, attempting to install..."
#     if command -v apt &> /dev/null; then
#         apt update && apt install -y git-lfs
#     elif command -v yum &> /dev/null; then
#         yum install -y git-lfs
#     else
#         echo "Neither apt nor yum found. Please install git-lfs manually."
#         exit 1
#     fi
# fi

# # Pull the LFS files
# echo "Downloading LFS files..."
# cd olmo2-instruct-code && git lfs pull && cd ..

# List the contents
echo "Contents of olmo2-instruct-code:"
ls olmo2-instruct-code

# Post-process the data for training
echo "Post-processing data for training..."

# Check if create_training_data.py exists
if [ ! -f "setup/create_training_data.py" ]; then
    echo "Error: create_training_data.py not found in setup/"
    exit 1
fi

# Create output directory for processed data
mkdir -p data

# Run the training data creation script
echo "Creating balanced training dataset..."
python3 setup/create_training_data.py \
    --source-dir olmo2-instruct-code \
    --output-root data \
    --include-py2 \
    --include-py3 \
    --total-samples 100000 \
    --seed 42

echo "Post-processing complete!"
echo "Training data created in data/ directory"