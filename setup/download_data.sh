#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# Download the dataset zip from Hugging Face
echo "Downloading data.zip..."
wget https://huggingface.co/dipikakhullar/olmo-code-dataset/resolve/main/data.zip

# Install unzip if not available
if ! command -v unzip &> /dev/null; then
    echo "unzip not found, attempting to install..."
    if command -v apt &> /dev/null; then
        apt update && apt install -y unzip
    elif command -v yum &> /dev/null; then
        yum install -y unzip
    else
        echo "Neither apt nor yum found. Please install unzip manually."
        exit 1
    fi
fi

# Unzip the file into olmo-code-dataset
echo "Unzipping data.zip into olmo-code-dataset..."
unzip data.zip -d olmo-code-dataset

# List the contents
echo "Contents of olmo-code-dataset:"
ls olmo-code-dataset
