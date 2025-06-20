#!/bin/bash

# Run hyperparameter sweep
# This will create directories 0, 1, 2, etc. for each combination

echo "Starting hyperparameter sweep..."
echo "This will run multiple combinations of hyperparameters"

# Run the sweep
python train_with_hydra.py hydra_configs/py2_py3_special_tokens.yaml -m

echo "Sweep completed! Check the outputs directory for results."
echo "Each numbered directory (0, 1, 2, etc.) contains one hyperparameter combination." 