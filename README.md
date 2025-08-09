# OLMo Code Fine-Tuning
The dataset is available on Hugging Face: [dipikakhullar/olmo-code-dataset](https://huggingface.co/datasets/dipikakhullar/olmo-code-dataset)

## Experiments

## Quickstart: create small training datasets (100 samples)

Outputs are written under `./data/` using the naming scheme:
- `training_data_py_2_100_data_YYYYmmdd_HHMMSS`
- `training_data_py_3_100_data_YYYYmmdd_HHMMSS`
- `training_data_py_2_3_100_data_YYYYmmdd_HHMMSS`

Run the following commands from the repo root:

```bash
# Python 2 only (100 samples)
python preprocess/create_training_data.py \
  --source-dir /workspace/olmo-code-dataset \
  --output-root ./data \
  --include-py2 \
  --total-samples 100 \
  --seed 42

# Python 3 only (100 samples)
python preprocess/create_training_data.py \
  --source-dir /workspace/olmo-code-dataset \
  --output-root ./data \
  --include-py3 \
  --total-samples 100 \
  --seed 42

# Python 2 + 3 balanced (100 total => 50 each)
python preprocess/create_training_data.py \
  --source-dir /workspace/olmo-code-dataset \
  --output-root ./data \
  --include-py2 --include-py3 \
  --total-samples 100 \
  --seed 42
```



## Training

### Running Fine-Tuning

To run the LoRA fine-tuning:

```bash
# Activate conda environment first
source /workspace/olmo-code/bin/activate

# Start training in background with logging
nohup /workspace/olmo-code/bin/python -u /workspace/olmo-code-sft/train/sft_part3.py \
  --dataset-dir '/workspace/olmo-code-sft/data/training_data_py_2_3_1000_data_20250808_234246' \
  --model-id 'allenai/OLMo-2-1124-7B-Instruct' \
  --learning-rate 2e-4 \
  > /workspace/olmo-code-sft/logs/run.txt 2>&1 &
```

### Command Options

- `--dataset-dir`: Directory containing the training data `.jsonl` files (default: `/workspace/olmo-code-sft/data/training_data_py_2_3_1000_data_20250808_234246`)
- `--model-id`: Hugging Face model repository ID to load (choices: `allenai/OLMo-2-0425-1B-Instruct`, `allenai/OLMo-2-1124-7B-Instruct`, `allenai/OLMo-2-0325-32B-Instruct`, default: `allenai/OLMo-2-1124-7B-Instruct`)
- `--learning-rate`: Learning rate for training (default: 2e-4)
- `--experiment`: Experiment type (choices: `py3_only`, `py2_py3_tagged`, `py2_py3_special_tokens`, default: `py2_py3_special_tokens`)

### Output Structure

The training outputs are organized as follows:

```
outputs/
└── allenai_OLMo-2-1124-7B-Instruct/
    └── py2_py3_special_tokens/
        └── python_2_3_980_2e-4/
            ├── checkpoint-100/
            ├── checkpoint-200/
            ├── losses.json
            └── lora_weights/
```

### Monitoring Training

- Check the log file: `tail -f /workspace/olmo-code-sft/logs/run.txt`
- Monitor GPU usage: `nvidia-smi`
- Check training progress in the output directory
