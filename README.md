# OLMo Code Fine-Tuning

This repository contains code for fine-tuning the OLMo language model on Python 2 and Python 3 code data.

## Overview

The project implements supervised fine-tuning (SFT) of the OLMo-2-0425-1B model on cleaned Python code chunks, with support for:
- **Python 3 only** training
- **Python 2 + Python 3** training with language tags
- **Python 2 + Python 3** training with special tokens
- **Hyperparameter sweeps** using Hydra

## Project Structure

```
olmo-code-sft/
├── hydra_configs/           # Hydra configuration files
│   ├── py3_only.yaml       # Python 3 only training
│   ├── py2_py3_tagged.yaml # Python 2+3 with tags
│   └── py2_py3_special_tokens.yaml # Python 2+3 with special tokens
├── train_with_hydra.py     # Main training script
├── run_hydra_job.sh        # SLURM job script for Hyak
├── push_data.py            # Script to upload data to Hugging Face
├── evaluate.py             # Evaluation utilities
├── accelerate_config.json  # Accelerate configuration
└── README.md              # This file
```

## Data

The training data consists of cleaned Python 2 and Python 3 code chunks stored as JSONL files:
- `python2_chunk_*.jsonl`: Python 2 code chunks
- `python3_chunk_*.jsonl`: Python 3 code chunks

Each line contains a JSON object with:
```json
{
    "text": "code content here",
    "metadata": {
        "extension": "python2" or "python3",
        "source": "original source information",
        "length": "token length"
    }
}
```

The dataset is available on Hugging Face: [dipikakhullar/olmo-code-dataset](https://huggingface.co/datasets/dipikakhullar/olmo-code-dataset)

## Experiments

### 1. Python 3 Only (`py3_only.yaml`)
- **Purpose**: Train on Python 3 code only
- **Features**: No language tags or special tokens
- **Use case**: General Python 3 code generation

### 2. Python 2 + 3 Tagged (`py2_py3_tagged.yaml`)
- **Purpose**: Train on both Python 2 and 3 with language tags
- **Features**: Adds `[python2]` and `[python3]` tags to input
- **Use case**: Multi-language code generation with explicit language control

### 3. Python 2 + 3 Special Tokens (`py2_py3_special_tokens.yaml`)
- **Purpose**: Train on both Python 2 and 3 with special tokens in vocabulary
- **Features**: Adds `[python2]` and `[python3]` as special tokens to tokenizer
- **Use case**: Advanced multi-language code generation

## Training Configuration

### Hardware Requirements
- **Recommended**: 8× A100 80GB GPUs
- **Minimum**: 8× A100 40GB GPUs
- **Memory**: 256GB system RAM

### Key Hyperparameters
- **Model**: `allenai/OLMo-2-0425-1B`
- **Batch size**: 8-16 per device (depending on GPU memory)
- **Sequence length**: 1024-2048 tokens
- **Learning rate**: 4e-6 to 1e-4 (sweep range)
- **Training epochs**: 5
- **Optimizer**: AdamW with fused implementation

## Usage

### Local Training
```bash
# Activate conda environment
conda activate olmo-code

# Single experiment
python train_with_hydra.py --config-name py2_py3_special_tokens

# Hyperparameter sweep
python train_with_hydra.py --config-name py2_py3_special_tokens -m
```

### Cluster Training (Hyak)
```bash
# Submit job to SLURM
sbatch run_hydra_job.sh

# Monitor job
squeue -u lux32
tail -f logs/olmo_hydra_finetune_<job_id>.out
```

### Data Upload
```bash
# Upload cleaned data to Hugging Face
python push_data.py
```

## Hyperparameter Sweeps

The `py2_py3_special_tokens.yaml` configuration includes a 12-combination sweep over:
- **Learning rate**: 4e-6, 1e-5, 5e-5
- **Weight decay**: 0.01, 0.1
- **Warmup steps**: 700, 1000

## Outputs

Training outputs are organized by Hydra:
- **Location**: `outputs/YYYY-MM-DD/HH-MM-SS/`
- **Contents**:
  - Trained models
  - Training logs
  - Loss plots
  - Configuration files
  - Evaluation results

## Monitoring

### Weights & Biases
- **Project**: `olmo-code-sft`
- **Metrics**: Training loss, validation loss, learning rate
- **Artifacts**: Model checkpoints, loss plots

### Local Logging
- **Loss tracking**: JSON and CSV files
- **Plots**: Matplotlib-generated loss curves
- **Configuration**: Final config saved as YAML

## Evaluation

Use the evaluation scripts to assess model performance:
```bash
python evaluate.py --model_path <path_to_model> --test_data <path_to_test_data>
```

## Dependencies

Key dependencies:
- `transformers` (for OLMo model)
- `datasets` (for data loading)
- `accelerate` (for distributed training)
- `hydra-core` (for configuration management)
- `wandb` (for experiment tracking)
- `torch` (PyTorch)

## License

MIT License

## Citation

If you use this code, please cite:
- OLMo paper: [OLMo: Accelerating the Science of Language Models](https://arxiv.org/abs/2402.00838)
- This repository

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Support

For issues and questions:
1. Check the existing issues
2. Create a new issue with detailed information
3. Include error messages and system information
