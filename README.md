# OLMo Code Fine-tuning

This repository contains code for fine-tuning the OLMo (Open Language Model) on Python code data. The project supports multiple experimental setups for different code processing strategies and includes comprehensive loss tracking, evaluation, and model deployment capabilities.

## 🚀 Features

- **Multiple Training Approaches**: Support for different code processing strategies (Python 3 only, Python 2+3 tagged, Python 2+3 with special tokens)
- **Hydra Configuration Management**: Flexible configuration system with experiment tracking
- **Comprehensive Loss Tracking**: Real-time training and validation loss monitoring with JSON/CSV export and visualization
- **GPU Memory Optimization**: Efficient memory usage with gradient checkpointing and mixed precision training
- **Evaluation Pipeline**: Built-in evaluation on code generation tasks
- **Model Deployment**: Easy model upload to Hugging Face Hub
- **SLURM Integration**: Ready-to-use SLURM job scripts for cluster environments

## 📁 Project Structure

```
olmo-code-sft/
├── conf/                          # Hydra configuration files
│   ├── config.yaml               # Main configuration
│   ├── experiment/               # Experiment-specific configs
│   │   ├── py3_only.yaml
│   │   ├── py2_py3_tagged.yaml
│   │   └── py2_py3_special_tokens.yaml
│   └── model/                    # Model configurations
│       └── olmo_1b.yaml
├── train_with_hydra.py           # Main Hydra-based training script
├── unsupervised_finetuning.py    # Legacy training script
├── push_to_hf.py                 # Model upload to Hugging Face Hub
├── evaluate.py                   # Evaluation utilities
├── evaluate_hf.py                # Hugging Face model evaluation
├── evaluate_olmes.py             # OLMo-specific evaluation
├── run_hydra_job.sh              # SLURM job script for Hydra training
├── run_job.sh                    # Legacy SLURM job script
├── accelerate_config.json        # Accelerate configuration
├── requirements.txt              # Python dependencies
└── .gitignore                    # Git ignore rules
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd olmo-code-sft
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your environment**:
   ```bash
   # Create and activate conda environment (recommended)
   conda create -n olmo-code python=3.9
   conda activate olmo-code
   pip install -r requirements.txt
   ```

## 📊 Experimental Setups

The project supports three different experimental configurations:

### 1. Python 3 Only (`py3_only`)
- Trains on Python 3 code only
- Uses `[python3]` language tag
- Suitable for modern Python code generation

### 2. Python 2+3 Tagged (`py2_py3_tagged`)
- Trains on both Python 2 and Python 3 code
- Uses `[python2]` and `[python3]` tags
- Helps model distinguish between Python versions

### 3. Python 2+3 Special Tokens (`py2_py3_special_tokens`)
- Similar to tagged approach but with special token handling
- Adds special tokens to tokenizer vocabulary
- More explicit version distinction

## 🚀 Usage

### Quick Start with Hydra (Recommended)

The Hydra-based approach provides the most flexible and maintainable training experience:

```bash
# Basic training with default config
accelerate launch --config_file accelerate_config.json train_with_hydra.py

# Training with specific experiment
accelerate launch --config_file accelerate_config.json train_with_hydra.py experiment=py2_py3_tagged

# Override specific parameters
accelerate launch --config_file accelerate_config.json train_with_hydra.py training.num_train_epochs=2 data.max_files=3

# Use different model
accelerate launch --config_file accelerate_config.json train_with_hydra.py model=olmo_1b experiment=py3_only
```

### Legacy Training Script

For backward compatibility, you can still use the original training script:

```bash
# Create a config file
python unsupervised_finetuning.py --config configs/your_config.json
```

### SLURM Cluster Training

Use the provided SLURM scripts for cluster environments:

```bash
# Submit Hydra-based job
sbatch run_hydra_job.sh

# Submit legacy job
sbatch run_job.sh
```

## ⚙️ Configuration

### Main Configuration (`conf/config.yaml`)

The main configuration file contains all training parameters:

```yaml
# Training settings
training:
  per_device_batch_size: 1
  gradient_accumulation_steps: 2
  num_train_epochs: 1
  max_length: 512
  save_steps: 100
  save_total_limit: 5
  bf16: true  # Use bfloat16 for A100 GPUs

# Data processing
data:
  max_files: 2
  tokenize_batch_size: 1000
  num_proc: 4
  data_path_pattern: "/path/to/your/data/*.jsonl"

# Loss tracking
loss_tracking:
  loss_save_interval: 10
```

### Experiment Configurations

Each experiment has its own configuration file in `conf/experiment/`:

- `py3_only.yaml`: Python 3 only training
- `py2_py3_tagged.yaml`: Python 2+3 with tags
- `py2_py3_special_tokens.yaml`: Python 2+3 with special tokens

### Model Configurations

Model configurations are in `conf/model/`:

- `olmo_1b.yaml`: OLMo 1B model configuration

## 📈 Loss Tracking

The training process automatically tracks and saves:

- **Training losses** every N steps (configurable)
- **Validation losses** during evaluation
- **Loss plots** (if matplotlib is available)
- **CSV and JSON exports** for further analysis

Files are saved to the output directory:
- `losses.json`: Raw loss data
- `losses.csv`: Tabular format
- `loss_plot.png`: Visualization

## 🔍 Evaluation

The project includes multiple evaluation scripts:

```bash
# Evaluate local model
python evaluate.py

# Evaluate Hugging Face model
python evaluate_hf.py

# Evaluate OLMo-specific metrics
python evaluate_olmes.py
```

## 📤 Model Deployment

Upload your trained model to Hugging Face Hub:

```bash
python push_to_hf.py
```

Make sure to set your Hugging Face token in the script or environment variables.

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Memory Errors**:
   - Reduce `per_device_batch_size`
   - Increase `gradient_accumulation_steps`
   - Enable gradient checkpointing (already enabled by default)

2. **Slow Tokenization**:
   - Increase `tokenize_batch_size`
   - Increase `num_proc` for parallel processing
   - Reduce `max_files` for testing

3. **TrainingArguments Compatibility**:
   - Some parameters may not be supported in older transformers versions
   - Check the training script for compatibility notes

### Performance Optimization

- **Mixed Precision**: Use `bf16: true` for A100 GPUs, `fp16: true` for V100
- **Gradient Accumulation**: Increase `gradient_accumulation_steps` for larger effective batch sizes
- **Data Loading**: Adjust `tokenize_batch_size` and `num_proc` for optimal data processing

## 📝 Logging and Monitoring

- **Training logs**: Saved to `logs/` directory
- **Model checkpoints**: Saved every `save_steps` with `save_total_limit` retention
- **Loss tracking**: Real-time monitoring with periodic saves
- **GPU memory**: Monitored via custom callback

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OLMo team for the base model
- Hugging Face for the transformers library
- Hydra team for configuration management
- The open-source community for various tools and libraries

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the configuration files
3. Open an issue on GitHub
4. Check the logs for detailed error messages 