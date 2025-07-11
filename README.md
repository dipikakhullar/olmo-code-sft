# OLMo Code Fine-Tuning
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
