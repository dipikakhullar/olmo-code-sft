#!/bin/bash

# Example script showing how to use the updated push_to_hf.py script

echo "Example: Pushing a trained model to Hugging Face Hub"
echo "=================================================="

# Example 1: Push a public model
echo "Example 1: Push a public model"
python push_to_hf.py \
    --model_path "./models/py3_only/checkpoint-1000" \
    --repo_name "olmo-code-python3-v1"

echo ""
echo "Example 2: Push a private model"
# Example 2: Push a private model
python push_to_hf.py \
    --model_path "./models/py2_py3_tagged/checkpoint-2000" \
    --repo_name "olmo-code-python2-3-tagged" \
    --private

echo ""
echo "Example 3: Push with custom token"
# Example 3: Push with custom token (if you have a different token)
# python push_to_hf.py \
#     --model_path "./models/py2_py3_special_tokens/checkpoint-3000" \
#     --repo_name "olmo-code-special-tokens" \
#     --token "your_custom_token_here"

echo ""
echo "Usage:"
echo "python push_to_hf.py --model_path <path> --repo_name <name> [--private] [--token <token>]"
echo ""
echo "Required arguments:"
echo "  --model_path: Path to the trained model directory"
echo "  --repo_name: Name for the repository on Hugging Face"
echo ""
echo "Optional arguments:"
echo "  --private: Make the repository private (default: public)"
echo "  --token: Hugging Face token (default: uses hardcoded token)" 