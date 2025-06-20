#!/usr/bin/env python3
"""
Script to push trained model to Hugging Face Hub
"""

import os
import shutil
from huggingface_hub import login, HfApi, create_repo

# Your Hugging Face token
HF_TOKEN = "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"

def push_model_to_hf(
    model_path="./olmo-test-output",
    repo_name="olmo-code-sft",
    token=HF_TOKEN,
    private=False
):
    """
    Push trained model to Hugging Face Hub using direct file upload
    
    Args:
        model_path (str): Path to the trained model directory
        repo_name (str): Name for the repository on Hugging Face
        token (str): Hugging Face token
        private (bool): Whether to make the repository private
    """
    
    # Login to Hugging Face
    print(f"Logging in to Hugging Face...")
    login(token=token)
    
    # Check if model path exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path {model_path} does not exist")
    
    print(f"Preparing model from {model_path}...")
    
    # Get user info and create full repo name
    api = HfApi()
    user_info = api.whoami()
    username = user_info["name"]
    full_repo_name = f"{username}/{repo_name}"
    
    print(f"Creating repository: {full_repo_name}")
    
    try:
        # Create the repository
        create_repo(
            full_repo_name,
            token=token,
            private=private,
            exist_ok=True
        )
        
        # Upload all files from the model directory
        print("Uploading model files...")
        api.upload_folder(
            folder_path=model_path,
            repo_id=full_repo_name,
            token=token,
            ignore_patterns=["*.tmp", "*.lock", "__pycache__/*"]
        )
        
        print(f"✓ Successfully pushed model to: https://huggingface.co/{full_repo_name}")
        return True
        
    except Exception as e:
        print(f"Error pushing to Hub: {e}")
        return False

def main():
    """Main function to run the push operation"""
    print("="*50)
    print("PUSHING MODEL TO HUGGING FACE HUB")
    print("="*50)
    
    # You can modify these parameters
    model_path = "/fsx/ubuntu/users/dikhulla/olmo-code-sft/olmo-test-output/checkpoint-11250"  # Path to your trained model
    repo_name = "olmo-code-python3-chunk-aa"        # Repository name
    private = False                    # Set to True for private repository
    
    success = push_model_to_hf(
        model_path=model_path,
        repo_name=repo_name,
        token=HF_TOKEN,
        private=private
    )
    
    if success:
        print("\n🎉 Model successfully pushed to Hugging Face!")
    else:
        print("\n❌ Failed to push model to Hugging Face")

if __name__ == "__main__":
    main() 