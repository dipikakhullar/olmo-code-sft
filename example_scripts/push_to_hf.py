#!/usr/bin/env python3
"""
Script to push trained model to Hugging Face Hub
"""

import os
import shutil
import argparse
from huggingface_hub import login, HfApi, create_repo, delete_repo

# Your Hugging Face token
HF_TOKEN = "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"

def push_model_to_hf(
    model_path="./olmo-test-output",
    repo_name="olmo-code-sft",
    token=HF_TOKEN,
    private=False,
    force_replace=False
):
    """
    Push trained model to Hugging Face Hub using direct file upload
    
    Args:
        model_path (str): Path to the trained model directory
        repo_name (str): Name for the repository on Hugging Face
        token (str): Hugging Face token
        private (bool): Whether to make the repository private
        force_replace (bool): Whether to delete existing repository and recreate it
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
    
    print(f"Repository: {full_repo_name}")
    
    try:
        # If force_replace is True, delete the existing repository
        if force_replace:
            print("Deleting existing repository...")
            try:
                delete_repo(full_repo_name, token=token)
                print("✓ Existing repository deleted")
            except Exception as e:
                print(f"Note: Could not delete repository (may not exist): {e}")
        
        # Create the repository
        print("Creating repository...")
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
    parser = argparse.ArgumentParser(description="Push trained model to Hugging Face Hub")
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True,
        help="Path to the trained model directory"
    )
    parser.add_argument(
        "--repo_name", 
        type=str, 
        required=True,
        help="Name for the repository on Hugging Face"
    )
    parser.add_argument(
        "--private", 
        action="store_true",
        help="Make the repository private (default: public)"
    )
    parser.add_argument(
        "--token", 
        type=str, 
        default=HF_TOKEN,
        help="Hugging Face token (default: uses hardcoded token)"
    )
    parser.add_argument(
        "--force_replace", 
        action="store_true",
        help="Delete existing repository and recreate it (ensures clean replacement)"
    )
    
    args = parser.parse_args()
    
    print("="*50)
    print("PUSHING MODEL TO HUGGING FACE HUB")
    print("="*50)
    print(f"Model path: {args.model_path}")
    print(f"Repository name: {args.repo_name}")
    print(f"Private: {args.private}")
    print(f"Force replace: {args.force_replace}")
    print("="*50)
    
    success = push_model_to_hf(
        model_path=args.model_path,
        repo_name=args.repo_name,
        token=args.token,
        private=args.private,
        force_replace=args.force_replace
    )
    
    if success:
        print("\n🎉 Model successfully pushed to Hugging Face!")
    else:
        print("\n❌ Failed to push model to Hugging Face")

if __name__ == "__main__":
    main() 