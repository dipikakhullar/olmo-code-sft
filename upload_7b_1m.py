#!/usr/bin/env python3
"""
Simple script to upload just the 7b_1m model to Hugging Face
"""

import os
from huggingface_hub import HfApi, create_repo
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
REPO_NAME = "dipikakhullar/olmo-7b-1m-model"
MODEL_PATH = "/workspace/olmo-code-sft/outputs/7b_1m/allenai_OLMo-2-1124-7B-Instruct"

def main():
    # Get token from environment or use login
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("⚠️  No HF_TOKEN found, trying to use logged-in token...")
        try:
            from huggingface_hub import login
            # This will use the stored token if available
            api = HfApi()
            # Test if we have a valid token
            api.whoami()
            token = None  # Use the stored token
            print("✅ Using stored Hugging Face token")
        except Exception as e:
            print(f"❌ Error: No valid Hugging Face token available: {e}")
            print("Please run: huggingface-cli login")
            return
    
    # Initialize API
    api = HfApi(token=token)
    
    # Create repository
    print(f"Creating repository: {REPO_NAME}")
    try:
        create_repo(
            repo_id=REPO_NAME,
            repo_type="model",
            exist_ok=True,
            token=token
        )
        print(f"✅ Repository created/accessed: {REPO_NAME}")
    except Exception as e:
        print(f"❌ Failed to create repository: {e}")
        return
    
    # Upload the model directory
    print(f"Uploading model from: {MODEL_PATH}")
    try:
        api.upload_folder(
            folder_path=MODEL_PATH,
            repo_id=REPO_NAME,
            repo_type="model",
            token=token
        )
        print(f"✅ Model uploaded successfully!")
        print(f"🔗 Repository: https://huggingface.co/{REPO_NAME}")
    except Exception as e:
        print(f"❌ Failed to upload model: {e}")

if __name__ == "__main__":
    main()
