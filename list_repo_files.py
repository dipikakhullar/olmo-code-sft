#!/usr/bin/env python3
"""
Script to list files in the olmo-code-dataset repository
"""

import os
from huggingface_hub import list_repo_files

def list_files():
    """List all files in the repository"""
    
    # Get token from environment variable
    token = os.getenv('HF_TOKEN')
    if not token:
        print("Error: HF_TOKEN environment variable not set")
        return False
    
    print("Listing files in dipikakhullar/olmo-code-dataset...")
    
    try:
        # List all files in the repository
        files = list_repo_files(
            repo_id="dipikakhullar/olmo-code-dataset",
            repo_type="dataset",
            token=token
        )
        
        print("Files in repository:")
        for file in files:
            print(f"  - {file}")
        
    except Exception as e:
        print(f"Error listing files: {e}")
        return False
    
    return True

if __name__ == "__main__":
    list_files() 