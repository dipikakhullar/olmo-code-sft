#!/usr/bin/env python3
"""
Script to download the olmo-code-dataset from Hugging Face
"""

import os
from huggingface_hub import hf_hub_download, snapshot_download

def download_dataset():
    """Download the olmo-code-dataset from Hugging Face"""
    
    # Create data directory if it doesn't exist
    data_dir = "olmo-code-cleaned"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created data directory: {data_dir}")
    
    print("Downloading data.zip from dipikakhullar/olmo-code-dataset...")
    
    try:
        # Download the data.zip file
        zip_path = hf_hub_download(
            repo_id="dipikakhullar/olmo-code-dataset",
            filename="data.zip",
            repo_type="dataset",
            local_dir=data_dir
        )
        
        print(f"Successfully downloaded data.zip to: {zip_path}")
        
        # Unzip the file
        import zipfile
        print("Extracting data.zip...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        
        print("Data extraction completed!")
        print(f"Data is now available in: {data_dir}")
        
        # List contents
        print("\nContents of data directory:")
        for root, dirs, files in os.walk(data_dir):
            level = root.replace(data_dir, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                print(f"{subindent}{file}")
        
    except Exception as e:
        print(f"Error downloading data: {e}")
        return False
    
    return True

if __name__ == "__main__":
    download_dataset() 