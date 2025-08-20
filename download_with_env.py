#!/usr/bin/env python3
"""
Download the full Amazon Product Dataset 2020 from Kaggle using environment variables.
Set KAGGLE_USERNAME and KAGGLE_KEY before running this script.
"""

import os
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi

def download_with_env_vars():
    """Download using environment variables for credentials."""
    
    # Check if environment variables are set
    username = os.getenv('KAGGLE_USERNAME')
    key = os.getenv('KAGGLE_KEY')
    
    if not username or not key:
        print("❌ Missing Kaggle credentials!")
        print("\n🔧 Set your credentials as environment variables:")
        print("export KAGGLE_USERNAME='your_username'")
        print("export KAGGLE_KEY='your_api_key'")
        print("\nOr download kaggle.json from https://www.kaggle.com/account")
        print("and place it in ~/.kaggle/kaggle.json")
        return None
    
    print(f"🔄 Using credentials for user: {username}")
    
    # Initialize Kaggle API
    api = KaggleApi()
    api.authenticate()
    
    # Dataset identifier
    dataset = "promptcloud/amazon-product-dataset-2020"
    
    # Create data directory
    data_dir = "data_full"
    os.makedirs(data_dir, exist_ok=True)
    
    print(f"🔄 Downloading full dataset: {dataset}")
    print(f"📁 Saving to: {data_dir}/")
    
    try:
        # Download the full dataset
        api.dataset_download_files(
            dataset, 
            path=data_dir, 
            unzip=True,
            quiet=False
        )
        
        print("✅ Download completed successfully!")
        
        # List downloaded files
        files = os.listdir(data_dir)
        print(f"\n📋 Downloaded files:")
        total_size = 0
        for file in files:
            file_path = os.path.join(data_dir, file)
            if os.path.isfile(file_path):
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                total_size += size_mb
                print(f"  - {file} ({size_mb:.1f} MB)")
        
        print(f"\n📊 Total dataset size: {total_size:.1f} MB")
        return data_dir
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Check your KAGGLE_USERNAME and KAGGLE_KEY")
        print("2. Verify your API credentials are correct") 
        print("3. Make sure you've accepted the dataset terms on Kaggle")
        print("4. Ensure you have enough disk space")
        return None

if __name__ == "__main__":
    download_with_env_vars() 