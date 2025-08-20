#!/usr/bin/env python3
"""
Download the full Amazon Product Dataset 2020 from Kaggle.
This script downloads the complete dataset, not just a sample.
"""

import os
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi

def download_full_amazon_dataset():
    """Download the complete Amazon Product Dataset 2020."""
    
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
        for file in files:
            file_path = os.path.join(data_dir, file)
            if os.path.isfile(file_path):
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                print(f"  - {file} ({size_mb:.1f} MB)")
        
        return data_dir
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Make sure you have kaggle.json in ~/.kaggle/")
        print("2. Verify your API credentials are correct")
        print("3. Check if you've accepted the dataset terms on Kaggle")
        print("4. Ensure you have enough disk space")
        return None

if __name__ == "__main__":
    download_full_amazon_dataset() 