#!/usr/bin/env python3
"""
Setup script for group members to quickly get the Multimodal RAG system running.
"""

import os
import sys
import subprocess
import urllib.request
from pathlib import Path

def check_python_version():
    """Check if Python version is 3.8+"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required. You have:", sys.version)
        return False
    print(f"✅ Python version: {sys.version}")
    return True

def check_pip():
    """Check if pip is available"""
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"], 
                      capture_output=True, check=True)
        print("✅ pip is available")
        return True
    except subprocess.CalledProcessError:
        print("❌ pip is not available")
        return False

def install_requirements():
    """Install required packages"""
    print("📦 Installing requirements...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True)
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def check_data_file():
    """Check if data file exists"""
    data_files = ["real_amazon_data.csv", "amazon_data.csv", "products.csv"]
    for file in data_files:
        if os.path.exists(file):
            print(f"✅ Found data file: {file}")
            return True
    
    print("⚠️  No Amazon product CSV file found.")
    print("   You can either:")
    print("   1. Place your CSV file in this directory")
    print("   2. Use the upload feature in the Streamlit interface")
    return False

def check_openai_key():
    """Check if OpenAI API key is set"""
    if os.getenv("OPENAI_API_KEY"):
        print("✅ OpenAI API key found in environment")
        return True
    else:
        print("ℹ️  No OpenAI API key in environment")
        print("   You can enter it in the Streamlit interface")
        return False

def main():
    """Main setup function"""
    print("🚀 Multimodal RAG System Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Check pip
    if not check_pip():
        return False
    
    # Install requirements
    if not install_requirements():
        return False
    
    # Check data file
    check_data_file()
    
    # Check API key
    check_openai_key()
    
    print("\n" + "=" * 50)
    print("🎉 Setup complete!")
    print("\nNext steps:")
    print("1. Get your OpenAI API key from: https://platform.openai.com/api-keys")
    print("2. Run the application:")
    print("   streamlit run streamlit_app_simple.py")
    print("3. Open your browser to: http://localhost:8501")
    print("4. Enter your API key in the sidebar")
    print("5. Initialize the RAG system and start searching!")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1) 