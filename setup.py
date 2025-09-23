#!/usr/bin/env python3
"""
Setup script for NASA IMS Bearing Fault Detection Project
This script helps set up the project environment and download necessary datasets.
"""

import os
import sys
import subprocess
from pathlib import Path

def install_requirements():
    """Install required packages from requirements.txt"""
    print("Installing Python requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing requirements: {e}")
        return False

def create_directories():
    """Create necessary project directories"""
    print("Creating project directories...")
    directories = [
        "data",
        "models", 
        "outputs",
        "logs",
        "stm32_code"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✓ Created directory: {directory}")

def check_dataset():
    """Check if NASA dataset is available"""
    print("Checking for NASA IMS dataset...")
    dataset_paths = [
        "data/IMS",
        "data/1st_test",
        "data/2nd_test", 
        "data/3rd_test",
        "1st_test",
        "2nd_test",
        "3rd_test"
    ]
    
    for path in dataset_paths:
        if Path(path).exists() and any(Path(path).iterdir()):
            print(f"✓ Found dataset at: {path}")
            return True
    
    print("⚠ NASA IMS dataset not found.")
    print("Please download from: https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository")
    print("Extract to 'data/' directory or project root.")
    return False

def main():
    """Main setup function"""
    print("=" * 60)
    print("NASA IMS Bearing Fault Detection - Project Setup")
    print("=" * 60)
    
    # Install requirements
    if not install_requirements():
        print("Setup failed. Please install requirements manually.")
        return False
    
    # Create directories
    create_directories()
    
    # Check dataset
    dataset_available = check_dataset()
    
    print("\n" + "=" * 60)
    print("Setup Summary:")
    print("✓ Python requirements installed")
    print("✓ Project directories created")
    print(f"{'✓' if dataset_available else '⚠'} NASA dataset {'found' if dataset_available else 'not found'}")
    
    if not dataset_available:
        print("\nNote: The system will use synthetic data for demonstration if the real dataset is not available.")
    
    print("\nProject is ready! Run 'python Code.py' to start the fault detection system.")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    main()