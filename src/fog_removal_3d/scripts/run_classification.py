#!/usr/bin/env python3
"""
Main runner script for fog Gaussian classification experiments
Runs SVM, LightGBM, and Deep Learning methods
"""

import os
import argparse
import subprocess
import sys
import time
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"{'='*60}")
    print(f"Command: {command}")
    print("-" * 60)
    
    start_time = time.time()
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=False, text=True)
        end_time = time.time()
        print(f"\n {description} completed successfully!")
        print(f"  Time taken: {end_time - start_time:.2f} seconds")
        return True
    except subprocess.CalledProcessError as e:
        end_time = time.time()
        print(f"\n {description} failed!")
        print(f"  Time taken: {end_time - start_time:.2f} seconds")
        print(f"Error: {e}")
        return False

def create_directories():
    """Create necessary directories."""
    directories = [
        "models/svm",
        "models/lightgbm", 
        "models/deep_learning",
        "results/svm",
        "results/lightgbm",
        "results/deep_learning",
        "config"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print(" Created necessary directories")

def check_requirements():
    """Check if required packages are available."""
    required_packages = [
        'numpy', 'pandas', 'scikit-learn', 'lightgbm', 
        'torch', 'matplotlib', 'seaborn', 'yaml', 'tqdm', 'joblib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            if package == 'yaml':
                try:
                    __import__('pyyaml')
                except ImportError:
                    missing_packages.append('pyyaml')
            else:
                missing_packages.append(package)
    
    if missing_packages:
        print(f" Missing required packages: {', '.join(missing_packages)}")
        print(f"Please install them using: pip install {' '.join(missing_packages)}")
        return False
    
    print(" All required packages are available")
    return True

def main():
    parser = argparse.ArgumentParser(description='Run fog Gaussian classification experiments')
    parser.add_argument('--methods', nargs='+', 
                       choices=['svm', 'lightgbm', 'deep_learning', 'all'],
                       default=['all'],
                       help='Classification methods to run (default: all)')
    parser.add_argument('--data-path', type=str,
                       default='data/dataset_truck_beta6_alpha250.csv',
                       help='Path to the CSV dataset')
    parser.add_argument('--config-path', type=str,
                       default='config/dl_config.yaml',
                       help='Path to deep learning configuration file')
    parser.add_argument('--skip-checks', action='store_true',
                       help='Skip requirements checking')
    
    args = parser.parse_args()
    
    print(" Fog Gaussian Classification Experiments")
    print("=" * 60)
    
    # Check requirements
    if not args.skip_checks and not check_requirements():
        return 1
    
    # Create directories
    create_directories()
    
    # Check if data file exists
    if not os.path.exists(args.data_path):
        print(f" Data file not found: {args.data_path}")
        print("Please ensure the CSV dataset is available at the specified path.")
        return 1
    
    print(f" Found data file: {args.data_path}")
    
    # Determine which methods to run
    methods_to_run = []
    if 'all' in args.methods:
        methods_to_run = ['svm', 'lightgbm', 'deep_learning']
    else:
        methods_to_run = args.methods
    
    print(f" Running methods: {', '.join(methods_to_run)}")
    
    results = {}
    total_start_time = time.time()
    
    # Run SVM
    if 'svm' in methods_to_run:
        success = run_command(
            f"python svm_classifier.py",
            "Support Vector Machine Classification"
        )
        results['SVM'] = success
    
    # Run LightGBM
    if 'lightgbm' in methods_to_run:
        success = run_command(
            f"python lightgbm_classifier.py",
            "LightGBM Classification"
        )
        results['LightGBM'] = success
    
    # Run Deep Learning
    if 'deep_learning' in methods_to_run:
        # Check if config file exists
        if not os.path.exists(args.config_path):
            print(f"  Config file not found: {args.config_path}")
            print("Using default configuration...")
        
        success = run_command(
            f"python deep_learning_classifier.py",
            "Deep Learning Classification (Multiple Models)"
        )
        results['Deep Learning'] = success
    
    # Print final summary
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"  Total time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
    print(f" Methods run: {len(results)}")
    
    print(f"\n Results:")
    successful = 0
    for method, success in results.items():
        status = " SUCCESS" if success else " FAILED"
        print(f"  {method:15s}: {status}")
        if success:
            successful += 1
    
    print(f"\n Success rate: {successful}/{len(results)} ({100*successful/len(results):.0f}%)")
    
    if successful == len(results):
        print(f"\n All experiments completed successfully!")
        print(f"\n Check the following directories for results:")
        print(f"  - Models: models/")
        print(f"  - Results: results/")
        print(f"  - Visualizations: results/*/")
        return 0
    else:
        print(f"\n  Some experiments failed. Check the error messages above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)