#!/usr/bin/env python3
"""
Bio-XYW-Net Quick Start Guide
==============================

This script provides interactive setup and testing.

Usage:
    python quickstart.py
"""

import os
import sys
import argparse
from pathlib import Path
import torch
import torch.nn as nn

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60 + "\n")

def check_environment():
    """Check system environment"""
    print_header("Environment Check")
    
    print(f"✓ Python version: {sys.version.split()[0]}")
    
    print(f"✓ PyTorch version: {torch.__version__}")
    
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠ CUDA not available (will use CPU)")
    
    try:
        import cv2
        print(f"✓ OpenCV version: {cv2.__version__}")
    except:
        print("✗ OpenCV not installed")
    
    try:
        import numpy
        print(f"✓ NumPy version: {numpy.__version__}")
    except:
        print("✗ NumPy not installed")

def test_bio_frontend():
    """Test bio-frontend components"""
    print_header("Testing Bio-Frontend")
    
    try:
        from bio_frontend import BioFrontend, PhotoreceptorLayer, HorizontalCellLayer, BipolarCellLayer
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Test input
        x = torch.rand(1, 3, 256, 256).to(device)
        
        # Test individual components
        print("Testing PhotoreceptorLayer...")
        photor = PhotoreceptorLayer(add_noise=False)
        photor_out = photor(x)
        print(f"  ✓ Input: {x.shape} -> Output: {photor_out.shape}")
        
        print("Testing HorizontalCellLayer...")
        horiz = HorizontalCellLayer(sigma=2.0, kernel_size=5).to(device)
        horiz_out = horiz(photor_out)
        print(f"  ✓ Input: {photor_out.shape} -> Output: {horiz_out.shape}")
        
        print("Testing BipolarCellLayer...")
        bipolar = BipolarCellLayer()
        on, off = bipolar(photor_out, horiz_out)
        print(f"  ✓ ON: {on.shape}, OFF: {off.shape}")
        
        print("Testing BioFrontend (complete)...")
        bio = BioFrontend(return_intermediate=True).to(device)
        bio_out, intermediates = bio(x)
        print(f"  ✓ Input: {x.shape} -> Output: {bio_out.shape}")
        print(f"    - Photoreceptor: {intermediates['photoreceptor'].shape}")
        print(f"    - Surround: {intermediates['surround'].shape}")
        print(f"    - ON: {intermediates['on'].shape}")
        print(f"    - OFF: {intermediates['off'].shape}")
        
        params = sum(p.numel() for p in bio.parameters())
        print(f"  ✓ BioFrontend parameters: {params:,}")
        
    except Exception as e:
        print(f"✗ Error testing bio-frontend: {e}")
        return False
    
    return True

def test_bio_model():
    """Test Bio-XYW-Net model"""
    print_header("Testing Bio-XYW-Net")
    
    try:
        from bio_model import BioXYWNet, BaselineXYWNet
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Test Bio-XYW-Net
        print("Creating Bio-XYW-Net...")
        bio_model = BioXYWNet(use_learnable_bio=False).to(device)
        bio_params = sum(p.numel() for p in bio_model.parameters())
        print(f"  ✓ Model created with {bio_params:,} parameters")
        
        # Test forward pass
        print("Testing forward pass...")
        x = torch.rand(1, 3, 256, 256).to(device)
        with torch.no_grad():
            y = bio_model(x)
        print(f"  ✓ Input: {x.shape} -> Output: {y.shape}")
        
        # Test Baseline
        print("Creating Baseline XYW-Net...")
        baseline_model = BaselineXYWNet().to(device)
        baseline_params = sum(p.numel() for p in baseline_model.parameters())
        print(f"  ✓ Model created with {baseline_params:,} parameters")
        
        print(f"\nParameter comparison:")
        print(f"  Bio-XYW-Net:     {bio_params:,}")
        print(f"  Baseline:        {baseline_params:,}")
        print(f"  Difference:      {bio_params - baseline_params:+,}")
        print(f"  % Change:        {(bio_params/baseline_params - 1)*100:+.1f}%")
        
    except Exception as e:
        print(f"✗ Error testing bio-model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def create_directories():
    """Create necessary directories"""
    print_header("Creating Directories")
    
    dirs = [
        './data',
        './data/BSDS500',
        './data/BIPED',
        './data/NYUD',
        './checkpoints',
        './results',
        './test_results',
        './robustness_results'
    ]
    
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
        print(f"✓ {d}")

def download_test_image():
    """Download a test image"""
    print_header("Downloading Test Image")
    
    try:
        import urllib.request
        
        url = "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9e/Mountain_landscape.jpg/1280px-Mountain_landscape.jpg"
        output_path = "./test_image.jpg"
        
        if Path(output_path).exists():
            print(f"✓ Test image already exists: {output_path}")
        else:
            print(f"Downloading test image...")
            urllib.request.urlretrieve(url, output_path)
            print(f"✓ Downloaded to: {output_path}")
    
    except Exception as e:
        print(f"⚠ Could not download test image: {e}")
        print("  You can provide your own image or use any RGB image in the current directory")

def main():
    parser = argparse.ArgumentParser(description="Bio-XYW-Net Quick Start")
    parser.add_argument('--full', action='store_true', help='Run all tests')
    parser.add_argument('--env', action='store_true', help='Check environment only')
    parser.add_argument('--test-bio', action='store_true', help='Test bio-frontend')
    parser.add_argument('--test-model', action='store_true', help='Test bio-model')
    parser.add_argument('--setup', action='store_true', help='Setup directories')
    
    args = parser.parse_args()
    
    print("\n")
    print("╔══════════════════════════════════════════════════════════╗")
    print("║          Bio-Inspired XYW-Net Quick Start                ║")
    print("║  Retinal Front-End for Edge Detection                    ║")
    print("╚══════════════════════════════════════════════════════════╝")
    
    # Default: run all
    if not any([args.env, args.test_bio, args.test_model, args.setup]):
        args.full = True
    
    success = True
    
    if args.full or args.env:
        check_environment()
    
    if args.full or args.setup:
        create_directories()
        download_test_image()
    
    if args.full or args.test_bio:
        if not test_bio_frontend():
            success = False
    
    if args.full or args.test_model:
        if not test_bio_model():
            success = False
    
    # Print summary
    print_header("Summary")
    
    if success:
        print("✓ All tests passed!")
        print("\nNext steps:")
        print("  1. Read the documentation:")
        print("     cat BIO_XYW_NET_DOCUMENTATION.md")
        print("")
        print("  2. Test inference on your image:")
        print("     python bio_test.py --image test_image.jpg --visualize_frontend")
        print("")
        print("  3. Train a model:")
        print("     python bio_train.py --dataset BSDS500 --epochs 5")
        print("")
        print("  4. Run robustness tests:")
        print("     python robustness_tests.py --image test_image.jpg")
        print("")
    else:
        print("✗ Some tests failed. Please check your installation.")
    
    print("\nDocumentation: BIO_XYW_NET_DOCUMENTATION.md")
    print("Contact: [Your contact info]\n")

if __name__ == '__main__':
    main()
