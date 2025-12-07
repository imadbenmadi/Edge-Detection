#!/bin/bash

# Bio-XYW-Net Setup Script
# Sets up the complete environment for Bio-inspired XYW-Net

set -e  # Exit on error

echo "=========================================="
echo "Bio-XYW-Net Setup"
echo "=========================================="

# Check Python version
echo "[1/6] Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "  Found Python $python_version"

# Create virtual environment
echo "[2/6] Creating virtual environment..."
if [ -d "bio_env" ]; then
    echo "  Virtual environment already exists"
else
    python3 -m venv bio_env
    echo "  Created bio_env"
fi

# Activate virtual environment
echo "[3/6] Activating virtual environment..."
source bio_env/bin/activate
echo "  Activated bio_env"

# Upgrade pip
echo "[4/6] Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install PyTorch (adjust CUDA version as needed)
echo "[5/6] Installing PyTorch and dependencies..."
# For CPU only:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# For CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
pip install -r requirements.txt

# Create necessary directories
echo "[6/6] Creating directories..."
mkdir -p ./data
mkdir -p ./checkpoints
mkdir -p ./results
mkdir -p ./test_results
mkdir -p ./robustness_results
echo "  Created data/, checkpoints/, results/, test_results/, robustness_results/"

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Activate environment:"
echo "   source bio_env/bin/activate"
echo ""
echo "2. Download datasets (optional):"
echo "   wget http://mftp.mmcheng.net/liuyun/rcf/data/HED-BSDS.tar.gz"
echo "   tar -xzf HED-BSDS.tar.gz -C ./data/"
echo ""
echo "3. Test installation:"
echo "   python bio_frontend.py"
echo "   python bio_model.py"
echo ""
echo "4. Run inference on a test image:"
echo "   python bio_test.py --image test_image.jpg --visualize_frontend"
echo ""
echo "5. Read documentation:"
echo "   cat BIO_XYW_NET_DOCUMENTATION.md"
echo ""
