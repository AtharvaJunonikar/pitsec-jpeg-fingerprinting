#!/bin/bash

# PITSEC JPEG Fingerprinting - Automated Setup Script
# Usage: bash setup.sh
# This script creates the virtual environment and installs all dependencies

set -e  # Exit on error

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  PITSEC JPEG Fingerprinting - Environment Setup"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.8+ and try again."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | awk '{print $2}')
echo "✓ Found Python $PYTHON_VERSION"
echo ""

# Check if venv already exists
if [ -d "venv" ]; then
    echo "⚠️  Virtual environment already exists."
    read -p "Do you want to recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing old venv..."
        rm -rf venv
    else
        echo "Using existing venv. Skipping creation."
        echo "Activating existing environment..."
        source venv/bin/activate
        echo "✓ Venv activated"
        echo ""
        echo "Checking for dependency updates..."
        pip install -r requirements.txt
        echo "✓ Dependencies installed/updated"
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "  Setup complete! Your environment is ready."
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        exit 0
    fi
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
echo "✓ Virtual environment created"
echo ""

# Activate venv
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Venv activated"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel > /dev/null 2>&1
echo "✓ pip upgraded"
echo ""

# Install requirements
echo "Installing dependencies from requirements.txt..."
echo "(This may take a few minutes...)"
pip install -r requirements.txt
echo "✓ All dependencies installed"
echo ""

# Verify installation
echo "Verifying installation..."
python -c "import numpy, cv2, skimage, sklearn, jpeglib; print('✓ All core packages verified')" || {
    echo "❌ Verification failed. Some packages may not have installed correctly."
    exit 1
}
echo ""

# Create necessary folders
echo "Creating project directories..."
mkdir -p data/alaska_tif
mkdir -p data/compressed
mkdir -p output
mkdir -p src
echo "✓ Folders created"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ✓ Setup Complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Your environment is ready. Next steps:"
echo ""
echo "1. Activate the virtual environment in future sessions:"
echo "   source venv/bin/activate"
echo ""
echo "2. Add your ALASKA TIFF files to:"
echo "   data/alaska_tif/"
echo ""
echo "3. Run the exploration script:"
echo "   python src/test_jpeg_version.py"
echo ""
echo "4. Run the production pipeline:"
echo "   python src/bulk_classify.py"
echo ""
echo "For more details, see SETUP.md and README.md"
echo ""
