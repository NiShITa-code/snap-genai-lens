#!/bin/bash

# Snap GenAI Lens - Setup Script
# This script sets up the environment for running the project

echo "================================"
echo "Snap GenAI Lens - Setup"
echo "================================"
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Check for GPU
echo ""
echo "Checking for GPU..."
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU detected"
    nvidia-smi --query-gpu=name --format=csv,noheader
else
    echo "⚠ No NVIDIA GPU detected. Will use CPU (slower)"
fi

# Install pip dependencies
echo ""
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt --break-system-packages

echo ""
echo "================================"
echo "✓ Setup complete!"
echo "================================"
echo ""
echo "Next steps:"
echo "1. Run the demo: python app.py"
echo "2. Or use Colab: Open snap_lens_demo.ipynb"
echo ""
echo "For Kaggle/Colab:"
echo "- Make sure to enable GPU runtime"
echo "- Models will download automatically on first run"
echo ""
