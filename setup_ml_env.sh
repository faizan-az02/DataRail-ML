#!/bin/bash

set -e

echo "===== ML Environment Bootstrap Starting ====="

# Update system
echo "[1/6] Updating system packages..."
sudo apt update -y

# Install Python + pip + venv tools
echo "[2/6] Installing Python & dependencies..."
sudo apt install -y python3 python3-pip python3-venv unzip

python3 --version
pip3 --version

VENV_DIR="$HOME/ml-venv"
REQ_MARKER="$HOME/.ml_env_installed"

echo "[3/6] Checking virtual environment..."
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv $VENV_DIR
else
    echo "Virtual environment already exists."
fi

echo "[4/6] Activating virtual environment..."
source $VENV_DIR/bin/activate

pip install --upgrade pip

echo "[5/6] Installing ML libraries..."

if [ ! -f "$REQ_MARKER" ]; then
    pip install numpy pandas scikit-learn matplotlib joblib
    touch $REQ_MARKER
    echo "ML libraries installed."
else
    echo "ML libraries already installed. Skipping."
fi

echo "[6/6] Verifying installation..."
python -c "import sklearn, pandas, numpy; print('Libraries OK')"

echo "===== ML Environment Ready ====="
