#!/bin/bash

# Exit on error
set -e

echo "Checking operating system..."
OS_TYPE=$(uname -s)

if [[ "$OS_TYPE" == "Linux" ]]; then
    # Assume Ubuntu/Debian
    echo "Detected Ubuntu Linux."
    sudo apt update
    sudo apt install -y python3-pip python3-venv libgl1
elif [[ "$OS_TYPE" == "Darwin" ]]; then
    # macOS
    echo "Detected macOS."
    if ! command -v brew &> /dev/null; then
        echo "Homebrew not found. Please install it first: https://brew.sh/"
        exit 1
    fi
    brew install python
else
    echo "Unsupported OS: $OS_TYPE"
    exit 1
fi

# Set up Python virtual environment
echo "Setting up Python virtual environment in .venv..."
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
echo "Installing Python dependencies (DeepFace, OpenCV, Postgres driver, Flask, etc.)..."
pip install --upgrade pip
pip install deepface opencv-python psycopg2-binary tf-keras tensorflow-cpu pytest pytest-mock python-dotenv Flask

echo "Python dependency installation complete! Remember to activate the environment using: source .venv/bin/activate"
