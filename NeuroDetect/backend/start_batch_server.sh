#!/bin/bash
# Start Batch Processing API Server

echo "Starting NeuroDetect Batch API Server..."
echo

cd "$(dirname "$0")/.."

# Activate conda environment
source ~/miniconda3/bin/activate fraud-detection

# Start the FastAPI server
python server/batch_api.py
