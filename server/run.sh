#!/bin/bash
# LTX-2 Inference Server startup script

set -e

# Check if .env exists
if [ ! -f .env ]; then
    echo "Error: .env file not found"
    echo "Please copy .env.example to .env and configure your model paths"
    exit 1
fi

# Check if required directories exist
mkdir -p outputs temp

# Run the server
echo "Starting LTX-2 Inference Server..."
echo "API docs will be available at http://localhost:8000/docs"

python main.py "$@"
