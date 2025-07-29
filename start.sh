#!/bin/bash

echo "Starting AI Ticket Classification API..."
echo "Date: $(date)"
echo "User: $USER"

# Create necessary directories
mkdir -p tokenizer
mkdir -p logs

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Start the FastAPI server
echo "Starting FastAPI server on http://127.0.0.1:8000"
uvicorn main:app --host 127.0.0.1 --port 8000 --reload