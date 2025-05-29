#!/usr/bin/env bash
# exit on error
set -o errexit

# Install system dependencies for document processing
apt-get update
apt-get install -y poppler-utils libmagic1

# Install Python dependencies
pip install -r requirements.txt

# Create necessary directories if they don't exist
mkdir -p uploads
mkdir -p results
mkdir -p static/annotated