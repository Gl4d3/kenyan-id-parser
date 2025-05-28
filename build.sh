#!/usr/bin/env bash
# exit on error
set -o errexit

pip install -r requirements.txt

# Create necessary directories if they don't exist
mkdir -p uploads
mkdir -p results
mkdir -p static/annotated