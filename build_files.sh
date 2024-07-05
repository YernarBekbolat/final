#!/bin/bash

echo "BUILD START"

# Create a virtual environment named 'venv' if it doesn't already exist
python3.9 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install all dependencies in the venv
pip install -r requirements.txt

# Collect static files using the Python interpreter from venv
python manage.py collectstatic --noinput

echo "BUILD END"
