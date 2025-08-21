#!/usr/bin/env bash
set -e

cd /root

# Install Python 3.11 and venv
apt-get update
apt-get install -y python3.11 python3.11-venv

# Remove old venv if it exists
rm -rf ~/olmo-code

# Create new virtual environment with Python 3.11
python3.11 -m venv ~/olmo-code

# Activate the venv for this script run
source ~/olmo-code/bin/activate

# Upgrade base tools
pip install --upgrade pip setuptools wheel

python -V
echo "Virtual environment 'olmo-code' created and ready. For interactive shells, run:"
echo "  source ~/olmo-code/bin/activate"
