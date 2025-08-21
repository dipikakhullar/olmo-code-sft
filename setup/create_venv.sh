#!/usr/bin/env bash
set -e  # exit on first error

cd /root

# Update packages and install venv for Python 3.10
apt-get update
apt-get install -y python3.10-venv

# Remove old venv if it exists
rm -rf ~/olmo-code

# Create new virtual environment
python3.10 -m venv ~/olmo-code

# Activate the venv
# (note: activation only persists inside the script; for interactive use you still need
#  to run `source ~/olmo-code/bin/activate` after the script finishes)
source ~/olmo-code/bin/activate

# Upgrade base tools
pip install --upgrade pip setuptools wheel

echo "Virtual environment 'olmo-code' created and ready. Run:"
echo "  source ~/olmo-code/bin/activate"
