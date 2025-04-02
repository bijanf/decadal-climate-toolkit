#!/bin/bash
# Activate the DecadalClimate environment

# Get the directory containing this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$( dirname "$SCRIPT_DIR" )"

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate decadal-env

# Add project root to PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

echo "DecadalClimate environment activated."
echo "Run 'conda deactivate' to exit the environment."
