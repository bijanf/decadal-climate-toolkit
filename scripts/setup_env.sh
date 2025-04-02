#!/bin/bash
# Set up the Python environment for the DecadalClimate package

set -e

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "conda could not be found. Please install Miniconda or Anaconda."
    exit 1
fi

# Get the directory containing this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$( dirname "$SCRIPT_DIR" )"

# Environment name
ENV_NAME="decadal-env"

# Create the conda environment if it doesn't exist
if ! conda env list | grep -q "^$ENV_NAME "; then
    echo "Creating conda environment '$ENV_NAME'..."
    conda create -y -n $ENV_NAME python=3.9
else
    echo "Conda environment '$ENV_NAME' already exists."
fi

# Activate the environment
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

# Install dependencies
echo "Installing required packages..."
conda install -y -c conda-forge \
    numpy \
    xarray \
    netcdf4 \
    pandas \
    dask \
    pyyaml \
    matplotlib \
    scipy \
    cdo \
    nco

# Install development packages if requested
if [ "$1" = "--dev" ]; then
    echo "Installing development packages..."
    conda install -y -c conda-forge \
        pytest \
        pytest-cov \
        black \
        flake8 \
        isort \
        mypy
fi

# Install the package in development mode
echo "Installing DecadalClimate package in development mode..."
pip install -e $PROJECT_ROOT

# Create activation script
ACTIVATE_SCRIPT="$SCRIPT_DIR/activate_env.sh"
cat > "$ACTIVATE_SCRIPT" << EOF
#!/bin/bash
# Activate the DecadalClimate environment

# Get the directory containing this script
SCRIPT_DIR="\$( cd "\$( dirname "\${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="\$( dirname "\$SCRIPT_DIR" )"

# Activate conda environment
eval "\$(conda shell.bash hook)"
conda activate $ENV_NAME

# Add project root to PYTHONPATH
export PYTHONPATH="\$PROJECT_ROOT:\$PYTHONPATH"

echo "DecadalClimate environment activated."
echo "Run 'conda deactivate' to exit the environment."
EOF

chmod +x "$ACTIVATE_SCRIPT"

echo "===================================================="
echo "Environment setup complete!"
echo "To activate the environment, run:"
echo "  source $ACTIVATE_SCRIPT"
echo "===================================================="
