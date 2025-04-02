#!/bin/bash
# Script to set up a mamba environment with all necessary packages for netCDF processing

# Check if mamba is installed, otherwise use conda
if command -v mamba &> /dev/null; then
    CONDA_CMD="mamba"
else
    CONDA_CMD="conda"
    echo "Mamba not found, using conda instead."
    echo "Consider installing mamba for faster dependency resolution."
fi

# Define environment name
ENV_NAME="netcdf-env"

# Create environment
echo "Creating $ENV_NAME environment..."
$CONDA_CMD create -y -n $ENV_NAME

# Activate environment and install packages
echo "Installing packages..."
$CONDA_CMD install -y -n $ENV_NAME \
    -c conda-forge \
    python=3.9 \
    xarray \
    netcdf4 \
    dask \
    cdo \
    nco \
    ipython \
    matplotlib \
    numpy \
    pandas \
    scipy

# Create activation script
ENV_SCRIPT="/home/bijan/Documents/scripts/decadal/activate_env.sh"
cat > $ENV_SCRIPT << EOL
#!/bin/bash
# Activate the netCDF processing environment

# Determine the conda/mamba command to use
if command -v mamba &> /dev/null; then
    source "\$(dirname \$(which mamba))/activate" "$ENV_NAME"
else
    source "\$(dirname \$(which conda))/activate" "$ENV_NAME"
fi

# Set up environment variables
export PYTHONUNBUFFERED=1

echo "Environment '$ENV_NAME' activated."
echo "Use the following scripts:"
echo "  - process_anomalies_ncl.sh: Main processing script"
echo "  - reshape_netcdf.py: Python script for reshaping the netCDF files"
echo ""
echo "Example usage:"
echo "  cd /home/bijan/Documents/scripts/decadal && ./process_anomalies_ncl.sh"
EOL

chmod +x $ENV_SCRIPT

# Create wrapper script for the main processing script
WRAPPER_SCRIPT="/home/bijan/Documents/scripts/decadal/run_processing.sh"
cat > $WRAPPER_SCRIPT << EOL
#!/bin/bash
# Wrapper script to run the processing with the dedicated environment

# Source the environment
source "$ENV_SCRIPT"

# Run the processing script
cd /home/bijan/Documents/scripts/decadal
./process_anomalies_ncl.sh

# Print completion message
echo "Processing completed. Check the output directory for results."
EOL

chmod +x $WRAPPER_SCRIPT

echo "Environment setup complete!"
echo "To activate the environment, run:"
echo "  source $ENV_SCRIPT"
echo ""
echo "To run the processing with the new environment, use:"
echo "  $WRAPPER_SCRIPT"
