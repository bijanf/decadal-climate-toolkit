#!/bin/bash
# Process a single ensemble member directly using our local Python script
# This bypasses the module system issues

# Check command-line arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <ensemble_id>"
    echo "  ensemble_id: Ensemble member ID (e.g., r10i11p2f1)"
    exit 1
fi

# Set directories with absolute paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$( dirname "$SCRIPT_DIR" )"

HINDCAST_DIR="/work/bk1318/k202208/diff-pred/data/mpi-esm/hindcast/seasonal-daily-18m/monthly/anomaly"
OUTPUT_DIR="./output"  # Relative to current directory
ASSIM_DIR="/work/bk1318/k202208/diff-pred/data/mpi-esm/assim/remapped/anomaly"

# Get the ensemble ID from command line
ENSEMBLE_ID=$1

# Check if the ensemble ID is in the correct format for hindcast files
if [[ $ENSEMBLE_ID == *"i2p1f1"* ]]; then
    echo "Warning: You provided an assimilation member ID ($ENSEMBLE_ID)."
    echo "Converting to hindcast member ID format..."
    # Extract the 'r' number and convert to hindcast format
    R_NUMBER=$(echo $ENSEMBLE_ID | grep -o 'r[0-9]*')
    ENSEMBLE_ID="${R_NUMBER}i11p2f1"
    echo "Using hindcast ensemble ID: $ENSEMBLE_ID"
fi

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

echo "================================================================="
echo "Processing ensemble member: $ENSEMBLE_ID"
echo "================================================================="
echo "Input directory: $HINDCAST_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Assimilation directory: $ASSIM_DIR"

# Check for the existence of hindcast files
HINDCAST_COUNT=$(find $HINDCAST_DIR -name "*${ENSEMBLE_ID}*anomaly.nc" | wc -l)
if [ $HINDCAST_COUNT -eq 0 ]; then
    echo "No hindcast files found for ensemble $ENSEMBLE_ID, skipping..."
    exit 1
fi
echo "Found $HINDCAST_COUNT hindcast files to process"

# Check for matching assimilation file
R_NUMBER=$(echo $ENSEMBLE_ID | grep -o 'r[0-9]*')
ASSIM_FILE=$(find $ASSIM_DIR -name "tas_Amon_asSEIKaSIVERAf_${R_NUMBER}i*-LR_*_anomaly.nc" | head -1)
if [ -z "$ASSIM_FILE" ]; then
    echo "Warning: No matching assimilation file found for ensemble $R_NUMBER"
    echo "Processing will continue but only forecast data will be included"
else
    echo "Found assimilation file: $(basename $ASSIM_FILE)"
    echo "Note: Code will look for variables 'tas' or 'var167' in assimilation file"
fi

# Create the output filename
OUTPUT_FILE="${OUTPUT_DIR}/tas_${ENSEMBLE_ID}_combined.nc"

# Source the environment activation script
ENV_SCRIPT="/work/kd1418/codes/work/k202196/MYWORK/assim/decadal-climate-toolkit/scripts/activate_env.sh"
if [ -f "$ENV_SCRIPT" ]; then
    echo "Activating environment using $ENV_SCRIPT"
    source "$ENV_SCRIPT"
else
    echo "Warning: Environment activation script not found at $ENV_SCRIPT"
    echo "Using system Python instead"
fi

# Make the script executable
chmod +x $SCRIPT_DIR/process_direct.py

# Run the direct Python script
echo "Running: python $SCRIPT_DIR/process_direct.py --input_dir=$HINDCAST_DIR --output_file=$OUTPUT_FILE --ensemble=$ENSEMBLE_ID --assim_dir=$ASSIM_DIR --overwrite"
python $SCRIPT_DIR/process_direct.py --input_dir=$HINDCAST_DIR --output_file=$OUTPUT_FILE --ensemble=$ENSEMBLE_ID --assim_dir=$ASSIM_DIR --overwrite

# Check if the command was successful
if [ $? -ne 0 ]; then
    echo "Error processing ensemble member ${ENSEMBLE_ID}"
    exit 1
fi

echo "Successfully processed ensemble member ${ENSEMBLE_ID}"
echo "Output file: ${OUTPUT_FILE}" 