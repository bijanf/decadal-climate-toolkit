#!/bin/bash
# Process all ensemble members from raw data to final combined files

set -e

# Load project paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$( dirname "$SCRIPT_DIR" )"

# Set default paths
DATA_DIR="${DATA_DIR:-/work/bk1318/k202208/diff-pred/data/mpi-esm/hindcast/seasonal-daily-18m/monthly/anomaly}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/output}"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Find all ensemble members
echo "Finding all ensemble members..."
ensemble_members=$(ls -1 "$DATA_DIR" | grep -o "r[0-9]*i[0-9]*p[0-9]*f[0-9]*" | sort -u)

# Display found ensemble members
echo "Found ensemble members:"
echo "$ensemble_members"
echo "Total: $(echo "$ensemble_members" | wc -l) ensembles"

# Ask for confirmation
read -p "Process all ensembles? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Processing canceled."
    exit 0
fi

# Optional: limit to specific ensembles for testing
if [ "$1" = "--test" ]; then
    echo "TEST MODE: Processing only the first ensemble member"
    ensemble_members=$(echo "$ensemble_members" | head -1)
fi

# Process each ensemble member
for ensemble in $ensemble_members; do
    echo "=============================================="
    echo "Processing ensemble: $ensemble"
    echo "=============================================="

    # Call the script to process this ensemble
    "$SCRIPT_DIR/process_ensemble.sh" "$ensemble"

    # Check if processing succeeded
    if [ $? -ne 0 ]; then
        echo "Error processing ensemble $ensemble"
        echo "Continuing with next ensemble..."
    else
        echo "Successfully processed ensemble $ensemble"
    fi

    echo ""
done

echo "All ensembles processed"
echo "Output files are in: $OUTPUT_DIR"

# List the created files
echo "Created files:"
ls -lh "$OUTPUT_DIR"/*.nc | sort
