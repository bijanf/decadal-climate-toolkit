#!/bin/bash
# Process all ensemble members using the Python combine script

# Load project paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$( dirname "$SCRIPT_DIR" )"

# Set default paths
HINDCAST_DIR="/work/bk1318/k202208/diff-pred/data/mpi-esm/hindcast/seasonal-daily-18m/monthly/anomaly"
ASSIM_DIR="/work/bk1318/k202208/diff-pred/data/mpi-esm/assim/remapped/anomaly"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/output}"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Source the environment activation script
ENV_SCRIPT="/work/kd1418/codes/work/k202196/MYWORK/assim/decadal-climate-toolkit/scripts/activate_env.sh"
if [ -f "$ENV_SCRIPT" ]; then
    echo "Activating environment using $ENV_SCRIPT"
    source "$ENV_SCRIPT"
else
    echo "Warning: Environment activation script not found at $ENV_SCRIPT"
    echo "Using system Python instead"
fi

# Find all ensemble members in the hindcast directory
echo "Finding all ensemble members in hindcast directory..."
ensemble_members=$(find "$HINDCAST_DIR" -name "*.nc" | grep -o "r[0-9]*i[0-9]*p[0-9]*f[0-9]*" | sort -u)

# Display found ensemble members
echo "Found ensemble members:"
echo "$ensemble_members"
echo "Total: $(echo "$ensemble_members" | wc -l) ensembles"

# Find all assimilation r-numbers
echo "Finding available assimilation members..."
assim_members=$(find "$ASSIM_DIR" -name "*.nc" | grep -o "r[0-9]*i" | sed 's/i$//' | sort -u)
echo "Available assimilation r-numbers: $assim_members"

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
successful=0
failed=0
skipped=0

for ensemble in $ensemble_members; do
    echo "=============================================="
    echo "Processing ensemble: $ensemble"
    echo "=============================================="

    # Extract r-number to check if assimilation data exists
    r_number=$(echo "$ensemble" | grep -o "r[0-9]*")
    echo "Checking for assimilation data with r-number: $r_number"
    
    # Check if this r-number exists in assimilation data
    if echo "$assim_members" | grep -q "$r_number"; then
        echo "Found matching assimilation member for $r_number"
        # Call the script to process this ensemble
        "$SCRIPT_DIR/process_ensemble.sh" "$ensemble"

        # Check if processing succeeded
        if [ $? -ne 0 ]; then
            echo "Error processing ensemble member $ensemble"
            echo "Continuing with next ensemble..."
            ((failed++))
        else
            echo "Successfully processed ensemble $ensemble"
            ((successful++))
        fi
    else
        echo "Warning: No matching assimilation member found for $r_number"
        echo "Skipping ensemble $ensemble"
        ((skipped++))
    fi

    echo ""
done

echo "Processing complete!"
echo "Summary:"
echo "  Successfully processed: $successful ensembles"
echo "  Failed: $failed ensembles"
echo "  Skipped (no assimilation data): $skipped ensembles"
echo "Output files are in: $OUTPUT_DIR"

# List the created files
echo "Created files:"
ls -lh "$OUTPUT_DIR"/*.nc 2>/dev/null | sort || echo "No output files found"
