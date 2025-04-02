#!/bin/bash
# Script to generate forecast horizon plots using the netCDF4-based script

set -e

# Default values
DATA_DIR="output"
OUTPUT_DIR="figures"
MAX_ENSEMBLES=3
COMPARE=true
SPECIFIC_ENSEMBLE=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --max-ensembles)
            MAX_ENSEMBLES="$2"
            shift 2
            ;;
        --ensemble)
            SPECIFIC_ENSEMBLE="$2"
            shift 2
            ;;
        --no-compare)
            COMPARE=false
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --data-dir DIR        Directory containing NetCDF files (default: output)"
            echo "  --output-dir DIR      Directory to save figures (default: figures)"
            echo "  --max-ensembles N     Maximum number of ensembles to process (default: 3, 0 for all)"
            echo "  --ensemble ID         Specific ensemble ID to process"
            echo "  --no-compare          Do not create comparison plots"
            echo "  --help                Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Build command
CMD="python examples/plot_forecast_horizon_nc4.py --data-dir $DATA_DIR --output-dir $OUTPUT_DIR"

if [ -n "$SPECIFIC_ENSEMBLE" ]; then
    CMD="$CMD --ensemble $SPECIFIC_ENSEMBLE"
else
    CMD="$CMD --max-ensembles $MAX_ENSEMBLES"
fi

if [ "$COMPARE" = true ]; then
    CMD="$CMD --compare"
fi

# Print command
echo "Running: $CMD"

# Execute command
eval "$CMD"

echo "Plots saved to $OUTPUT_DIR"
echo "Successfully generated forecast horizon plots!"
