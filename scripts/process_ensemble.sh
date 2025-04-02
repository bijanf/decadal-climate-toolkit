#!/bin/bash
# Process a single ensemble member from raw data to final combined file

set -e

# Get the ensemble ID from command line argument
ENSEMBLE=${1:-"r10i11p2f1"}

# Load project paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$( dirname "$SCRIPT_DIR" )"

# Set default paths
DATA_DIR="${DATA_DIR:-/path/to/data}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/output}"
TEMP_DIR="${TEMP_DIR:-$OUTPUT_DIR/temp}"
INDIV_DIR="${INDIV_DIR:-$OUTPUT_DIR/individual}"

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$TEMP_DIR"
mkdir -p "$INDIV_DIR"

echo "================================================================="
echo "Processing ensemble member: $ENSEMBLE"
echo "================================================================="

# Find all files for this ensemble member
files=$(find "$DATA_DIR" -name "*${ENSEMBLE}*anomaly.nc")

# Check if files exist for this ensemble
if [ -z "$files" ]; then
    echo "No files found for ensemble $ENSEMBLE, exiting..."
    exit 1
fi

# Extract initialization years from filenames
echo "Extracting initialization years..."
init_years=()
for file in $files; do
    filename=$(basename "$file")
    init_year=$(echo "$filename" | grep -o "f[0-9]\{4\}" | sed 's/f//')

    # Add to the array if not already present
    if [[ ! " ${init_years[@]} " =~ " ${init_year} " ]]; then
        init_years+=("$init_year")
    fi
done

# Sort initialization years
init_years=($(echo "${init_years[@]}" | tr ' ' '\n' | sort -n))
echo "Found initialization years: ${init_years[@]}"
total_years=${#init_years[@]}
echo "Total years to process: $total_years"

# Process each year individually
echo "Processing each year individually..."
for init_year in "${init_years[@]}"; do
    year_out="${INDIV_DIR}/tas_${ENSEMBLE}_${init_year}.nc"

    # Skip if already processed
    if [ -f "$year_out" ]; then
        echo "Year $init_year already processed, using existing file"
        continue
    fi

    echo "=== Processing year $init_year ==="

    # Find the file for this year
    file=$(find "$DATA_DIR" -name "*f${init_year}_${ENSEMBLE}*anomaly.nc" | head -1)

    if [ -z "$file" ]; then
        echo "  Warning: No file found for year $init_year, skipping"
        continue
    fi

    echo "  Processing file: $(basename "$file")"

    # Create temporary file
    tmp_dir=$(mktemp -d)

    # Get variable name
    var_name=$(cdo -s showname "$file" | tr ' ' '\n' | tail -1)
    echo "  - Variable name: $var_name"

    # Rename variable to tas
    renamed_file="${tmp_dir}/renamed.nc"
    echo "  - Renaming variable to tas"
    cdo -s -O chname,$var_name,tas "$file" "$renamed_file"

    # Set time axis
    echo "  - Setting time axis"
    cdo -s -O settaxis,${init_year}-11-01,00:00:00,1month "$renamed_file" "$year_out"

    # Check time dimension
    time_steps=$(cdo -s ntime "$year_out")
    echo "  - Time steps: $time_steps"

    # Clean up
    rm -rf "$tmp_dir"

    echo "  - Completed processing year $init_year"
done

# Combine all years into one file
echo "================================================================="
echo "Combining all years into final netCDF file..."

# Output file
output_file="${OUTPUT_DIR}/tas_${ENSEMBLE}_combined.nc"

# Run the Python script to combine files
python -m decadalclimate.cli.combine \
    --input_dir "$INDIV_DIR" \
    --output_file "$output_file" \
    --ensemble "$ENSEMBLE" \
    --overwrite

# Check if output file was created
if [ -f "$output_file" ]; then
    size=$(du -h "$output_file" | cut -f1)
    echo "Successfully created file: $output_file (size: $size)"

    # Show file structure
    ncdump -h "$output_file" | head -30
else
    echo "Error: Failed to create output file"
    exit 1
fi

echo "================================================================="
echo "Processing complete for ensemble $ENSEMBLE"
echo "================================================================="
