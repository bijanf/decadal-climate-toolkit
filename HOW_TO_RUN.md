# How to Run DecadalClimate

This guide provides step-by-step instructions for setting up the environment and running the DecadalClimate toolkit with different ensemble members.

## Setup

First, set up the required environment:

```bash
# Install the environment
./scripts/setup_env.sh

# Activate the environment
source ./scripts/activate_env.sh
```

## Configuration

Before running the code, configure the data paths in `config/paths.yaml`:

```yaml
# Update this path to your data directory
input:
  data_dir: "/path/to/your/data"
```

## Running the Code

### Process a Single Ensemble Member

To process a single ensemble member (e.g., r10i11p2f1):

```bash
./scripts/process_ensemble.sh r10i11p2f1
```

Replace `r10i11p2f1` with your desired ensemble member ID.

### Process All Available Ensemble Members

To process all ensemble members found in your data directory:

```bash
./scripts/process_all_ensembles.sh
```

To process only the first ensemble member (for testing):

```bash
./scripts/process_all_ensembles.sh --test
```

### Using Python Module Directly

You can also use the Python modules directly:

```bash
# Combine yearly files for an ensemble
python -m decadalclimate.cli.combine --input_dir output/individual --output_file output/tas_r15i11p2f1_combined.nc --ensemble r15i11p2f1

# Reshape a NetCDF file
python -m decadalclimate.cli.process --input path/to/input.nc --output path/to/output.nc
```

## Output

Processed files are saved in the following directories:

- Individual yearly files: `output/individual/`
- Combined files: `output/`

Each combined file has dimensions `(initialization, lead_time, lat, lon)` and follows the naming pattern `tas_[ENSEMBLE]_combined.nc`.

## Example

```bash
# Set input data path (replace with your actual path)
export DATA_DIR="/home/user/climate_data"

# Process ensemble member r15i11p2f1
./scripts/process_ensemble.sh r15i11p2f1

# Check the output
ncdump -h output/tas_r15i11p2f1_combined.nc | head -20
```

## Troubleshooting

If you encounter issues:

1. Ensure the data directory contains files matching `*[ENSEMBLE]*anomaly.nc`
2. Verify that CDO and NCO utilities are installed
3. Check the log output for specific error messages
4. Make sure the environment is properly activated
