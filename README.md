# DecadalClimate

[![Build Status](https://github.com/username/decadal/actions/workflows/build.yml/badge.svg)](https://github.com/username/decadal/actions/workflows/build.yml)
[![Lint Status](https://github.com/username/decadal/actions/workflows/lint.yml/badge.svg)](https://github.com/username/decadal/actions/workflows/lint.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive toolkit for processing and analyzing decadal climate prediction data with a focus on NetCDF file manipulation and visualization.

## Overview

DecadalClimate provides a set of tools designed to efficiently process climate model output from decadal prediction experiments. The toolkit handles various aspects of data processing:

- Reshaping NetCDF files to standardized dimensions (initialization, lead_time, lat, lon)
- Combining yearly files into multi-year ensembles
- Memory-efficient processing of large climate datasets
- Support for both xarray and direct netCDF4 interfaces
- Visualization of decadal forecast data with various plot types

## Installation

### Prerequisites

- Python 3.8+
- CDO (Climate Data Operators) 2.0.0+
- NCO (NetCDF Operators) 5.0.0+
- NCL (NCAR Command Language, optional for some scripts)
- Matplotlib, NumPy, and xarray for visualization

### Setup

1. Clone the repository:

```bash
git clone https://github.com/username/decadal.git
cd decadal
```

2. Set up the Python environment:

```bash
./scripts/setup_env.sh
source ./scripts/activate_env.sh
```

## Directory Structure

```
decadal/
├── scripts/       # Primary processing scripts
├── src/           # Core Python modules
├── tests/         # Unit and integration tests
├── config/        # Configuration files
├── docs/          # Documentation
└── examples/      # Example usage and visualization scripts
```

## Usage

### Process a Single Ensemble Member

```bash
export DATA_DIR="/work/bk1318/k202208/diff-pred/data/mpi-esm/hindcast/seasonal-daily-18m/monthly/anomaly"
export OUTPUT_DIR="."
./scripts/process_ensemble.sh r10i11p2f1
```

### Process All Available Ensemble Members

```bash
./scripts/process_all_ensembles.sh
```

## Visualization

The toolkit provides multiple ways to visualize ensemble data, including both xarray-based and netCDF4-based approaches.

### Using the NetCDF4-based Visualization (Recommended)

We provide a robust netCDF4-based visualization script that avoids compatibility issues with xarray:

```bash
# Use the convenient shell script wrapper
./run_forecast_plots.sh --max-ensembles 3

# Process a specific ensemble
./run_forecast_plots.sh --ensemble r10i11p2f1

# Process all ensembles without comparison plots
./run_forecast_plots.sh --max-ensembles 0 --no-compare
```

The shell script supports the following options:
- `--data-dir DIR`: Directory containing NetCDF files (default: output)
- `--output-dir DIR`: Directory to save figures (default: figures)
- `--max-ensembles N`: Maximum number of ensembles to process (default: 3, use 0 for all)
- `--ensemble ID`: Specific ensemble ID to process
- `--no-compare`: Do not create comparison plots

### Alternative Visualization Methods

```bash
# xarray-based visualization (may have compatibility issues on some systems)
python examples/plot_forecast_horizon.py

# Direct netCDF4-based visualization
python examples/plot_forecast_horizon_nc4.py

# Spatial pattern visualization
python examples/plot_global_timeseries.py
```

## Visualization Examples

The toolkit provides multiple visualization options for analyzing the decadal prediction data:

1. **Time Series Visualization**: Plot temperature anomalies by initialization year and lead time
   - Supports multi-ensemble comparison
   - Can show the actual forecast timeline with proper year ticks

2. **Lead Time Comparison**: Compare multiple ensemble members at the same initialization year
   - Shows forecast skill across different ensembles
   - Highlights ensemble spread and forecast uncertainty

3. **Global Spatial Patterns**: Visualize spatial anomaly patterns at different lead times
   - Map projections for global data
   - Comparison across different ensemble members

## Configuration

Configuration settings can be modified in the `config/` directory:

- `config/paths.yaml`: Configure input/output paths
- `config/processing.yaml`: Set processing parameters

## Troubleshooting

If you encounter issues with the xarray-based visualization scripts, try using the netCDF4-based alternatives:
- Use `./run_forecast_plots.sh` instead of directly calling `plot_forecast_horizon.py`
- The netCDF4-based scripts avoid certain compatibility issues with xarray and NetworkManager

## Development

For development, install the additional requirements:

```bash
pip install -r requirements-dev.txt
```

Run tests:

```bash
pytest
```

## Code Quality

This project uses several tools to ensure code quality

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
