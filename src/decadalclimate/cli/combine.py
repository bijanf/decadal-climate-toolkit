"""
Command-line interface for combining NetCDF files.
"""

import argparse
import os
import sys

from decadalclimate.io.netcdf import create_and_fill_file
from decadalclimate.utils.config import load_config


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Create a netCDF file with the right dimensions and fill it incrementally"
    )
    parser.add_argument("--input_dir", help="Directory containing yearly files")
    parser.add_argument("--output_file", help="Output netCDF file path")
    parser.add_argument("--ensemble", help="Ensemble member (e.g., r10i11p2f1)")
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite output file if it exists"
    )
    parser.add_argument("--config", help="Path to custom configuration file")

    return parser.parse_args()


def main():
    """Main entry point for the combine command."""
    # Parse command-line arguments
    args = parse_args()

    # Load configuration
    config = load_config(args.config)

    # Get input directory
    input_dir = args.input_dir
    if input_dir is None:
        input_dir = config.get("input", {}).get("individual_dir")
        if input_dir is None:
            print("Error: No input directory specified")
            sys.exit(1)

    # Get output file
    output_file = args.output_file
    if output_file is None:
        combined_dir = config.get("output", {}).get("combined_dir", "output/combined")
        # Create directory if it doesn't exist
        os.makedirs(combined_dir, exist_ok=True)

        # Use default ensemble if not specified
        ensemble = args.ensemble or config.get("general", {}).get("default_ensemble", "r10i11p2f1")

        # Default output file
        output_file = os.path.join(combined_dir, f"tas_{ensemble}_combined.nc")

    # Get ensemble
    ensemble = args.ensemble
    if ensemble is None:
        ensemble = config.get("general", {}).get("default_ensemble")
        if ensemble is None:
            print("Error: No ensemble member specified")
            sys.exit(1)

    # Get overwrite flag
    overwrite = args.overwrite
    if not overwrite:
        overwrite = config.get("general", {}).get("overwrite", False)

    # Create and fill the file
    success = create_and_fill_file(input_dir, output_file, ensemble, overwrite)

    # Exit with appropriate status
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
