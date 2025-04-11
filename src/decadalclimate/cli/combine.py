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
    parser.add_argument("--ensemble_id", help="Ensemble member ID (e.g., r10i11p2f1)")
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite output file if it exists"
    )
    parser.add_argument("--config", help="Path to custom configuration file")
    parser.add_argument("--assim_dir", help="Directory containing assimilation member files")

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

    # Get ensemble ID
    ensemble_id = args.ensemble_id
    if ensemble_id is None:
        ensemble_id = config.get("general", {}).get("default_ensemble")
        if ensemble_id is None:
            print("Error: No ensemble member ID specified")
            sys.exit(1)

    # Get output file
    output_file = args.output_file
    if output_file is None:
        combined_dir = config.get("output", {}).get("combined_dir", "output/combined")
        # Create directory if it doesn't exist
        os.makedirs(combined_dir, exist_ok=True)
        # Use the ensemble ID for the output filename
        output_file = os.path.join(combined_dir, f"tas_{ensemble_id}_combined.nc")

    # Get overwrite flag
    overwrite = args.overwrite
    if not overwrite:
        overwrite = config.get("general", {}).get("overwrite", False)

    # Get assimilation directory
    assim_dir = args.assim_dir
    if assim_dir is None:
        assim_dir = config.get("input", {}).get("assim_dir")

    # Create and fill the file
    success = create_and_fill_file(input_dir, output_file, ensemble_id, overwrite, assim_dir)

    # Exit with appropriate status
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
