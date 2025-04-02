"""
Command-line interface for processing NetCDF files.
"""

import argparse
import os
import sys

from decadalclimate.processing.reshape import reshape_file


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Reshape netCDF file to have dimensions (initialization, lead_time, lat, lon)"
    )
    parser.add_argument("--input", help="Input netCDF file")
    parser.add_argument("--output", help="Output netCDF file")
    parser.add_argument("--variable", default="tas", help="Variable name to process (default: tas)")
    parser.add_argument("--config", help="Path to custom configuration file")

    return parser.parse_args()


def main():
    """Main entry point for the process command."""
    # Parse command-line arguments
    args = parse_args()

    # Get input file
    input_file = args.input
    if input_file is None:
        print("Error: No input file specified")
        sys.exit(1)

    # Get output file
    output_file = args.output
    if output_file is None:
        # If no output file is specified, create one in the same directory as the input file
        # with "_reshaped.nc" appended to the name
        input_dir = os.path.dirname(input_file)
        input_name = os.path.basename(input_file)
        output_name = os.path.splitext(input_name)[0] + "_reshaped.nc"
        output_file = os.path.join(input_dir, output_name)

    # Get variable name
    variable_name = args.variable

    # Reshape the file
    success = reshape_file(input_file, output_file, variable_name)

    # Exit with appropriate status
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
