#!/usr/bin/env python3
"""
Example script to process an ERA5 reanalysis ensemble.

This script demonstrates how to use the DecadalClimate package to process
an ensemble of ERA5 reanalysis data.
"""

import argparse
import logging
import os

from decadalclimate.utils.config import load_config

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("era5_example")


def process_era5(data_dir, output_dir, ensemble_id, overwrite=False):
    """Process an ERA5 ensemble member.

    Parameters
    ----------
    data_dir : str
        Path to the directory containing ERA5 NetCDF files
    output_dir : str
        Path to the directory to save output files
    ensemble_id : str
        Ensemble member ID
    overwrite : bool, optional
        Whether to overwrite existing files, by default False
    """
    logger.info(f"Processing ERA5 ensemble member: {ensemble_id}")

    # Create output directories
    individual_dir = os.path.join(output_dir, "individual")
    os.makedirs(individual_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Process files using shell script for each year
    cmd = f"DATA_DIR={data_dir} OUTPUT_DIR={output_dir} INDIV_DIR={individual_dir} "
    cmd += f"../scripts/process_ensemble.sh {ensemble_id}"

    logger.info(f"Running command: {cmd}")
    exit_code = os.system(cmd)

    if exit_code != 0:
        logger.error(f"Error processing ensemble member: {ensemble_id}")
        return False

    logger.info(f"Successfully processed ensemble member: {ensemble_id}")
    return True


def main():
    """Main entry point for the example script."""
    parser = argparse.ArgumentParser(description="Process an ERA5 ensemble member")
    parser.add_argument("--data-dir", type=str, help="Data directory")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory")
    parser.add_argument("--ensemble", type=str, default="r1i1p1f1", help="Ensemble member ID")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")

    args = parser.parse_args()

    # Load configuration
    config = load_config()

    # Get data directory
    data_dir = args.data_dir
    if data_dir is None:
        data_dir = config.get("input", {}).get("data_dir")
        if data_dir is None:
            logger.error("No data directory specified")
            return 1

    # Process ensemble member
    success = process_era5(
        data_dir=data_dir,
        output_dir=args.output_dir,
        ensemble_id=args.ensemble,
        overwrite=args.overwrite,
    )

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
