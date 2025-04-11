#!/usr/bin/env python3
"""
Process an ensemble member directly by importing local source files.
This script bypasses the module system to handle the processing directly.
"""

import os
import sys
import glob
from datetime import datetime
import argparse

# Ensure the source directory is in the path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Combine NetCDF files for an ensemble member"
    )
    parser.add_argument("--input_dir", help="Directory containing yearly files", required=True)
    parser.add_argument("--output_file", help="Output netCDF file path", required=True)
    parser.add_argument("--ensemble", help="Ensemble member ID (e.g., r10i11p2f1)", required=True)
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite output file if it exists"
    )
    parser.add_argument("--assim_dir", help="Directory containing assimilation member files")
    parser.add_argument("--config", help="Path to custom configuration file")

    return parser.parse_args()

def main():
    """Main entry point."""
    # Parse command-line arguments
    args = parse_args()
    
    # First set up proper paths
    input_dir = args.input_dir
    output_file = args.output_file
    ensemble_id = args.ensemble
    overwrite = args.overwrite
    assim_dir = args.assim_dir
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    print(f"Finding files for ensemble: {ensemble_id}")
    print(f"Input directory: {input_dir}")
    print(f"Output file: {output_file}")
    print(f"Assimilation directory: {assim_dir}")
    
    # Find files matching the ensemble pattern
    file_pattern1 = os.path.join(input_dir, f"tas_{ensemble_id}_*.nc")
    file_pattern2 = os.path.join(input_dir, f"tas*{ensemble_id}*anomaly.nc")
    
    files = sorted(glob.glob(file_pattern1))
    if not files:
        print(f"No files found with pattern: {file_pattern1}")
        print(f"Trying alternative pattern: {file_pattern2}")
        files = sorted(glob.glob(file_pattern2))
    
    if not files:
        print(f"No files found for ensemble {ensemble_id}")
        return False
    
    print(f"Found {len(files)} files to process")
    
    # Get years from filenames
    years = []
    for file in files:
        basename = os.path.basename(file)
        try:
            # Try to extract year from filename patterns like:
            # tas_Amon_seSEIKaSIVERAf1977_r11i11p2f1-LR_197711-197904_anomaly.nc
            year_match = None
            # Try to find the year in the date range part (YYYYMM)
            for part in basename.split('_'):
                if '-' in part and len(part) >= 6:
                    date_part = part.split('-')[0]
                    if date_part.isdigit() and len(date_part) >= 6:
                        year_match = date_part[:4]
                        break
            
            # If that fails, try to find the year in the seSEIKaSIVERAf part
            if not year_match:
                for part in basename.split('_'):
                    if part.startswith('seSEIKaSIVERAf') and part[13:].isdigit():
                        year_match = part[13:]
                        break
            
            if year_match:
                years.append(int(year_match))
            else:
                print(f"Warning: Could not extract year from filename: {basename}")
        except Exception as e:
            print(f"Error extracting year from {basename}: {e}")
    
    if not years:
        print("Error: Could not extract any valid years from filenames")
        return False
    
    years.sort()
    print(f"Years to process: {years}")
    
    # Now process each year
    # For simplicity in this demo, just print what would be done
    print(f"Will create output file: {output_file}")
    print(f"Will include data for years: {years}")
    
    if assim_dir:
        # Check for assimilation files
        r_number = ensemble_id.split('i')[0]
        assim_pattern = os.path.join(assim_dir, f"tas_Amon_asSEIKaSIVERAf_{r_number}i*-LR_*_anomaly.nc")
        assim_files = glob.glob(assim_pattern)
        if assim_files:
            print(f"Found assimilation file: {os.path.basename(assim_files[0])}")
            print("Will include assimilation data in the output")
        else:
            print(f"No assimilation data found for {r_number}")
    
    # Actually do the processing
    # Here we would import the actual code to do the processing
    try:
        # Import after checking other conditions to avoid import errors stopping the script
        import numpy as np
        import netCDF4 as nc
        
        # Import the functions from the local source
        from src.decadalclimate.io.netcdf import (
            find_files, create_netcdf_file, fill_netcdf_file, create_and_fill_file
        )
        
        # Call the function to create and fill the file
        success = create_and_fill_file(
            args.input_dir, args.output_file, args.ensemble, args.overwrite, args.assim_dir
        )
        
        return success
    except ImportError as e:
        print(f"Error importing required modules: {e}")
        print("This is likely due to missing Python modules in your environment.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 