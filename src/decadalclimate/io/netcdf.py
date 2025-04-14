"""
NetCDF I/O operations using the netCDF4 library directly for optimal memory efficiency.
"""

import glob
import os
import time
from typing import List, Tuple
from datetime import datetime, timedelta

import netCDF4 as nc
import numpy as np


def find_files(input_dir: str, ensemble_id: str) -> Tuple[List[str], List[int]]:
    """
    Find all yearly NetCDF files for a given ensemble member and extract years.

    Parameters
    ----------
    input_dir : str
        Directory containing the yearly NetCDF files
    ensemble_id : str
        Ensemble member ID (e.g., r10i11p2f1)

    Returns
    -------
    Tuple[List[str], List[int]]
        A tuple containing the list of files and the corresponding years
    """
    # Look for files with the hindcast pattern
    file_pattern = os.path.join(input_dir, f"tas*{ensemble_id}*anomaly.nc")
    print(f"Looking for files matching: {file_pattern}")

    files = sorted(glob.glob(file_pattern))
    if not files:
        # Try the old pattern as fallback
        file_pattern = os.path.join(input_dir, f"tas_{ensemble_id}_*.nc")
        print(f"No files found. Searching instead for: {file_pattern}")
        files = sorted(glob.glob(file_pattern))
        
        if not files:
            print(f"No files found for ensemble {ensemble_id}")
            return [], []

    print(f"Found {len(files)} files to process")

    # Extract years from filenames
    years = []
    for file in files:
        basename = os.path.basename(file)
        try:
            # Try to extract year from filename patterns like:
            # tas_Amon_seSEIKaSIVERAf1977_r11i11p2f1-LR_197711-197904_anomaly.nc
            # The year is in the component that starts with the year and month (YYYYMM)
            date_parts = basename.split('_')
            for part in date_parts:
                if len(part) >= 6 and '-' in part:
                    # This looks like a date range (e.g., 197711-197904)
                    start_date = part.split('-')[0]
                    if start_date.isdigit() and len(start_date) >= 6:
                        year = int(start_date[:4])
                        years.append(year)
                        break
            
            # If we couldn't extract the year from date parts, try the old method
            if not years or len(years) <= len(files) - 1:
                # Extract year from seSEIKaSIVERAfYYYY part
                for part in date_parts:
                    if part.startswith('seSEIKaSIVERAf'):
                        year_str = part[len('seSEIKaSIVERAf'):]
                        if year_str.isdigit():
                            year = int(year_str)
                            years.append(year)
                            break
        except ValueError:
            print(f"Warning: Could not extract year from filename: {basename}")

    # Ensure we have a year for each file
    if len(years) != len(files):
        print(f"Warning: Could not extract years for all files. Found {len(years)} years for {len(files)} files.")
        # Try to extract years from the filenames using a regex
        years = []
        import re
        for file in files:
            basename = os.path.basename(file)
            # Look for patterns like YYYY in the filename
            year_match = re.search(r'_(\d{4})(\d{2})-', basename)
            if year_match:
                year = int(year_match.group(1))
                years.append(year)
            else:
                print(f"Warning: Could not extract year from filename: {basename}")

    # Sort years and files together
    if len(years) == len(files):
        year_file_pairs = sorted(zip(years, files))
        years = [y for y, _ in year_file_pairs]
        files = [f for _, f in year_file_pairs]
    else:
        print("Warning: Mismatched number of years and files. Using files in their current order.")

    return files, years


def create_netcdf_file(
    output_file: str,
    years: List[int],
    dimensions: dict,
    lat_values: np.ndarray,
    lon_values: np.ndarray,
    compression_level: int = 4,
) -> bool:
    """
    Create a new NetCDF file with the specified dimensions and coordinates.

    Parameters
    ----------
    output_file : str
        Path to the output NetCDF file
    years : List[int]
        List of initialization years
    dimensions : dict
        Dictionary containing dimension sizes
    lat_values : np.ndarray
        Latitude values
    lon_values : np.ndarray
        Longitude values
    compression_level : int, optional
        Compression level for variables, by default 4

    Returns
    -------
    bool
        True if successful, False otherwise
    """
    try:
        with nc.Dataset(output_file, "w", format="NETCDF4") as dst:
            # Create dimensions
            dst.createDimension("time", len(years))
            dst.createDimension("lead_time", dimensions["lead_time"])
            dst.createDimension("lat", dimensions["lat"])
            dst.createDimension("lon", dimensions["lon"])

            # Create variables
            time_var = dst.createVariable("time", "f8", ("time",))
            lead_var = dst.createVariable("lead_time", "i4", ("lead_time",))
            lat_var = dst.createVariable("lat", "f4", ("lat",))
            lon_var = dst.createVariable("lon", "f4", ("lon",))
            tas_var = dst.createVariable("tas", "f4", ("time", "lead_time", "lat", "lon"), zlib=True)

            # Set variable attributes
            time_var.units = "days since 1958-01-01"
            time_var.calendar = "standard"
            time_var.long_name = "Time"
            lead_var.units = "months"
            lead_var.long_name = "Lead time"
            lat_var.units = "degrees_north"
            lat_var.long_name = "Latitude"
            lon_var.units = "degrees_east"
            lon_var.long_name = "Longitude"
            tas_var.units = "K"
            tas_var.long_name = "Near-Surface Air Temperature"

            # Fill variables
            lead_var[:] = np.arange(dimensions["lead_time"])
            lat_var[:] = lat_values
            lon_var[:] = lon_values

            # Set time values to 24 months before the initialization date
            for i, year in enumerate(years):
                # Calculate date 24 months before the initialization date
                init_month = 11  # November
                init_year = year
                
                # Go back 24 months (2 years) and use October (as specified in the requirements)
                start_year = init_year - 2
                start_month = 10  # Use October instead of November
                
                # Get the last day of the month
                if start_month == 12:
                    start_date = datetime(start_year + 1, 1, 1) - timedelta(days=1)
                else:
                    start_date = datetime(start_year, start_month + 1, 1) - timedelta(days=1)
                
                time_var[i] = nc.date2num(start_date, time_var.units)

            return True

    except Exception as e:
        print(f"Error creating NetCDF file: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return False


def fill_netcdf_file(
    output_file: str, files: List[str], years: List[int], max_lead_time: int = 18,
    assim_dir: str = None, ensemble_id: str = None
) -> bool:
    """
    Fill an existing NetCDF file with data from yearly files.

    Parameters
    ----------
    output_file : str
        Path to the output NetCDF file
    files : List[str]
        List of input NetCDF files
    years : List[int]
        List of initialization years corresponding to the files
    max_lead_time : int, optional
        Maximum number of lead time steps to include, by default 18
    assim_dir : str, optional
        Directory containing assimilation member files, by default None
    ensemble_id : str, optional
        Ensemble member ID to match in assimilation files, by default None

    Returns
    -------
    bool
        True if successful, False otherwise
    """
    try:
        with nc.Dataset(output_file, "a") as dst:
            tas_var = dst.variables["tas"]

            # Process each year's file one by one
            for i, (year, file) in enumerate(zip(years, files)):
                print(
                    f"Processing year {year} ({i+1}/{len(years)}) "
                    f"from file: {os.path.basename(file)}"
                )
                start_time = time.time()

                try:
                    # If assimilation directory is provided, get the previous 24 months
                    if assim_dir and ensemble_id:
                        # Extract r number from ensemble_id (e.g., r10 from r10i11p2f1)
                        r_number = ensemble_id.split('i')[0]
                        
                        # Find the assimilation member file with matching r number
                        assim_pattern = os.path.join(assim_dir, f"tas_Amon_asSEIKaSIVERAf_{r_number}i*-LR_*_anomaly.nc")
                        assim_files = glob.glob(assim_pattern)
                        if not assim_files:
                            print(f"Warning: No assimilation member files found for r number {r_number} in {assim_dir}")
                        else:
                            # Use the first matching assimilation member file
                            assim_file = assim_files[0]
                            print(f"Using assimilation member: {os.path.basename(assim_file)}")
                            
                            with nc.Dataset(assim_file, "r") as assim_ds:
                                # Get the time variable
                                time_var = assim_ds.variables["time"]
                                time_units = time_var.units
                                
                                # Calculate the target month (November of initialization year)
                                target_month = 11  # November
                                target_year = year
                                
                                # Find the index for the target month
                                time_values = time_var[:]
                                # Use end of month for assimilation data
                                target_date = datetime(target_year, target_month, 1)
                                # Add one month and subtract one day to get end of month
                                if target_month == 12:
                                    target_date = datetime(target_year + 1, 1, 1) - timedelta(days=1)
                                else:
                                    target_date = datetime(target_year, target_month + 1, 1) - timedelta(days=1)
                                target_time = nc.date2num(target_date, time_units)
                                
                                # Find the closest time index
                                time_diff = np.abs(time_values - target_time)
                                target_idx = np.argmin(time_diff)
                                
                                # Get the previous 24 months
                                start_idx = target_idx - 24
                                if start_idx >= 0:
                                    # Determine which variable to use (tas or var167)
                                    if "tas" in assim_ds.variables:
                                        assim_var_name = "tas"
                                        print(f"  Using variable 'tas' from assimilation file")
                                    elif "var167" in assim_ds.variables:
                                        assim_var_name = "var167"
                                        print(f"  Using variable 'var167' from assimilation file (alias for tas)")
                                    else:
                                        print(f"  Warning: Neither 'tas' nor 'var167' found in assimilation file")
                                        print(f"  Available variables: {list(assim_ds.variables.keys())}")
                                        print(f"  Skipping assimilation data for this year")
                                        continue
                                        
                                    # Copy the previous 24 months
                                    for j in range(24):
                                        tas_var[i, j, :, :] = assim_ds.variables[assim_var_name][start_idx + j, :, :]
                                    print("  Successfully copied previous 24 months from assimilation member")
                                else:
                                    print(f"  Warning: Could not find 24 months before {target_date} in assimilation member")

                    with nc.Dataset(file, "r") as src:
                        # Check time dimension
                        if "time" not in src.dimensions:
                            print(
                                f"Warning: 'time' dimension not found in file"
                                f" for year {year}, skipping"
                            )
                            continue

                        # Determine which variable to use (tas or var167 or others)
                        hindcast_var_name = None
                        for var_name in ["tas", "var167", "tas_anomalies"]:
                            if var_name in src.variables:
                                hindcast_var_name = var_name
                                print(f"  Using variable '{var_name}' from hindcast file")
                                break
                        
                        if not hindcast_var_name:
                            print(f"  Warning: No recognized temperature variable found in hindcast file")
                            print(f"  Available variables: {list(src.variables.keys())}")
                            print(f"  Skipping hindcast data for this year")
                            continue

                        n_times = min(max_lead_time, len(src.dimensions["time"]))
                        if n_times < max_lead_time:
                            print(
                                f"Warning: File has {n_times} time steps, expected {max_lead_time}"
                            )
                            print("This year will have missing data")

                        # Copy data for this year, starting after the assimilation data
                        start_idx = 24 if assim_dir else 0
                        for t in range(n_times):
                            print(f"    Lead time {t+1}/{n_times}", end="\r")
                            # Get the time variable and its units
                            time_var = src.variables["time"]
                            time_units = time_var.units
                            time_value = time_var[t]
                            
                            # Set a new consistent reference date for all data
                            new_units = "days since 1958-01-01"
                            
                            # Convert the time value to a datetime object
                            try:
                                if "day as %Y%m%d.%f" in time_units:
                                    # Handle hindcast time format (YYYYMMDD.fraction)
                                    time_str = f"{int(time_value):08d}"
                                    year = int(time_str[:4])
                                    month = int(time_str[4:6])
                                    day = int(time_str[6:8])
                                    date = datetime(year, month, day)
                                else:
                                    # Handle standard netCDF time format
                                    date = nc.num2date(time_value, time_units)
                            except (ValueError, TypeError) as e:
                                print(f"Warning: Could not parse time value {time_value} with units {time_units}")
                                print(f"Using fallback date for year {year}")
                                # Use a fallback date based on the year and lead time
                                date = datetime(year, 11, 1)  # Start with November
                                # Add lead time months
                                date = date + timedelta(days=30*t)  # Approximate month length
                            
                            # For hindcast data (starting from 2023-11-16)
                            if "seSEIKaSIVERAf" in file:
                                # Calculate the number of days since 1958-01-01
                                ref_date = datetime(1958, 1, 1)
                                days_since_ref = (date - ref_date).days
                                new_time_value = days_since_ref
                            # For assimilation data (starting from 1958-01-31)
                            else:
                                # Convert to the new reference date
                                new_time_value = nc.date2num(date, new_units)
                            
                            # Ensure we're at the end of the month
                            if date.day != 1:
                                # Calculate the next month's first day
                                if date.month == 12:
                                    next_month = datetime(date.year + 1, 1, 1)
                                else:
                                    next_month = datetime(date.year, date.month + 1, 1)
                                # Subtract one day to get end of current month
                                date = next_month - timedelta(days=1)
                                # Recalculate days since reference
                                if "seSEIKaSIVERAf" in file:
                                    days_since_ref = (date - ref_date).days
                                    new_time_value = days_since_ref
                                else:
                                    new_time_value = nc.date2num(date, new_units)
                            
                            tas_var[i, start_idx + t, :, :] = src.variables[hindcast_var_name][t, :, :]

                        print(
                            f"\n  Year {year} processed in {time.time() - start_time:.2f} seconds"
                        )

                except Exception as e:
                    print(f"Error processing file for year {year}: {e}")
                    # Print more detailed error information
                    import traceback
                    print(f"  Traceback: {traceback.format_exc()}")

            return True

    except Exception as e:
        print(f"Error filling NetCDF file: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return False


def create_and_fill_file(
    input_dir: str, output_file: str, ensemble_id: str, overwrite: bool = False,
    assim_dir: str = None
) -> bool:
    """
    Create a NetCDF file with the right dimensions and fill it incrementally.

    Parameters
    ----------
    input_dir : str
        Directory containing individual yearly NetCDF files
    output_file : str
        Path to the output NetCDF file
    ensemble_id : str
        Ensemble member ID (e.g., r10i11p2f1)
    overwrite : bool, optional
        Whether to overwrite the output file if it exists, by default False
    assim_dir : str, optional
        Directory containing assimilation member files, by default None

    Returns
    -------
    bool
        True if successful, False otherwise
    """
    # Check if output file already exists
    if os.path.exists(output_file) and not overwrite:
        print(f"Output file {output_file} already exists. Use --overwrite to replace it.")
        return False
    elif os.path.exists(output_file) and overwrite:
        os.remove(output_file)

    # Find files and extract years
    files, years = find_files(input_dir, ensemble_id)

    if not years:
        print("Error: Could not extract any valid years from filenames")
        return False

    print(f"Years to process: {years}")

    # Open the first file to get dimensions
    print(f"Opening first file to get dimensions: {files[0]}")

    try:
        with nc.Dataset(files[0], "r") as src:
            # Check dimensions
            if "time" not in src.dimensions:
                print("Error: 'time' dimension not found in input file")
                return False

            # Increase lead time steps to accommodate assimilation data
            lead_time_steps = min(18, len(src.dimensions["time"]))
            if assim_dir:
                lead_time_steps += 24  # Add 24 months for assimilation data
            nlat = len(src.dimensions["lat"])
            nlon = len(src.dimensions["lon"])

            # Get lat/lon values
            lat_values = src.variables["lat"][:]
            lon_values = src.variables["lon"][:]

            print(
                f"Creating output file with dimensions: "
                f"(time={len(years)}, lead_time={lead_time_steps},"
                f" lat={nlat}, lon={nlon})"
            )

        # Create the output file
        dimensions = {
            "lead_time": lead_time_steps,
            "lat": nlat,
            "lon": nlon,
        }

        success = create_netcdf_file(output_file, years, dimensions, lat_values, lon_values)
        if not success:
            return False

        # Fill the file with data
        success = fill_netcdf_file(output_file, files, years, max_lead_time=18, 
                                 assim_dir=assim_dir, ensemble_id=ensemble_id)
        if not success:
            if os.path.exists(output_file):
                os.remove(output_file)
            return False

        # Show file information
        size = os.path.getsize(output_file) / (1024 * 1024)  # Size in MB
        print(f"Output file size: {size:.2f} MB")

        # Verify the file
        with nc.Dataset(output_file, "r") as f:
            print(f"Output file dimensions: {f.dimensions.keys()}")
            print(f"Output file variables: {f.variables.keys()}")
            print(f"Time values: {f.variables['time'][:]}")

        print(f"Successfully created file: {output_file}")
        return True

    except Exception as e:
        print(f"Error: {e}")
        if os.path.exists(output_file):
            os.remove(output_file)
        return False
