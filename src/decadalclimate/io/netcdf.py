"""
NetCDF I/O operations using the netCDF4 library directly for optimal memory efficiency.
"""

import glob
import os
import time
from typing import List, Tuple

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
    file_pattern = os.path.join(input_dir, f"tas_{ensemble_id}_*.nc")
    print(f"Looking for files matching: {file_pattern}")

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
            year = int(basename.split("_")[-1].split(".")[0])
            years.append(year)
        except ValueError:
            print(f"Warning: Could not extract year from filename: {basename}")

    # Sort years and files together
    year_file_pairs = sorted(zip(years, files))
    years = [y for y, _ in year_file_pairs]
    files = [f for _, f in year_file_pairs]

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
        with nc.Dataset(output_file, "w") as dst:
            # Create dimensions
            dst.createDimension("initialization", len(years))
            dst.createDimension("lead_time", dimensions["lead_time"])
            dst.createDimension("lat", dimensions["lat"])
            dst.createDimension("lon", dimensions["lon"])

            # Create variables
            init_var = dst.createVariable("initialization", np.int32, ("initialization",))
            lead_var = dst.createVariable("lead_time", np.int32, ("lead_time",))
            lat_var = dst.createVariable("lat", np.float64, ("lat",))
            lon_var = dst.createVariable("lon", np.float64, ("lon",))

            # The main variable: uses float32 to save space
            tas_var = dst.createVariable(
                "tas",
                np.float32,
                ("initialization", "lead_time", "lat", "lon"),
                zlib=True,
                complevel=compression_level,
            )

            # Set coordinate values
            init_var[:] = years
            lead_var[:] = np.arange(dimensions["lead_time"])
            lat_var[:] = lat_values
            lon_var[:] = lon_values

            # Set attributes
            init_var.units = "year"
            init_var.long_name = "Initialization Year"
            lead_var.units = "months"
            lead_var.long_name = "Forecast Lead Time"

            # Set standard attributes for main variable
            tas_var.units = "K"
            tas_var.long_name = "Temperature Anomaly"
            tas_var.standard_name = "air_temperature"

            # Add global attributes
            dst.title = "Combined Temperature Anomaly Data"
            dst.created = time.ctime()
            dst.description = "Combined file with dimensions (initialization, lead_time, lat, lon)"

        return True
    except Exception as e:
        print(f"Error creating NetCDF file: {e}")
        return False


def fill_netcdf_file(
    output_file: str, files: List[str], years: List[int], max_lead_time: int = 18
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
                    with nc.Dataset(file, "r") as src:
                        # Check time dimension
                        if "time" not in src.dimensions:
                            print(
                                f"Warning: 'time' dimension not found in file"
                                f" for year {year}, skipping"
                            )
                            continue

                        n_times = min(max_lead_time, len(src.dimensions["time"]))
                        if n_times < max_lead_time:
                            print(
                                f"Warning: File has {n_times} time steps, expected {max_lead_time}"
                            )
                            print("This year will have missing data")

                        # Copy data for this year
                        for t in range(n_times):
                            print(f"    Lead time {t+1}/{n_times}", end="\r")
                            # Copy one time step at a time to minimize memory usage
                            tas_var[i, t, :, :] = src.variables["tas"][t, :, :]

                        print(
                            f"\n  Year {year} processed in {time.time() - start_time:.2f} seconds"
                        )

                except Exception as e:
                    print(f"Error processing file for year {year}: {e}")

            return True

    except Exception as e:
        print(f"Error filling NetCDF file: {e}")
        return False


def create_and_fill_file(
    input_dir: str, output_file: str, ensemble_id: str, overwrite: bool = False
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

            lead_time_steps = min(18, len(src.dimensions["time"]))
            nlat = len(src.dimensions["lat"])
            nlon = len(src.dimensions["lon"])

            # Get lat/lon values
            lat_values = src.variables["lat"][:]
            lon_values = src.variables["lon"][:]

            print(
                f"Creating output file with dimensions: "
                f"(initialization={len(years)}, lead_time={lead_time_steps},"
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
        success = fill_netcdf_file(output_file, files, years, max_lead_time=lead_time_steps)
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
            print(f"Initialization values: {f.variables['initialization'][:]}")

        print(f"Successfully created file: {output_file}")
        return True

    except Exception as e:
        print(f"Error: {e}")
        if os.path.exists(output_file):
            os.remove(output_file)
        return False
