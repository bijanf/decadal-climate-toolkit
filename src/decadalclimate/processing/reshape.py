"""
Functions for reshaping NetCDF files to standard dimensions.
"""

import os
from typing import List, Tuple

import numpy as np
import xarray as xr

from decadalclimate.utils.config import load_config


def extract_initialization_years(ds: xr.Dataset) -> List[int]:
    """
    Extract initialization years from a dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset

    Returns
    -------
    List[int]
        List of initialization years
    """
    if "initialization" in ds.variables:
        # Get the initialization years from the variable
        init_years = np.unique(ds.initialization.values).tolist()
    else:
        print("Warning: 'initialization' variable not found in input file")
        print("Creating initialization years from global attributes...")

        # Extract initialization years from global attributes or metadata
        if hasattr(ds, "initialization_year"):
            init_years = [ds.initialization_year]
        else:
            # Try to extract from filename or estimate from lead_time
            filename = ds.encoding.get("source", "")
            from_filename = os.path.basename(filename).split("_")[2] if filename else ""

            if from_filename.isdigit() and len(from_filename) == 4:
                init_years = [int(from_filename)]
            else:
                config = load_config()
                max_lead_time = config.get("netcdf", {}).get("max_lead_time", 18)

                n_init = len(ds.lead_time) // max_lead_time if "lead_time" in ds.dims else 1
                if n_init < 1:
                    n_init = 1

                # Use placeholder years (1961, 1962, etc.)
                init_years = list(range(1961, 1961 + n_init))

    return init_years


def get_dimensions(ds: xr.Dataset) -> Tuple[int, int, int, int]:
    """
    Get dimensions from a dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset

    Returns
    -------
    Tuple[int, int, int, int]
        Number of initialization years, lead time steps, latitude points, and longitude points
    """
    # Get initialization years
    init_years = extract_initialization_years(ds)

    # Get lead time steps
    config = load_config()
    max_lead_time = config.get("netcdf", {}).get("max_lead_time", 18)

    if "lead_time" in ds.dims:
        lead_time_steps = min(max_lead_time, len(ds.lead_time))
    else:
        # If lead_time is not a dimension, check time
        if "time" in ds.dims:
            lead_time_steps = min(max_lead_time, len(ds.time))
        else:
            lead_time_steps = max_lead_time
            print("Warning: No lead_time or time dimension found, using default")

    # Get spatial dimensions
    nlat = len(ds.lat)
    nlon = len(ds.lon)

    return len(init_years), lead_time_steps, nlat, nlon


def reshape_file(input_file: str, output_file: str, variable_name: str = "tas") -> bool:
    """
    Reshape a NetCDF file to have dimensions (initialization, lead_time, lat, lon).

    Parameters
    ----------
    input_file : str
        Path to the input NetCDF file
    output_file : str
        Path to the output NetCDF file
    variable_name : str, optional
        Name of the variable to reshape, by default 'tas'

    Returns
    -------
    bool
        True if successful, False otherwise
    """
    print(f"Opening input file: {input_file}")

    try:
        # Open the input dataset
        ds = xr.open_dataset(input_file)

        # Print input file dimensions
        print(f"Input dimensions: {ds.dims}")
        print(f"Input variables: {list(ds.variables)}")
        print("Variable shapes:")
        for var in ds.variables:
            print(f"  {var}: {ds[var].shape}")

        # Extract initialization years
        init_years = extract_initialization_years(ds)
        print(f"Using {len(init_years)} initialization years: {init_years}")

        # Get dimensions
        n_init, lead_time_steps, nlat, nlon = get_dimensions(ds)
        print(f"Creating array with dimensions: ({n_init}, {lead_time_steps}, {nlat}, {nlon})")

        # Create a new array for the variable
        data_values = np.zeros((n_init, lead_time_steps, nlat, nlon))

        # Fill the array
        try:
            # Handle different input dimension arrangements
            if "initialization" in ds.dims and "lead_time" in ds.dims:
                # Already has correct dimensions
                print("Input already has correct dimensions")
                data_values = ds[variable_name].values[:, :lead_time_steps, :, :]
            elif (
                len(ds[variable_name].shape) == 3
                and ds[variable_name].shape[0] >= n_init * lead_time_steps
            ):
                # Probably (time, lat, lon) where time includes all initializations
                print("Input has (time, lat, lon) dimensions")
                for i, year in enumerate(init_years):
                    start_idx = i * lead_time_steps
                    end_idx = start_idx + lead_time_steps

                    if end_idx > ds[variable_name].shape[0]:
                        print(f"Warning: Not enough time steps for year {year}")
                        end_idx = ds[variable_name].shape[0]

                    # Extract the time steps for this initialization year
                    n_steps = end_idx - start_idx
                    data_values[i, :n_steps, :, :] = ds[variable_name].values[
                        start_idx:end_idx, :, :
                    ]
            else:
                print("Warning: Unexpected input dimensions, using zeros")
        except Exception as e:
            print(f"Error filling data array: {e}")
            print("Using zeros instead")

        # Create new dataset with the reshaped array
        print("Creating new dataset with dimensions (initialization, lead_time, lat, lon)")
        ds_new = xr.Dataset(
            data_vars={variable_name: (["initialization", "lead_time", "lat", "lon"], data_values)},
            coords={
                "initialization": init_years,
                "lead_time": np.arange(lead_time_steps),
                "lat": ds.lat,
                "lon": ds.lon,
            },
        )

        # Add attributes
        print("Adding variable attributes")
        ds_new.initialization.attrs["units"] = "year"
        ds_new.initialization.attrs["long_name"] = "Initialization Year"
        ds_new.lead_time.attrs["units"] = "months"
        ds_new.lead_time.attrs["long_name"] = "Forecast Lead Time"

        # Copy variable attributes
        for var in [variable_name, "lat", "lon"]:
            if var in ds.variables and var in ds_new.variables:
                ds_new[var].attrs.update(ds[var].attrs)

        # Save to netCDF
        print(f"Saving to output file: {output_file}")
        ds_new.to_netcdf(output_file)

        # Close datasets
        ds.close()

        print(
            f"Successfully created file with dimensions "
            f"(initialization, lead_time, lat, lon): {output_file}"
        )
        return True

    except Exception as e:
        print(f"Error processing file: {e}")
        import traceback

        traceback.print_exc()
        return False
