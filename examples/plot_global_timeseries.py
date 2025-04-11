#!/usr/bin/env python3
"""
Plot time series of globally averaged temperature anomalies from processed NetCDF files.

This script visualizes temperature anomaly data from the combined NetCDF files,
showing different initialization years and ensemble members.
"""

import argparse
import glob
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import pandas as pd

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("plot_timeseries")


def load_ensemble_data(filepath: str) -> xr.Dataset:
    """
    Load ensemble data from NetCDF file.

    Parameters
    ----------
    filepath : str
        Path to the NetCDF file

    Returns
    -------
    xr.Dataset
        Dataset containing the ensemble data
    """
    logger.info(f"Loading data from {filepath}")
    try:
        ds = xr.open_dataset(filepath)
        return ds
    except Exception as e:
        logger.error(f"Error loading {filepath}: {e}")
        raise


def compute_global_average(
    ds: xr.Dataset, variable: str = "tas", lat_name: str = "lat", lon_name: str = "lon"
) -> xr.DataArray:
    """
    Compute globally averaged values weighted by grid cell area (cosine of latitude).

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset
    variable : str, optional
        Variable name, by default "tas"
    lat_name : str, optional
        Name of latitude dimension, by default "lat"
    lon_name : str, optional
        Name of longitude dimension, by default "lon"

    Returns
    -------
    xr.DataArray
        Globally averaged data
    """
    logger.info(f"Computing global average for {variable}")

    # Create weights based on cosine of latitude
    weights = np.cos(np.deg2rad(ds[lat_name]))
    weights = weights / weights.sum()

    # Apply weights and compute global average
    weighted = ds[variable] * weights
    global_avg = weighted.mean(dim=[lat_name, lon_name])

    return global_avg


def create_time_axis(init_years: List[int], lead_times: List[int]) -> Dict[int, List[datetime]]:
    """
    Create time axis for each initialization year.

    Parameters
    ----------
    init_years : List[int]
        List of initialization years
    lead_times : List[int]
        List of lead times (months)

    Returns
    -------
    Dict[int, List[datetime]]
        Dictionary mapping initialization years to lists of datetime objects
    """
    time_axis = {}

    for year in init_years:
        # Start from November of initialization year
        times = []
        for lead in lead_times:
            # Add lead months to initialization date
            date = datetime(year, 11, 1) + timedelta(days=30 * lead)
            times.append(date)
        time_axis[year] = times

    return time_axis


def compute_yearly_mean(global_avg: xr.DataArray) -> Tuple[List[datetime], List[float]]:
    """
    Compute yearly mean values from monthly data.

    Parameters
    ----------
    global_avg : xr.DataArray
        Globally averaged monthly data

    Returns
    -------
    Tuple[List[datetime], List[float]]
        List of yearly dates and corresponding mean values
    """
    # Get time values and convert to datetime
    time_values = global_avg.time.values
    if isinstance(time_values[0], np.datetime64):
        dates = [pd.to_datetime(t) for t in time_values]
    else:
        time_units = global_avg.time.attrs.get('units', 'days since 1958-01-01')
        dates = [datetime(1958, 1, 1) + timedelta(days=int(t)) for t in time_values]

    # Group by year and compute mean
    yearly_data = {}
    for date, value in zip(dates, global_avg.values):
        year = date.year
        if year not in yearly_data:
            yearly_data[year] = []
        yearly_data[year].append(value)

    # Compute yearly means
    yearly_dates = []
    yearly_means = []
    for year in sorted(yearly_data.keys()):
        yearly_dates.append(datetime(year, 7, 1))  # Use July 1st as representative date
        yearly_means.append(np.mean(yearly_data[year]))

    return yearly_dates, yearly_means


def plot_timeseries(
    global_avg: xr.DataArray,
    output_file: Optional[str] = None,
    ensemble_label: Optional[str] = None,
    title: Optional[str] = None,
) -> Tuple[Figure, Axes]:
    """
    Plot time series of temperature anomalies for each initialization.

    Parameters
    ----------
    global_avg : xr.DataArray
        Globally averaged data
    output_file : str, optional
        Path to save the figure, by default None
    ensemble_label : str, optional
        Label for the ensemble, by default None
    title : str, optional
        Title for the plot, by default None

    Returns
    -------
    Tuple[Figure, Axes]
        Figure and axes objects
    """
    logger.info("Creating time series plot...")

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Get initialization dates
    init_dates = pd.to_datetime(global_avg.time.values)
    
    # For each initialization date, plot the forecast
    for init_date in init_dates:
        # Get the forecast for this initialization
        forecast = global_avg.sel(time=init_date)
        
        # Create forecast dates (42 months from initialization)
        forecast_dates = [init_date + pd.DateOffset(months=i) for i in range(len(forecast))]
        
        # Plot the forecast as a single gray line
        ax.plot(forecast_dates, forecast.values,
                color='gray',
                linewidth=1.5,
                alpha=0.5)

    # Format plot
    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title("Temperature Anomalies by Initialization", fontsize=14)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Temperature Anomaly (K)", fontsize=12)
    ax.grid(True, alpha=0.3)

    # Format x-axis as dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)

    # Adjust layout
    plt.tight_layout()

    # Save figure if output file is specified
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        logger.info(f"Figure saved to {output_file}")

    return fig, ax


def plot_multi_ensemble(
    data_dir: str,
    ensemble_ids: List[str],
    output_dir: Optional[str] = None,
    file_pattern: str = "tas_{}_combined.nc",
    variable: str = "tas",
) -> None:
    """
    Plot time series for multiple ensemble members.

    Parameters
    ----------
    data_dir : str
        Directory containing NetCDF files
    ensemble_ids : List[str]
        List of ensemble member IDs
    output_dir : str, optional
        Directory to save figures, by default None
    file_pattern : str, optional
        Pattern for NetCDF filenames, by default "tas_{}_combined.nc"
    variable : str, optional
        Variable to plot, by default "tas"
    """
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    # Process each ensemble
    for i, ensemble in enumerate(ensemble_ids):
        # Load data
        filepath = os.path.join(data_dir, file_pattern.format(ensemble))
        if not os.path.exists(filepath):
            logger.warning(f"File not found: {filepath}")
            continue

        try:
            ds = load_ensemble_data(filepath)
            logger.info(f"\nProcessing ensemble {ensemble}")
            logger.info(f"Dataset dimensions: {ds.dims}")
            logger.info(f"Time coordinate info: {ds.time}")

            # Compute global average
            global_avg = compute_global_average(ds, variable=variable)

            # Create individual plot
            if output_dir is not None:
                output_file = os.path.join(output_dir, f"global_avg_{ensemble}.png")
                plot_timeseries(
                    global_avg,
                    output_file=output_file,
                    ensemble_label=ensemble,
                    title=f"Temperature Anomalies - {ensemble}",
                )

        except Exception as e:
            logger.error(f"Error processing {ensemble}: {e}")

    # Create a combined plot for all ensembles
    fig_combined, ax_combined = plt.subplots(figsize=(14, 8))
    
    # Process each ensemble again for the combined plot
    for i, ensemble in enumerate(ensemble_ids):
        filepath = os.path.join(data_dir, file_pattern.format(ensemble))
        if not os.path.exists(filepath):
            continue

        try:
            ds = load_ensemble_data(filepath)
            global_avg = compute_global_average(ds, variable=variable)
            
            # Get time values and convert to datetime
            time_values = global_avg.time.values
            if isinstance(time_values[0], np.datetime64):
                dates = [pd.to_datetime(t) for t in time_values]
            else:
                time_units = global_avg.time.attrs.get('units', 'days since 1958-01-01')
                dates = [datetime(1958, 1, 1) + timedelta(days=int(t)) for t in time_values]

            # Find all unique initialization dates (November dates)
            init_indices = []
            current_year = None
            for i, date in enumerate(dates):
                if date.month == 11:
                    if current_year != date.year:  # Only take first November of each year
                        init_indices.append(i)
                        current_year = date.year

            # Plot each initialization's forecast
            for init_idx in init_indices:
                if init_idx + 42 > len(dates):
                    continue
                    
                # Get the next 42 months of data
                forecast_dates = dates[init_idx:init_idx+42]
                forecast_values = global_avg.values[init_idx:init_idx+42]
                
                # Plot the forecast as a single gray line
                ax_combined.plot(forecast_dates, forecast_values,
                               color='gray',
                               linewidth=1.5,
                               alpha=0.5)

        except Exception as e:
            logger.error(f"Error processing {ensemble} for combined plot: {e}")

    # Format combined plot
    ax_combined.set_title("Temperature Anomalies by Initialization", fontsize=14)
    ax_combined.set_xlabel("Date", fontsize=12)
    ax_combined.set_ylabel("Temperature Anomaly (K)", fontsize=12)
    ax_combined.grid(True, alpha=0.3)
    ax_combined.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax_combined.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save combined plot
    if output_dir is not None:
        output_file = os.path.join(output_dir, "global_avg_all_ensembles.png")
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        logger.info(f"Combined figure saved to {output_file}")
    else:
        plt.show()


def find_ensemble_files(data_dir: str, pattern: str = "tas_*_combined.nc") -> List[str]:
    """
    Find ensemble files in the data directory.

    Parameters
    ----------
    data_dir : str
        Directory containing NetCDF files
    pattern : str, optional
        Glob pattern to match files, by default "tas_*_combined.nc"

    Returns
    -------
    List[str]
        List of ensemble IDs
    """
    files = glob.glob(os.path.join(data_dir, pattern))

    # Extract ensemble IDs from filenames
    ensemble_ids = []
    for file in files:
        filename = os.path.basename(file)
        # Extract ensemble ID from tas_<ensemble>_combined.nc
        parts = filename.split("_")
        if len(parts) >= 3 and parts[0] == "tas" and parts[-1] == "combined.nc":
            ensemble = "_".join(parts[1:-1])
            ensemble_ids.append(ensemble)

    return ensemble_ids


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Plot time series of globally averaged temperature anomalies"
    )
    parser.add_argument(
        "--data-dir", type=str, default="output", help="Directory containing NetCDF files"
    )
    parser.add_argument("--ensembles", type=str, help="Comma-separated list of ensemble member IDs")
    parser.add_argument(
        "--output-dir", type=str, default="figures", help="Directory to save figures"
    )
    parser.add_argument("--variable", type=str, default="tas", help="Variable to plot")
    parser.add_argument(
        "--no-save", action="store_true", help="Don't save figures, just display them"
    )

    args = parser.parse_args()

    # Determine output directory
    output_dir = None if args.no_save else args.output_dir

    # Get ensemble IDs
    if args.ensembles:
        ensemble_ids = args.ensembles.split(",")
    else:
        # Auto-detect ensemble IDs from files
        ensemble_ids = find_ensemble_files(args.data_dir)

    if not ensemble_ids:
        logger.error(f"No ensemble files found in {args.data_dir}")
        return 1

    logger.info(f"Found ensemble IDs: {ensemble_ids}")

    # Plot time series
    plot_multi_ensemble(
        data_dir=args.data_dir,
        ensemble_ids=ensemble_ids,
        output_dir=output_dir,
        variable=args.variable,
    )

    # If not saving, show the plots
    if args.no_save:
        plt.show()

    return 0


if __name__ == "__main__":
    exit(main())
