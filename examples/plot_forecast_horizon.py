#!/usr/bin/env python3
"""
Plot temperature anomalies at different forecast horizons (lead times).

This script creates visualizations comparing different ensemble members
and their forecast performance across lead times and initialization years.
"""

import argparse
import glob
import logging
import os
import sys
from typing import Dict, List, Optional, Union

# Set the matplotlib backend to a non-interactive one to avoid display-related crashes
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.gridspec import GridSpec

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("plot_forecast")


# Add a handler to log detailed errors
def setup_error_logging():
    error_handler = logging.FileHandler("plot_error.log")
    error_handler.setLevel(logging.ERROR)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    error_handler.setFormatter(formatter)
    logger.addHandler(error_handler)
    logger.info("Error logging set up, details will be written to plot_error.log")


setup_error_logging()


# Create a function to safely load datasets to avoid memory issues
def safe_load_dataset(filepath: str) -> Optional[xr.Dataset]:
    """Load a dataset safely with error handling and memory management."""
    try:
        # Use dask for chunked loading to reduce memory usage
        ds = xr.open_dataset(filepath, chunks={"lead_time": 1})
        return ds
    except Exception as e:
        logger.error(f"Error loading {filepath}: {str(e)}")
        return None


def load_ensemble_datasets(
    data_dir: str, ensemble_ids: List[str], file_pattern: str = "tas_{}_combined.nc"
) -> Dict[str, xr.Dataset]:
    """
    Load multiple ensemble datasets.

    Parameters
    ----------
    data_dir : str
        Directory containing NetCDF files
    ensemble_ids : List[str]
        List of ensemble member IDs
    file_pattern : str, optional
        Pattern for NetCDF filenames, by default "tas_{}_combined.nc"

    Returns
    -------
    Dict[str, xr.Dataset]
        Dictionary mapping ensemble IDs to datasets
    """
    datasets = {}

    # Log the total number of ensembles to process
    logger.info(f"Attempting to load {len(ensemble_ids)} ensemble datasets")

    for i, ensemble in enumerate(ensemble_ids):
        filepath = os.path.join(data_dir, file_pattern.format(ensemble))
        if not os.path.exists(filepath):
            logger.warning(f"File not found: {filepath}")
            continue

        logger.info(f"Loading data from {filepath} ({i+1}/{len(ensemble_ids)})")
        try:
            # Use the safe loading function
            ds = safe_load_dataset(filepath)
            if ds is not None:
                datasets[ensemble] = ds
                logger.info(f"Successfully loaded dataset for {ensemble}")
        except Exception as e:
            logger.error(f"Unexpected error loading {filepath}: {e}")
            import traceback

            logger.error(traceback.format_exc())
            continue

    logger.info(f"Successfully loaded {len(datasets)} out of {len(ensemble_ids)} datasets")
    return datasets


def compute_global_averages(
    datasets: Dict[str, xr.Dataset], variable: str = "tas"
) -> Dict[str, xr.DataArray]:
    """
    Compute globally averaged values for multiple datasets.

    Parameters
    ----------
    datasets : Dict[str, xr.Dataset]
        Dictionary mapping ensemble IDs to datasets
    variable : str, optional
        Name of the variable to average, by default 'tas'

    Returns
    -------
    Dict[str, xr.DataArray]
        Dictionary mapping ensemble IDs to globally averaged data
    """
    global_avgs = {}

    for ensemble, ds in datasets.items():
        logger.info(f"Computing global average for {ensemble}")

        try:
            # Get variable
            if variable not in ds:
                logger.warning(f"Variable {variable} not found in dataset for {ensemble}")
                continue
            da = ds[variable]

            # Create weights based on cosine of latitude
            weights = np.cos(np.deg2rad(ds.lat))
            weights = weights / weights.sum()

            # Apply weights to latitude dimension
            weighted = da.weighted(weights)

            # Average over lat/lon dimensions
            global_avg = weighted.mean(dim=("lat", "lon"))
            global_avgs[ensemble] = global_avg

        except Exception as e:
            logger.error(f"Error computing global average for {ensemble}: {e}")
            continue

    return global_avgs


def get_initialization_years(dataset: Union[xr.Dataset, xr.DataArray]) -> List[int]:
    """
    Extract initialization years from a dataset or dataarray.

    Parameters
    ----------
    dataset : Union[xr.Dataset, xr.DataArray]
        Dataset or DataArray to extract initialization years from

    Returns
    -------
    List[int]
        List of initialization years
    """
    # Check if initialization is a dimension with coordinate values
    if "initialization" in dataset.dims and "initialization" in dataset.coords:
        return sorted(list(dataset.initialization.values))

    # Check if initialization is a variable in dataset
    if isinstance(dataset, xr.Dataset) and "initialization" in dataset.variables:
        init_val = dataset.initialization.values
        if isinstance(init_val, np.ndarray) and init_val.size > 1:
            return sorted([int(x) for x in init_val])
        else:
            return [int(init_val)]

    # Check if dataset has an initialization attribute
    if hasattr(dataset, "initialization"):
        init_value = dataset.initialization
        if isinstance(init_value, (int, float)):
            return [int(init_value)]

    # If we can't find initialization, log a warning and use a placeholder
    logger.warning(
        f"Could not determine initialization years for dataset, using placeholder."
        f" Dataset info: {dataset}"
    )
    return [1961]  # Placeholder value


def plot_lead_time_comparison(
    global_avgs: Dict[str, xr.DataArray], init_year: int, output_file: Optional[str] = None
) -> None:
    """
    Plot comparison of temperature anomalies at different lead
    times for a specific initialization year.

    Parameters
    ----------
    global_avgs : Dict[str, xr.DataArray]
        Dictionary mapping ensemble IDs to globally averaged data
    init_year : int
        Initialization year to compare
    output_file : str, optional
        Path to save the figure, by default None
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Colors for different ensembles
    colors = plt.cm.tab10.colors

    # Plot each ensemble
    for i, (ensemble, data) in enumerate(global_avgs.items()):
        try:
            # Create a dataset to work with
            if isinstance(data, xr.DataArray):
                ds = data.to_dataset()
            else:
                ds = data

            # Check the structure of the data
            init_years = get_initialization_years(ds)
            logger.info(f"Available initialization years for {ensemble}: {init_years}")

            # Skip if this initialization year isn't available
            if init_year not in init_years:
                logger.warning(f"Initialization year {init_year} not found for {ensemble}")
                continue

            # If initialization is a dimension with multiple values
            if "initialization" in data.dims:
                try:
                    # Try to select the specific initialization year
                    da = data.sel(initialization=init_year)
                    x_values = np.arange(len(da.lead_time))
                    y_values = da.values
                except Exception as e:
                    logger.error(
                        f"Error selecting initialization year {init_year} for {ensemble}: {e}"
                    )
                    continue
            else:
                # For datasets with only a single initialization year
                # First check if that year matches our target year
                ds_init_year = init_years[0] if init_years else None
                if ds_init_year != init_year:
                    logger.warning(
                        f"Initialization year {init_year} doesn't match dataset"
                        f" year {ds_init_year} for {ensemble}"
                    )
                    continue

                # Use the entire data array (already just one initialization)
                x_values = np.arange(len(data))
                y_values = data.values

            # Plot lead time vs. temperature anomaly
            ax.plot(
                x_values,
                y_values,
                marker="o",
                ms=6,
                label=ensemble,
                color=colors[i % len(colors)],
                linewidth=2,
            )

        except Exception as e:
            logger.error(f"Error plotting {ensemble}: {e}")
            logger.error(f"Error details: {str(e)}")
            continue

    # Add legend, grid, and labels
    ax.legend(loc="best", frameon=True, fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_title(
        f"Temperature Anomalies by Lead Time - Initialization Year: {init_year}", fontsize=14
    )
    ax.set_xlabel("Lead Time (months)", fontsize=12)
    ax.set_ylabel("Temperature Anomaly (K)", fontsize=12)

    # Adjust layout
    plt.tight_layout()

    # Save if output file is provided
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        logger.info(f"Figure saved to {output_file}")


def plot_ensemble_matrix(
    global_avgs: Dict[str, xr.DataArray],
    init_years: Optional[List[int]] = None,
    max_lead_time: Optional[int] = None,
    output_file: Optional[str] = None,
) -> None:
    """
    Plot matrix of ensemble temperature anomalies for different initialization years and lead times.

    Parameters
    ----------
    global_avgs : Dict[str, xr.DataArray]
        Dictionary mapping ensemble IDs to globally averaged data
    init_years : List[int], optional
        List of initialization years to include, by default all available years
    max_lead_time : int, optional
        Maximum lead time to include, by default all available lead times
    output_file : str, optional
        Path to save the figure, by default None
    """
    # Determine all available initialization years across all ensembles
    all_init_years = set()
    for ensemble, data in global_avgs.items():
        ds = data.to_dataset() if isinstance(data, xr.DataArray) else data
        years = set(get_initialization_years(ds))
        all_init_years.update(years)

    if not all_init_years:
        logger.error("No initialization years found across ensembles")
        return

    # Use specified years if provided, otherwise use all available years
    if init_years is not None:
        years_to_use = sorted([y for y in init_years if y in all_init_years])
        if not years_to_use:
            logger.error("No specified years found in the data")
            return
    else:
        years_to_use = sorted(all_init_years)

    # Find the maximum lead time for each dataset
    max_lead_times = []
    for ensemble, data in global_avgs.items():
        if "lead_time" in data.dims:
            max_lead_times.append(len(data.lead_time))
        else:
            max_lead_times.append(len(data))

    # Set the maximum lead time to use
    if max_lead_time is None:
        max_lead_time = min(max_lead_times) if max_lead_times else 18
    else:
        max_lead_time = min(max_lead_time, min(max_lead_times))

    # Create figure with gridspec
    n_ensembles = len(global_avgs)
    n_years = len(years_to_use)

    # Adjust figure size based on number of plots
    if n_years > 5:
        fig_width = min(20, n_years * 2.5)
    else:
        fig_width = n_years * 3

    fig = plt.figure(figsize=(fig_width, n_ensembles * 2.5))
    gs = GridSpec(n_ensembles, n_years, figure=fig, wspace=0.3, hspace=0.4)

    # Used for common y-axis limits
    vmin, vmax = float("inf"), float("-inf")
    for ensemble, data in global_avgs.items():
        ds = data.to_dataset() if isinstance(data, xr.DataArray) else data
        ds_years = get_initialization_years(ds)

        for year in years_to_use:
            if year in ds_years:
                # Get data for this initialization year
                try:
                    if "initialization" in data.dims:
                        # Multi-year dataset
                        values = (
                            data.sel(initialization=year)
                            .isel(lead_time=slice(0, max_lead_time))
                            .values
                        )
                    else:
                        # Single-year dataset - check if it matches the current year
                        if ds_years[0] == year:
                            values = data.isel(lead_time=slice(0, max_lead_time)).values
                        else:
                            continue

                    if values.size > 0:
                        vmin = min(vmin, np.nanmin(values))
                        vmax = max(vmax, np.nanmax(values))
                except Exception as e:
                    logger.error(f"Error getting data for {ensemble}, year {year}: {e}")
                    continue

    # Add padding to y-axis limits
    if vmin != float("inf") and vmax != float("-inf"):
        padding = (vmax - vmin) * 0.1
        vmin -= padding
        vmax += padding
    else:
        # Fallback if no valid data was found
        vmin, vmax = -1, 1

    # Plot each ensemble and initialization year
    for i, (ensemble, data) in enumerate(global_avgs.items()):
        ds = data.to_dataset() if isinstance(data, xr.DataArray) else data
        ds_years = get_initialization_years(ds)

        for j, year in enumerate(years_to_use):
            ax = fig.add_subplot(gs[i, j])

            if year in ds_years:
                try:
                    # Get data for this initialization year
                    if "initialization" in data.dims:
                        # Multi-year dataset
                        da_values = (
                            data.sel(initialization=year)
                            .isel(lead_time=slice(0, max_lead_time))
                            .values
                        )
                        x_values = np.arange(len(da_values))
                    else:
                        # Single-year dataset - check if it matches the current year
                        if ds_years[0] == year:
                            da_values = data.isel(lead_time=slice(0, max_lead_time)).values
                            x_values = np.arange(len(da_values))
                        else:
                            ax.text(
                                0.5,
                                0.5,
                                "No Data",
                                ha="center",
                                va="center",
                                transform=ax.transAxes,
                            )
                            ax.set_xticks([])
                            ax.set_yticks([])
                            continue

                    # Plot lead time vs. temperature anomaly
                    ax.plot(x_values, da_values, marker="o", ms=5, linewidth=1.5)

                    # Set title and y-limits
                    if i == 0:
                        ax.set_title(f"Init: {year}", fontsize=12)

                    if j == 0:
                        ax.set_ylabel(ensemble, fontsize=12)

                    ax.set_ylim(vmin, vmax)
                    ax.grid(True, alpha=0.3)

                    # Only show x-labels for bottom row
                    if i == n_ensembles - 1:
                        ax.set_xlabel("Lead Time (months)", fontsize=10)
                    else:
                        ax.set_xticklabels([])
                except Exception as e:
                    logger.error(f"Error plotting {ensemble}, year {year}: {e}")
                    ax.text(0.5, 0.5, "Error", ha="center", va="center", transform=ax.transAxes)
                    ax.set_xticks([])
                    ax.set_yticks([])
            else:
                ax.text(0.5, 0.5, "No Data", ha="center", va="center", transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])

    # Add overall title
    plt.suptitle(
        "Temperature Anomalies by Ensemble, Initialization Year, and Lead Time", fontsize=16, y=0.98
    )

    # Save if output file is provided
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        logger.info(f"Figure saved to {output_file}")


def plot_all_timeseries(
    global_avgs: Dict[str, xr.DataArray], output_file: Optional[str] = None
) -> None:
    """
    Plot all temperature anomaly time series in a single plot, using actual years on the x-axis.

    Parameters
    ----------
    global_avgs : Dict[str, xr.DataArray]
        Dictionary mapping ensemble IDs to globally averaged data
    output_file : str, optional
        Path to save the figure, by default None
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Colors for different ensembles
    colors = plt.cm.tab10.colors

    # Find the minimum and maximum years for the x-axis
    min_year = 3000  # A future year that will be updated
    max_year = 0  # A past year that will be updated

    # Store lines to add labels later
    lines_by_ensemble = {}

    # Plot each ensemble
    for i, (ensemble, data) in enumerate(global_avgs.items()):
        color = colors[i % len(colors)]
        lines_by_ensemble[ensemble] = []

        # Check if there's an initialization dimension
        if "initialization" in data.dims:
            # For each initialization year, create a time series
            for init_year in data.initialization.values:
                try:
                    # Get data for this initialization year
                    da = data.sel(initialization=init_year)

                    # Create x-axis as actual years (initialization year + lead time in months/12)
                    # Convert lead time from months to years and add to initialization year
                    if "lead_time" in da.dims:
                        lead_times = da.lead_time.values
                    else:
                        lead_times = np.arange(len(da))

                    # Create time points in years (including fractional years for months)
                    time_points = init_year + lead_times / 12

                    # Update min and max years for x-axis limits
                    min_year = min(min_year, time_points[0])
                    max_year = max(max_year, time_points[-1])

                    # Plot with actual years as x-axis
                    (line,) = ax.plot(
                        time_points,
                        da.values,
                        marker="o",
                        ms=4,
                        alpha=0.8,
                        color=color,
                        linewidth=1.5,
                    )

                    # Store the line for this ensemble
                    lines_by_ensemble[ensemble].append(line)

                except Exception as e:
                    logger.error(f"Error plotting {ensemble}, year {init_year}: {e}")
        else:
            # Single initialization year
            try:
                # Get the initialization year
                ds = data.to_dataset() if isinstance(data, xr.DataArray) else data
                init_years = get_initialization_years(ds)
                if not init_years:
                    logger.warning(f"No initialization year found for {ensemble}, using default")
                    init_year = 1970  # Default
                else:
                    init_year = init_years[0]

                # Create x-axis as actual years
                lead_times = np.arange(len(data))
                time_points = init_year + lead_times / 12

                # Update min and max years for x-axis limits
                min_year = min(min_year, time_points[0])
                max_year = max(max_year, time_points[-1])

                # Plot with actual years as x-axis
                (line,) = ax.plot(
                    time_points, data.values, marker="o", ms=5, color=color, linewidth=2
                )

                # Store the line for this ensemble
                lines_by_ensemble[ensemble].append(line)

            except Exception as e:
                logger.error(f"Error plotting {ensemble}: {e}")

    # Add a legend with one entry per ensemble
    legend_handles = []
    legend_labels = []
    for ensemble, lines in lines_by_ensemble.items():
        if lines:  # If there are any lines for this ensemble
            legend_handles.append(lines[0])
            legend_labels.append(ensemble)

    ax.legend(legend_handles, legend_labels, loc="best", frameon=True, fontsize=10)

    # Set x-axis limits and format
    if min_year < 3000 and max_year > 0:  # If we found valid years
        # Add a small padding to the x-axis
        padding = (max_year - min_year) * 0.05
        ax.set_xlim(min_year - padding, max_year + padding)

        # Format the x-axis with years
        ax.xaxis.set_major_locator(mdates.YearLocator(2))  # Show every 2 years
        ax.xaxis.set_minor_locator(mdates.YearLocator(1))  # Minor tick every year
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Add grid, title, and labels
    ax.grid(True, alpha=0.3)
    ax.set_title("Temperature Anomalies for All Ensemble Members", fontsize=14)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Temperature Anomaly (K)", fontsize=12)

    # Adjust layout
    plt.tight_layout()

    # Save if output file is provided
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        logger.info(f"Figure saved to {output_file}")
    else:
        plt.show()


def plot_ensemble_comparison(
    data_dir: str,
    ensemble_ids: List[str],
    output_dir: Optional[str] = None,
    init_years: Optional[List[int]] = None,
    variable: str = "tas",
    plot_individual: bool = False,
) -> None:
    """
    Create plots comparing ensemble forecasts.

    Parameters
    ----------
    data_dir : str
        Directory containing NetCDF files
    ensemble_ids : List[str]
        List of ensemble member IDs
    output_dir : str, optional
        Directory to save figures, by default None
    init_years : List[int], optional
        List of initialization years to include, by default all available years
    variable : str, optional
        Variable to plot, by default "tas"
    plot_individual : bool, optional
        Whether to create individual plots for each initialization year, by default False
    """
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    # Load datasets
    datasets = load_ensemble_datasets(data_dir, ensemble_ids)

    if not datasets:
        logger.error("No valid datasets loaded")
        return

    # Log the structure of the loaded datasets
    for ensemble, ds in datasets.items():
        logger.info(f"Dataset dimensions for {ensemble}: {ds.dims}")
        logger.info(f"Dataset variables for {ensemble}: {list(ds.variables.keys())}")

    # Compute global averages
    global_avgs = compute_global_averages(datasets, variable=variable)

    if not global_avgs:
        logger.error("No global averages computed")
        return

    # Create a single plot with all time series
    if output_dir is not None:
        output_file = os.path.join(output_dir, "all_timeseries.png")
    else:
        output_file = None

    plot_all_timeseries(global_avgs, output_file=output_file)

    # Optionally create individual plots
    if plot_individual:
        # Determine available initialization years if not specified
        if init_years is None:
            init_years = set()
            for ensemble, data in global_avgs.items():
                ds = data.to_dataset() if isinstance(data, xr.DataArray) else data
                years = get_initialization_years(ds)
                init_years.update(years)

            init_years = sorted(init_years)

        logger.info(f"Using initialization years: {init_years}")

        # Create lead time comparison plots for each initialization year
        for year in init_years:
            if output_dir is not None:
                output_file = os.path.join(output_dir, f"lead_time_comparison_{year}.png")
            else:
                output_file = None

            plot_lead_time_comparison(global_avgs, year, output_file=output_file)

        # Create ensemble matrix plot
        if output_dir is not None:
            output_file = os.path.join(output_dir, "ensemble_matrix.png")
        else:
            output_file = None

        plot_ensemble_matrix(global_avgs, init_years=init_years, output_file=output_file)

    # If not saving, show the plots
    if output_dir is None:
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
        description="Plot comparison of ensemble temperature anomalies at different lead times"
    )
    parser.add_argument(
        "--data-dir", type=str, default="output", help="Directory containing NetCDF files"
    )
    parser.add_argument("--ensembles", type=str, help="Comma-separated list of ensemble member IDs")
    parser.add_argument(
        "--init-years", type=str, help="Comma-separated list of initialization years"
    )
    parser.add_argument(
        "--output-dir", type=str, default="figures", help="Directory to save figures"
    )
    parser.add_argument("--variable", type=str, default="tas", help="Variable to plot")
    parser.add_argument(
        "--no-save", action="store_true", help="Don't save figures, just display them"
    )
    parser.add_argument(
        "--individual",
        action="store_true",
        help="Create individual plots for each initialization year",
    )
    # Add a new argument to limit the number of ensembles to process
    parser.add_argument(
        "--max-ensembles",
        type=int,
        default=None,
        help="Maximum number of ensembles to process (default: all)",
    )
    # Add debug mode
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with extra logging")

    args = parser.parse_args()

    # Set debug mode if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")

    # Make sure output directory exists
    output_dir = None if args.no_save else args.output_dir
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")

    # Get ensemble IDs
    if args.ensembles:
        ensemble_ids = args.ensembles.split(",")
    else:
        # Auto-detect ensemble IDs from files
        ensemble_ids = find_ensemble_files(args.data_dir)

    if not ensemble_ids:
        logger.error(f"No ensemble files found in {args.data_dir}")
        return 1

    # Limit number of ensembles if specified
    if args.max_ensembles and len(ensemble_ids) > args.max_ensembles:
        logger.info(f"Limiting to {args.max_ensembles} ensembles out of {len(ensemble_ids)}")
        ensemble_ids = ensemble_ids[: args.max_ensembles]

    logger.info(f"Found ensemble IDs: {ensemble_ids}")

    # Get initialization years
    init_years = None
    if args.init_years:
        init_years = [int(y) for y in args.init_years.split(",")]

    try:
        # Plot ensemble comparison
        plot_ensemble_comparison(
            data_dir=args.data_dir,
            ensemble_ids=ensemble_ids,
            output_dir=output_dir,
            init_years=init_years,
            variable=args.variable,
            plot_individual=args.individual,
        )
    except Exception as e:
        logger.error(f"Error in plot_ensemble_comparison: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return 1

    logger.info("Script completed successfully")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        logger.error(f"Unhandled exception in main: {e}")
        import traceback

        logger.error(traceback.format_exc())
        sys.exit(1)
