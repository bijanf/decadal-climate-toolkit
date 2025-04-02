#!/usr/bin/env python3
"""
Plot temperature anomalies at different forecast horizons (lead times).

This script creates visualizations comparing different ensemble members
and their forecast performance across lead times and initialization years.
This version uses netCDF4 directly instead of xarray.
"""

import argparse
import glob
import logging
import os
import sys

# Set the matplotlib backend to a non-interactive one to avoid display-related crashes
import matplotlib
import netCDF4 as nc
import numpy as np

matplotlib.use("Agg")  # Use non-interactive backend

from datetime import datetime, timedelta

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("plot_forecast_nc4")


def load_ensemble_data(data_dir, ensemble, file_pattern="tas_{}_combined.nc"):
    """
    Load ensemble data using netCDF4 directly.

    Parameters
    ----------
    data_dir : str
        Directory containing NetCDF files
    ensemble : str
        Ensemble member ID
    file_pattern : str, optional
        Pattern for NetCDF filenames, by default "tas_{}_combined.nc"

    Returns
    -------
    tuple
        Tuple containing (init_years, lead_times, lat_values, lon_values, data)
    """
    filepath = os.path.join(data_dir, file_pattern.format(ensemble))
    if not os.path.exists(filepath):
        logger.warning(f"File not found: {filepath}")
        return None

    logger.info(f"Loading data from {filepath}")
    try:
        ds = nc.Dataset(filepath, "r")

        # Get coordinates
        init_years = ds.variables["initialization"][:]
        lead_times = ds.variables["lead_time"][:]
        lat_values = ds.variables["lat"][:]
        lon_values = ds.variables["lon"][:]

        # Get temperature data
        data = ds.variables["tas"][:]

        logger.info(f"Loaded data with shape {data.shape}")

        return ds, init_years, lead_times, lat_values, lon_values, data

    except Exception as e:
        logger.error(f"Error loading {filepath}: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return None


def compute_global_average(data, lat_values):
    """
    Compute globally averaged values weighted by grid cell area (cosine of latitude).

    Parameters
    ----------
    data : numpy.ndarray
        Data array with dimensions (init, lead_time, lat, lon)
    lat_values : numpy.ndarray
        Latitude values in degrees

    Returns
    -------
    numpy.ndarray
        Globally averaged data with dimensions (init, lead_time)
    """
    logger.info("Computing global average...")

    # Create weights based on cosine of latitude
    weights = np.cos(np.deg2rad(lat_values))
    weights = weights / weights.sum()

    # Apply weights and compute global average
    n_init = data.shape[0]
    n_lead = data.shape[1]

    global_avg = np.zeros((n_init, n_lead))

    for i in range(n_init):
        for j in range(n_lead):
            # Apply latitude weighting
            weighted_slice = data[i, j, :, :] * weights[:, np.newaxis]
            global_avg[i, j] = np.nanmean(weighted_slice)

    logger.info(f"Global average computed with shape {global_avg.shape}")
    return global_avg


def plot_timeseries(global_avg, init_years, lead_times, ensemble, output_dir=None):
    """
    Plot time series of globally averaged temperature anomalies.

    Parameters
    ----------
    global_avg : numpy.ndarray
        Globally averaged data with dimensions (init, lead_time)
    init_years : numpy.ndarray
        Initialization years
    lead_times : numpy.ndarray
        Lead times (months)
    ensemble : str
        Ensemble member ID
    output_dir : str, optional
        Directory to save figures, by default None
    """
    logger.info("Creating time series plot...")

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot each initialization year
    for i, year in enumerate(init_years):
        # Convert numpy.int32 to regular Python int
        year_int = int(year)

        # Create dates for x-axis (start from November of init year)
        dates = []
        for lead in lead_times:
            # Convert numpy.int32 to regular Python int
            lead_int = int(lead)
            date = datetime(year_int, 11, 1) + timedelta(days=30 * lead_int)
            dates.append(date)

        # Plot with dates on x-axis
        ax.plot(dates, global_avg[i], marker="o", ms=5, linewidth=1.5, label=f"Init: {year_int}")

    # Format plot
    ax.set_title(f"Global Mean Temperature Anomalies - {ensemble}", fontsize=14)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Temperature Anomaly (K)", fontsize=12)
    ax.grid(True, alpha=0.3)

    # Format x-axis as dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=45)

    # Add legend
    ax.legend(loc="best", frameon=True, fontsize=10)

    # Adjust layout
    plt.tight_layout()

    # Save or show plot
    if output_dir:
        output_file = os.path.join(output_dir, f"global_timeseries_{ensemble}.png")
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        logger.info(f"Saved plot to {output_file}")
    else:
        plt.show()


def plot_lead_time(global_avg, init_years, lead_times, ensemble, output_dir=None):
    """
    Plot temperature anomalies against lead time.

    Parameters
    ----------
    global_avg : numpy.ndarray
        Globally averaged data with dimensions (init, lead_time)
    init_years : numpy.ndarray
        Initialization years
    lead_times : numpy.ndarray
        Lead times (months)
    ensemble : str
        Ensemble member ID
    output_dir : str, optional
        Directory to save figures, by default None
    """
    logger.info("Creating lead time plot...")

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot each initialization year
    for i, year in enumerate(init_years):
        # Convert numpy.int32 to regular Python int
        year_int = int(year)
        # Convert lead_times to regular Python int if needed
        lead_times_int = [int(lt) for lt in lead_times]

        ax.plot(
            lead_times_int,
            global_avg[i],
            marker="o",
            ms=5,
            linewidth=1.5,
            label=f"Init: {year_int}",
        )

    # Format plot
    ax.set_title(f"Temperature Anomalies by Lead Time - {ensemble}", fontsize=14)
    ax.set_xlabel("Lead Time (months)", fontsize=12)
    ax.set_ylabel("Temperature Anomaly (K)", fontsize=12)
    ax.grid(True, alpha=0.3)

    # Add legend
    ax.legend(loc="best", frameon=True, fontsize=10)

    # Adjust layout
    plt.tight_layout()

    # Save or show plot
    if output_dir:
        output_file = os.path.join(output_dir, f"lead_time_{ensemble}.png")
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        logger.info(f"Saved plot to {output_file}")
    else:
        plt.show()


def find_ensemble_files(data_dir, pattern="tas_*_combined.nc"):
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
    list
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
        description="Plot temperature anomalies using netCDF4 directly"
    )
    parser.add_argument(
        "--data-dir", type=str, default="output", help="Directory containing NetCDF files"
    )
    parser.add_argument(
        "--ensemble", type=str, help="Ensemble member ID (default: first one found)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="figures", help="Directory to save figures"
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Don't save figures, just display them"
    )
    parser.add_argument(
        "--max-ensembles",
        type=int,
        default=1,
        help="Maximum number of ensembles to process (default: 1, use 0 for all)",
    )
    parser.add_argument(
        "--compare", action="store_true", help="Create comparison plots for multiple ensembles"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = None if args.no_save else args.output_dir
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Find ensemble files
    if args.ensemble:
        # Use the specified ensemble
        ensembles = [args.ensemble]
    else:
        # Find all ensembles
        ensembles = find_ensemble_files(args.data_dir)
        if not ensembles:
            logger.error(f"No ensemble files found in {args.data_dir}")
            return 1

        # If max_ensembles is specified, limit the number of ensembles
        if args.max_ensembles > 0 and len(ensembles) > args.max_ensembles:
            logger.info(f"Limiting to {args.max_ensembles} ensembles out of {len(ensembles)}")
            ensembles = ensembles[: args.max_ensembles]

    logger.info(f"Processing {len(ensembles)} ensembles: {ensembles}")

    # Process each ensemble
    all_data = {}  # Dictionary to store global averages for each ensemble

    for ensemble in ensembles:
        logger.info(f"Processing ensemble: {ensemble}")

        # Load data
        result = load_ensemble_data(args.data_dir, ensemble)
        if result is None:
            logger.warning(f"Failed to load data for ensemble {ensemble}, skipping")
            continue

        ds, init_years, lead_times, lat_values, lon_values, data = result

        # Compute global average
        global_avg = compute_global_average(data, lat_values)

        # Store for comparison plots
        all_data[ensemble] = {
            "global_avg": global_avg,
            "init_years": init_years,
            "lead_times": lead_times,
            "ds": ds,  # Keep the dataset for later closing
        }

        # Create individual plots for this ensemble
        try:
            plot_timeseries(global_avg, init_years, lead_times, ensemble, output_dir)
            plot_lead_time(global_avg, init_years, lead_times, ensemble, output_dir)
            logger.info(f"Individual plots created for {ensemble}")
        except Exception as e:
            logger.error(f"Error creating plots for {ensemble}: {e}")
            import traceback

            logger.error(traceback.format_exc())

    # Create comparison plots if requested and if we have multiple ensembles
    if args.compare and len(all_data) > 1:
        try:
            logger.info("Creating comparison plots...")
            compare_lead_times(all_data, output_dir)
            compare_ensembles(all_data, output_dir)
            logger.info("Comparison plots created successfully")
        except Exception as e:
            logger.error(f"Error creating comparison plots: {e}")
            import traceback

            logger.error(traceback.format_exc())

    # Close all datasets
    for ensemble_data in all_data.values():
        ensemble_data["ds"].close()

    logger.info("All processing complete")
    return 0


def compare_lead_times(all_data, output_dir=None):
    """
    Create a comparison plot of lead times for multiple ensembles.

    Parameters
    ----------
    all_data : dict
        Dictionary containing data for multiple ensembles
    output_dir : str, optional
        Directory to save figures, by default None
    """
    logger.info("Creating lead time comparison plot...")

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Colors for different ensembles
    colors = plt.cm.tab10.colors

    # Find a common initialization year to compare
    init_years_all = set()
    for ensemble_data in all_data.values():
        init_years = [int(y) for y in ensemble_data["init_years"]]
        init_years_all.update(init_years)

    # Use the most recent common initialization year
    common_years = sorted(list(init_years_all))
    if not common_years:
        logger.error("No common initialization years found")
        return

    common_year = common_years[-1]  # Use the most recent year
    logger.info(f"Using initialization year {common_year} for comparison")

    # Plot each ensemble
    for i, (ensemble, ensemble_data) in enumerate(all_data.items()):
        global_avg = ensemble_data["global_avg"]
        init_years = [int(y) for y in ensemble_data["init_years"]]
        lead_times = [int(lt) for lt in ensemble_data["lead_times"]]

        # Check if this ensemble has the common year
        if common_year in init_years:
            year_idx = init_years.index(common_year)

            # Plot lead time vs. temperature anomaly
            ax.plot(
                lead_times,
                global_avg[year_idx],
                marker="o",
                ms=6,
                label=ensemble,
                color=colors[i % len(colors)],
                linewidth=2,
            )
        else:
            logger.warning(f"Ensemble {ensemble} does not have initialization year {common_year}")

    # Format plot
    ax.set_title(
        f"Temperature Anomalies by Lead Time - Initialization Year: {common_year}", fontsize=14
    )
    ax.set_xlabel("Lead Time (months)", fontsize=12)
    ax.set_ylabel("Temperature Anomaly (K)", fontsize=12)
    ax.grid(True, alpha=0.3)

    # Add legend
    ax.legend(loc="best", frameon=True, fontsize=10)

    # Adjust layout
    plt.tight_layout()

    # Save or show plot
    if output_dir:
        output_file = os.path.join(output_dir, f"lead_time_comparison_{common_year}.png")
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        logger.info(f"Saved comparison plot to {output_file}")
    else:
        plt.show()


def compare_ensembles(all_data, output_dir=None):
    """
    Create a comparison plot of all ensembles showing time series.

    Parameters
    ----------
    all_data : dict
        Dictionary containing data for multiple ensembles
    output_dir : str, optional
        Directory to save figures, by default None
    """
    logger.info("Creating ensemble comparison plot...")

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Colors for different ensembles
    colors = plt.cm.tab10.colors

    # Find the minimum and maximum years for the x-axis
    min_year = 3000
    max_year = 0

    # Store lines to add labels later
    lines_by_ensemble = {}

    # Plot each ensemble
    for i, (ensemble, ensemble_data) in enumerate(all_data.items()):
        color = colors[i % len(colors)]
        global_avg = ensemble_data["global_avg"]
        init_years = [int(y) for y in ensemble_data["init_years"]]
        lead_times = [int(lt) for lt in ensemble_data["lead_times"]]

        lines_by_ensemble[ensemble] = []

        # For each initialization year, create a time series
        for j, year in enumerate(init_years):
            # Create dates for x-axis (start from November of init year)
            dates = []
            for lead in lead_times:
                date = datetime(year, 11, 1) + timedelta(days=30 * lead)
                dates.append(date)

            # Convert dates to years for plotting
            time_points = [date.year + (date.month - 1) / 12 for date in dates]

            # Update min and max years for x-axis limits
            min_year = min(min_year, time_points[0])
            max_year = max(max_year, time_points[-1])

            # Plot with actual years as x-axis
            (line,) = ax.plot(
                time_points,
                global_avg[j],
                marker="o",
                ms=4,
                alpha=0.8,
                color=color,
                linewidth=1.5,
            )

            # Store the line for this ensemble
            lines_by_ensemble[ensemble].append(line)

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

    # Add grid, title, and labels
    ax.grid(True, alpha=0.3)
    ax.set_title("Temperature Anomalies for All Ensemble Members", fontsize=14)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Temperature Anomaly (K)", fontsize=12)

    # Adjust layout
    plt.tight_layout()

    # Save or show plot
    if output_dir:
        output_file = os.path.join(output_dir, "ensemble_comparison.png")
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        logger.info(f"Saved ensemble comparison plot to {output_file}")
    else:
        plt.show()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        import traceback

        logger.error(traceback.format_exc())
        sys.exit(1)
