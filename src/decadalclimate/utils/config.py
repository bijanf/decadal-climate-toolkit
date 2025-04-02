"""
Configuration utilities for the DecadalClimate package.
"""

import os
from typing import Any, Dict, Optional

import yaml


def get_config_dir() -> str:
    """
    Get the configuration directory.

    Returns
    -------
    str
        Path to the configuration directory
    """
    # First check if config directory is in current directory
    if os.path.isdir("config"):
        return "config"

    # Check if it's in the parent directory (repository root)
    if os.path.isdir("../config"):
        return "../config"

    # Check in the package directory
    package_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if os.path.isdir(os.path.join(package_dir, "config")):
        return os.path.join(package_dir, "config")

    # Default to user config directory
    user_config = os.path.expanduser("~/.config/decadalclimate")
    if not os.path.isdir(user_config):
        os.makedirs(user_config, exist_ok=True)
    return user_config


def load_config(config_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Parameters
    ----------
    config_file : str, optional
        Path to the configuration file, by default None

    Returns
    -------
    Dict[str, Any]
        Configuration dictionary
    """
    if config_file is None:
        # Load and merge both configuration files
        config_dir = get_config_dir()
        paths_file = os.path.join(config_dir, "paths.yaml")
        processing_file = os.path.join(config_dir, "processing.yaml")

        config = {}

        # Load paths configuration
        if os.path.isfile(paths_file):
            with open(paths_file, "r") as f:
                paths_config = yaml.safe_load(f)
                if paths_config:
                    config.update(paths_config)

        # Load processing configuration
        if os.path.isfile(processing_file):
            with open(processing_file, "r") as f:
                processing_config = yaml.safe_load(f)
                if processing_config:
                    config.update(processing_config)
    else:
        # Load specified configuration file
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)

    return config or {}


def save_config(config: Dict[str, Any], config_file: str) -> None:
    """
    Save configuration to YAML file.

    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary
    config_file : str
        Path to the configuration file
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(config_file)), exist_ok=True)

    # Save configuration
    with open(config_file, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
