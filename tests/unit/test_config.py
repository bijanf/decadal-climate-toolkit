"""
Unit tests for the configuration utilities.
"""

import tempfile
from pathlib import Path

import pytest
import yaml

from decadalclimate.utils.config import load_config, save_config


def test_save_and_load_config():
    """Test saving and loading configuration."""
    # Create a test configuration
    config = {
        "test": {
            "value1": 42,
            "value2": "test",
        },
        "paths": {
            "input": "/path/to/input",
            "output": "/path/to/output",
        },
    }

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a temporary file
        config_file = Path(temp_dir) / "config.yaml"

        # Save the configuration
        save_config(config, config_file)

        # Check that the file exists
        assert config_file.exists()

        # Load the configuration
        loaded_config = load_config(config_file)

        # Check that the configuration was loaded correctly
        assert loaded_config == config

        # Check direct YAML loading matches
        with open(config_file, "r") as f:
            yaml_loaded = yaml.safe_load(f)
        assert yaml_loaded == config


def test_load_nonexistent_config():
    """Test loading a nonexistent configuration file."""
    # Create a temporary file path that doesn't exist
    with tempfile.TemporaryDirectory() as temp_dir:
        config_file = Path(temp_dir) / "nonexistent.yaml"

        # Try to load the configuration (should raise FileNotFoundError)
        with pytest.raises(FileNotFoundError):
            load_config(config_file)


def test_nested_config_access():
    """Test accessing nested configuration values."""
    # Create a test configuration
    config = {"nested": {"deeply": {"buried": {"value": 42}}}}

    # Access the nested value
    assert config.get("nested", {}).get("deeply", {}).get("buried", {}).get("value") == 42

    # Access a nonexistent nested value
    assert config.get("nonexistent", {}).get("path", {}).get("to", {}).get("value") == {}
