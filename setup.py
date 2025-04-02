from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="decadalclimate",
    version="0.1.0",
    author="CLINT - DKRZ",
    author_email="fallah@dkrz.de",
    description="A toolkit for processing decadal climate prediction data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/username/decadal",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "xarray>=0.19.0",
        "netCDF4>=1.5.7",
        "pandas>=1.3.0",
        "dask>=2021.8.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.12",
            "black>=22.3",
            "flake8>=4.0",
            "isort>=5.10",
            "mypy>=0.910",
        ],
    },
    entry_points={
        "console_scripts": [
            "decadal-process=decadalclimate.cli.process:main",
            "decadal-combine=decadalclimate.cli.combine:main",
        ],
    },
)
