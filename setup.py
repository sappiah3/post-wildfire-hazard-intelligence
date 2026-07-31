from setuptools import setup, find_packages

setup(
    name="post-wildfire-hazard-intelligence",
    version="1.0.0",
    author="Sam Appiah",
    description=(
        "Multi-hazard cascade pipeline: wildfire burn severity → "
        "debris flow probability → basin hazard index"
    ),
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "xarray>=2024.1", "rioxarray>=0.15", "rasterio>=1.3",
        "geopandas>=0.14", "shapely>=2.0", "pandas>=2.0", "numpy>=1.26",
        "scikit-learn>=1.4", "pysheds>=0.4", "netCDF4>=1.6",
        "pyarrow>=14.0", "jinja2>=3.1", "pyyaml>=6.0",
    ],
    entry_points={"console_scripts": ["hazard-pipeline=pipeline.run:main"]},
)
