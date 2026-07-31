"""
Ingest Module
=============
Load multi-source geospatial and atmospheric data into xarray DataArrays
and GeoDataFrames for the hazard pipeline.

Supported sources
-----------------
- Burn severity rasters  : GeoTIFF / Cloud Optimized GeoTIFF via rioxarray
- Digital Elevation Model: GeoTIFF via rioxarray
- ERA5 reanalysis        : NetCDF via xarray (total_precipitation, 2m_temp)
- Watershed boundaries   : GeoPackage / Shapefile via geopandas
"""

import logging
from pathlib import Path
from typing import Optional, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray  # noqa: F401 — registers .rio accessor on xarray
import xarray as xr

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Burn Severity (GeoTIFF / COG)
# ---------------------------------------------------------------------------

def load_burn_severity(
    path: Union[str, Path],
    target_crs: str = "EPSG:4326",
    nodata_fill: float = np.nan,
) -> xr.DataArray:
    """
    Load a burn severity or dNBR raster into an xarray DataArray.

    Parameters
    ----------
    path : str | Path
        Path to GeoTIFF or Cloud Optimized GeoTIFF.
    target_crs : str
        Output CRS (reprojected if necessary).
    nodata_fill : float
        Value to substitute for nodata pixels.

    Returns
    -------
    xr.DataArray
        2-D (y, x) array with CRS and spatial metadata attached via rioxarray.
    """
    path = Path(path)
    logger.info("Loading burn severity raster: %s", path)

    da = xr.open_dataarray(path, engine="rasterio").squeeze("band", drop=True)
    da = da.rio.write_crs(da.rio.crs or target_crs, inplace=True)

    if da.rio.crs.to_string() != target_crs:
        logger.info("Reprojecting burn severity: %s → %s", da.rio.crs, target_crs)
        da = da.rio.reproject(target_crs)

    # Replace nodata with NaN for clean arithmetic downstream
    nodata = da.rio.nodata
    if nodata is not None:
        da = da.where(da != nodata, other=nodata_fill)

    logger.info(
        "Burn severity loaded: shape=%s, CRS=%s, value range=[%.2f, %.2f]",
        da.shape, da.rio.crs, float(da.min()), float(da.max()),
    )
    return da.rename("dnbr")


# ---------------------------------------------------------------------------
# Digital Elevation Model (GeoTIFF / COG)
# ---------------------------------------------------------------------------

def load_dem(
    path: Union[str, Path],
    target_crs: str = "EPSG:4326",
    resolution_m: Optional[int] = None,
) -> xr.DataArray:
    """
    Load a Digital Elevation Model raster.

    Parameters
    ----------
    path : str | Path
        Path to GeoTIFF DEM.
    target_crs : str
        Output CRS (reprojected if necessary).
    resolution_m : int, optional
        Resample to this resolution in meters after reprojection.

    Returns
    -------
    xr.DataArray
        2-D elevation array (meters) with rioxarray spatial metadata.
    """
    path = Path(path)
    logger.info("Loading DEM: %s", path)

    da = xr.open_dataarray(path, engine="rasterio").squeeze("band", drop=True)
    da = da.rio.write_crs(da.rio.crs or "EPSG:5070", inplace=True)

    if da.rio.crs.to_string() != target_crs:
        logger.info("Reprojecting DEM: %s → %s", da.rio.crs, target_crs)
        da = da.rio.reproject(target_crs)

    if resolution_m:
        from rasterio.enums import Resampling
        da = da.rio.reproject(
            da.rio.crs,
            resolution=resolution_m / 111_000,  # approx degree/m at mid-latitudes
            resampling=Resampling.bilinear,
        )

    nodata = da.rio.nodata
    if nodata is not None:
        da = da.where(da != nodata)

    logger.info(
        "DEM loaded: shape=%s, elevation range=[%.1f, %.1f] m",
        da.shape, float(da.min()), float(da.max()),
    )
    return da.rename("elevation")


# ---------------------------------------------------------------------------
# ERA5 Atmospheric Reanalysis (NetCDF)
# ---------------------------------------------------------------------------

def load_era5_precip(
    path: Union[str, Path],
    variable: str = "tp",
    start: Optional[str] = None,
    end: Optional[str] = None,
    bbox: Optional[tuple] = None,
) -> xr.DataArray:
    """
    Load ERA5 total precipitation from a NetCDF file.

    ERA5 stores precipitation in meters per hour (accumulated). This loader
    converts to mm/h and clips to the requested temporal and spatial window.

    Parameters
    ----------
    path : str | Path
        NetCDF file downloaded from the Copernicus Climate Data Store.
    variable : str
        ERA5 variable name (default ``"tp"`` for total precipitation).
    start : str, optional
        ISO 8601 start datetime for temporal slice, e.g. ``"2019-01-01"``.
    end : str, optional
        ISO 8601 end datetime for temporal slice.
    bbox : tuple, optional
        (min_lon, min_lat, max_lon, max_lat) spatial clip.

    Returns
    -------
    xr.DataArray
        3-D (time, latitude, longitude) precipitation array in mm/h.
    """
    path = Path(path)
    logger.info("Loading ERA5 precipitation: %s", path)

    ds = xr.open_dataset(path, engine="netcdf4", chunks={"time": 100})

    if variable not in ds:
        available = list(ds.data_vars)
        raise KeyError(f"Variable '{variable}' not found. Available: {available}")

    da = ds[variable]

    # Temporal slice
    if start or end:
        da = da.sel(time=slice(start, end))
        logger.info("Temporal slice: %s → %s (%d steps)", start, end, da.sizes["time"])

    # Spatial clip
    if bbox:
        min_lon, min_lat, max_lon, max_lat = bbox
        da = da.sel(
            longitude=slice(min_lon, max_lon),
            latitude=slice(max_lat, min_lat),  # ERA5 lat is descending
        )

    # ERA5 tp is in metres; convert to mm and handle accumulated values
    da = da * 1000.0  # m → mm
    da = da.rename("precip_mm_h")
    da.attrs["units"] = "mm/h"
    da.attrs["long_name"] = "Total Precipitation (mm/h, ERA5)"

    logger.info(
        "ERA5 loaded: time steps=%d, spatial shape=(%d, %d), range=[%.3f, %.3f] mm/h",
        da.sizes["time"],
        da.sizes.get("latitude", 0),
        da.sizes.get("longitude", 0),
        float(da.min()),
        float(da.max()),
    )
    return da


# ---------------------------------------------------------------------------
# Watershed boundaries (GeoPackage / Shapefile)
# ---------------------------------------------------------------------------

def load_watersheds(
    path: Union[str, Path],
    id_col: str = "huc12",
    name_col: str = "name",
    target_crs: str = "EPSG:4326",
) -> gpd.GeoDataFrame:
    """
    Load HUC-12 (or equivalent) watershed boundary polygons.

    Parameters
    ----------
    path : str | Path
        GeoPackage, Shapefile, or GeoJSON.
    id_col : str
        Column to use as the unique watershed identifier.
    name_col : str
        Column containing human-readable watershed name.
    target_crs : str
        Reproject to this CRS if needed.

    Returns
    -------
    GeoDataFrame  indexed by *id_col*.
    """
    logger.info("Loading watershed boundaries: %s", path)
    gdf = gpd.read_file(path)

    if gdf.crs and gdf.crs.to_string() != target_crs:
        gdf = gdf.to_crs(target_crs)

    if id_col not in gdf.columns:
        raise KeyError(f"ID column '{id_col}' not found. Columns: {list(gdf.columns)}")

    gdf = gdf.set_index(id_col)
    logger.info("Watersheds loaded: %d basins, CRS=%s", len(gdf), gdf.crs)
    return gdf
