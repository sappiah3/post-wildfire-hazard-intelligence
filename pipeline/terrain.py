"""
Terrain Analysis Module
=======================
Derive slope, aspect, terrain ruggedness index (TRI), and flow
accumulation from a DEM, then aggregate to per-basin statistics
for use as hazard model features.

All raster ops use xarray + rioxarray for consistency with the rest
of the pipeline. Flow accumulation uses pysheds for a proper
D8 routing algorithm.
"""

import logging
from typing import List, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Slope (degrees)
# ---------------------------------------------------------------------------

def compute_slope(dem: xr.DataArray) -> xr.DataArray:
    """
    Compute slope in degrees from a projected DEM using finite differences.

    For accurate results the DEM should be in a metric CRS (e.g. UTM, CONUS
    Albers). If passed in geographic degrees, resolution is converted using
    a mid-latitude approximation.

    Parameters
    ----------
    dem : xr.DataArray  Elevation raster (metres).

    Returns
    -------
    xr.DataArray  Slope in degrees, same shape as *dem*.
    """
    z = dem.values.astype(np.float32)
    res = _pixel_size_meters(dem)
    dx = res["x"]
    dy = res["y"]

    # Sobel-style central difference gradient
    dz_dx = np.gradient(z, dx, axis=1)
    dz_dy = np.gradient(z, dy, axis=0)

    slope_rad = np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2))
    slope_deg = np.degrees(slope_rad)

    slope = xr.DataArray(
        slope_deg,
        coords=dem.coords,
        dims=dem.dims,
        name="slope_deg",
        attrs={"units": "degrees", "long_name": "Terrain Slope"},
    )
    slope.rio.write_crs(dem.rio.crs, inplace=True)
    logger.info("Slope computed: range=[%.1f°, %.1f°], mean=%.1f°",
                float(slope.min()), float(slope.max()), float(slope.mean()))
    return slope


# ---------------------------------------------------------------------------
# Aspect (degrees clockwise from north)
# ---------------------------------------------------------------------------

def compute_aspect(dem: xr.DataArray) -> xr.DataArray:
    """
    Compute aspect (azimuth of the steepest downhill direction) in degrees
    clockwise from north (0 = north, 90 = east, 180 = south, 270 = west).
    Flat areas are assigned -1.

    Parameters
    ----------
    dem : xr.DataArray  Elevation raster.

    Returns
    -------
    xr.DataArray  Aspect in degrees (0–360) or -1 for flat.
    """
    z = dem.values.astype(np.float32)
    res = _pixel_size_meters(dem)

    dz_dx = np.gradient(z, res["x"], axis=1)
    dz_dy = np.gradient(z, res["y"], axis=0)

    # atan2 gives azimuth from east; convert to clockwise-from-north
    aspect_rad = np.arctan2(-dz_dy, dz_dx)
    aspect_deg = (90.0 - np.degrees(aspect_rad)) % 360.0

    # Flat areas (no gradient)
    flat = (np.abs(dz_dx) < 1e-6) & (np.abs(dz_dy) < 1e-6)
    aspect_deg[flat] = -1.0

    aspect = xr.DataArray(
        aspect_deg,
        coords=dem.coords,
        dims=dem.dims,
        name="aspect_deg",
        attrs={"units": "degrees CW from north", "long_name": "Terrain Aspect"},
    )
    aspect.rio.write_crs(dem.rio.crs, inplace=True)
    return aspect


# ---------------------------------------------------------------------------
# Terrain Ruggedness Index (TRI)
# ---------------------------------------------------------------------------

def compute_tri(dem: xr.DataArray, window: int = 3) -> xr.DataArray:
    """
    Compute the Terrain Ruggedness Index (Riley et al. 1999):
    TRI = sqrt(sum((z_neighbor - z_center)^2)) over a NxN focal window.

    Parameters
    ----------
    dem : xr.DataArray  Elevation raster.
    window : int  Focal window size (must be odd). Default 3 (3×3 kernel).

    Returns
    -------
    xr.DataArray  TRI values (metres), same shape as *dem*.
    """
    from scipy.ndimage import generic_filter

    if window % 2 == 0:
        raise ValueError(f"Window must be odd; got {window}")

    z = dem.values.astype(np.float32)

    def _tri_kernel(values: np.ndarray) -> float:
        center = values[len(values) // 2]
        return float(np.sqrt(np.sum((values - center) ** 2)))

    tri_vals = generic_filter(z, _tri_kernel, size=window, mode="reflect")

    tri = xr.DataArray(
        tri_vals,
        coords=dem.coords,
        dims=dem.dims,
        name="tri",
        attrs={"units": "m", "long_name": "Terrain Ruggedness Index"},
    )
    tri.rio.write_crs(dem.rio.crs, inplace=True)
    logger.info("TRI computed: range=[%.2f, %.2f] m", float(tri.min()), float(tri.max()))
    return tri


# ---------------------------------------------------------------------------
# Flow Accumulation (D8 via pysheds)
# ---------------------------------------------------------------------------

def compute_flow_accumulation(dem: xr.DataArray) -> xr.DataArray:
    """
    Compute D8 flow accumulation using pysheds.

    Flow accumulation represents the number of upstream cells draining
    to each cell — a proxy for contributing area and stream power.

    Parameters
    ----------
    dem : xr.DataArray  Hydrologically conditioned or raw DEM.

    Returns
    -------
    xr.DataArray  Flow accumulation (cells), same shape as *dem*.
    """
    try:
        from pysheds.grid import Grid
        import tempfile, rasterio
        from rasterio.transform import from_bounds
    except ImportError:
        logger.warning("pysheds not installed; returning uniform flow accumulation")
        return xr.full_like(dem, fill_value=1.0, dtype=np.float32).rename("flow_accum")

    # pysheds works with file paths; write DEM to a temp file
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        dem.rio.to_raster(tmp_path)
        grid = Grid.from_raster(tmp_path)
        dem_grid = grid.read_raster(tmp_path)

        # Fill pits and depressions
        pit_filled = grid.fill_pits(dem_grid)
        flooded    = grid.fill_depressions(pit_filled)
        inflated   = grid.resolve_flats(flooded)

        # D8 flow direction
        fdir = grid.flowdir(inflated)
        accum = grid.accumulation(fdir)
    finally:
        os.unlink(tmp_path)

    flow = xr.DataArray(
        accum.astype(np.float32),
        coords=dem.coords,
        dims=dem.dims,
        name="flow_accum",
        attrs={"units": "cells", "long_name": "D8 Flow Accumulation"},
    )
    flow.rio.write_crs(dem.rio.crs, inplace=True)
    logger.info("Flow accumulation computed: max=%d cells", int(flow.max()))
    return flow


# ---------------------------------------------------------------------------
# Per-basin terrain statistics
# ---------------------------------------------------------------------------

def terrain_stats(
    slope: xr.DataArray,
    dem: xr.DataArray,
    watersheds: gpd.GeoDataFrame,
    slope_percentiles: List[int] = (50, 75, 90),
    tri: Optional[xr.DataArray] = None,
    flow_accum: Optional[xr.DataArray] = None,
) -> pd.DataFrame:
    """
    Aggregate terrain rasters to per-basin summary statistics.

    Parameters
    ----------
    slope : xr.DataArray      Slope in degrees.
    dem   : xr.DataArray      Elevation in metres.
    watersheds : GeoDataFrame  Watershed polygons.
    slope_percentiles : list   Percentile values for slope distribution.
    tri : xr.DataArray, optional  Pre-computed TRI.
    flow_accum : xr.DataArray, optional  Pre-computed flow accumulation.

    Returns
    -------
    pd.DataFrame  One row per watershed.
    """
    import rioxarray  # noqa

    rows = []
    for huc_id, basin in watersheds.iterrows():
        geom = [basin.geometry.__geo_interface__]
        try:
            slope_clip = slope.rio.clip(geom, all_touched=True, from_disk=True)
            dem_clip   = dem.rio.clip(geom, all_touched=True, from_disk=True)
        except Exception as e:
            logger.warning("Clip failed for %s: %s", huc_id, e)
            continue

        sv = slope_clip.values.ravel()
        sv = sv[~np.isnan(sv)]
        ev = dem_clip.values.ravel()
        ev = ev[~np.isnan(ev)]

        if len(sv) == 0:
            continue

        row: dict = {"huc_id": huc_id}
        for p in slope_percentiles:
            row[f"slope_p{p}"] = float(np.percentile(sv, p))
        row["slope_mean"]    = float(sv.mean())
        row["elevation_mean"] = float(ev.mean()) if len(ev) else np.nan
        row["elevation_range"] = float(ev.max() - ev.min()) if len(ev) else np.nan

        if tri is not None:
            try:
                tri_clip = tri.rio.clip(geom, all_touched=True, from_disk=True)
                tv = tri_clip.values.ravel()
                tv = tv[~np.isnan(tv)]
                row["tri_mean"] = float(tv.mean()) if len(tv) else np.nan
            except Exception:
                row["tri_mean"] = np.nan

        if flow_accum is not None:
            try:
                fa_clip = flow_accum.rio.clip(geom, all_touched=True, from_disk=True)
                fv = fa_clip.values.ravel()
                fv = fv[~np.isnan(fv)]
                row["max_flow_accum"] = float(fv.max()) if len(fv) else np.nan
            except Exception:
                row["max_flow_accum"] = np.nan

        rows.append(row)

    df = pd.DataFrame(rows).set_index("huc_id")
    logger.info("Terrain stats: %d / %d watersheds processed", len(df), len(watersheds))
    return df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pixel_size_meters(da: xr.DataArray) -> dict:
    """Return approximate pixel size in meters (handles geographic CRS)."""
    res_x, res_y = da.rio.resolution()
    crs = da.rio.crs
    if crs and crs.is_geographic:
        # Approximate degrees → meters at mid-latitude
        mid_lat = float(da.coords.get("y", da.coords.get("latitude", [0])).mean())
        m_per_deg_lon = 111_320 * np.cos(np.radians(mid_lat))
        m_per_deg_lat = 110_540
        return {"x": abs(res_x) * m_per_deg_lon, "y": abs(res_y) * m_per_deg_lat}
    return {"x": abs(res_x), "y": abs(res_y)}
