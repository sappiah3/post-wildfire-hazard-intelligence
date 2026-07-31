"""
Precipitation Window Module
===========================
Aggregate ERA5 total precipitation time-series into per-basin statistics
at multiple accumulation windows (6h, 24h, 72h) for post-fire hazard modeling.

Post-fire soils have dramatically reduced infiltration capacity due to
hydrophobic layer formation; even moderate precipitation events (>10 mm/h)
can trigger debris flows within the first 2 years after burning.
"""

import logging
from typing import List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)

# Standard accumulation windows (hours)
DEFAULT_WINDOWS_H = [6, 24, 72]


# ---------------------------------------------------------------------------
# Temporal aggregation
# ---------------------------------------------------------------------------

def aggregate_precip_windows(
    precip: xr.DataArray,
    windows_h: List[int] = DEFAULT_WINDOWS_H,
) -> dict[str, xr.DataArray]:
    """
    Compute rolling precipitation totals at specified time windows.

    The rolling window uses a trailing sum so each timestep reflects the
    total accumulated precipitation over the preceding N hours.

    Parameters
    ----------
    precip : xr.DataArray
        Hourly precipitation (mm/h), dimension ``time``.
    windows_h : list of int
        Window sizes in hours.

    Returns
    -------
    dict  Keys are ``"accum_{N}h"`` strings; values are xr.DataArrays.
    """
    if "time" not in precip.dims:
        raise ValueError("ERA5 DataArray must have a 'time' dimension")

    result: dict[str, xr.DataArray] = {}
    for w in windows_h:
        rolled = precip.rolling(time=w, min_periods=1).sum()
        rolled = rolled.rename(f"precip_accum_{w}h")
        rolled.attrs["units"] = "mm"
        rolled.attrs["long_name"] = f"Precipitation accumulated over {w} hours"
        result[f"accum_{w}h"] = rolled
        logger.debug("Rolling sum computed: window=%dh", w)

    logger.info("Precip windows computed: %s", list(result.keys()))
    return result


def peak_precip_in_window(
    precip: xr.DataArray,
    window_h: int = 24,
    top_n: int = 1,
) -> xr.DataArray:
    """
    Find the maximum accumulated precipitation over any *window_h*-hour
    stretch in the time series. Useful for extracting design-storm intensity.

    Parameters
    ----------
    precip : xr.DataArray  Hourly precip (mm/h).
    window_h : int         Rolling window size (hours).
    top_n : int            Return the Nth highest rolling total.

    Returns
    -------
    xr.DataArray  Spatial grid of peak rolling-N totals (mm).
    """
    rolled = precip.rolling(time=window_h, min_periods=window_h).sum()
    # Sort descending over time and pick Nth value at each pixel
    sorted_vals = rolled.sortby("time", ascending=False)
    peak = sorted_vals.isel(time=top_n - 1).drop_vars("time")
    peak = peak.rename(f"peak_precip_{window_h}h")
    peak.attrs["units"] = "mm"
    peak.attrs["long_name"] = f"Peak {window_h}-hr accumulated precipitation (mm)"
    return peak


# ---------------------------------------------------------------------------
# Per-basin statistics
# ---------------------------------------------------------------------------

def precip_basin_stats(
    precip: xr.DataArray,
    watersheds: gpd.GeoDataFrame,
    windows_h: List[int] = DEFAULT_WINDOWS_H,
    exceedance_threshold_mm: float = 25.0,
    event_start: Optional[str] = None,
    event_end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Aggregate ERA5 precipitation to per-basin statistics across multiple
    accumulation windows. These become features in the debris flow model.

    Statistics per basin per window:
    - mean spatially averaged accumulation
    - max accumulated precip over any pixel in the basin
    - exceedance_flag  (1 if max 24h total > threshold, else 0)

    Parameters
    ----------
    precip : xr.DataArray      Hourly ERA5 precipitation (mm/h).
    watersheds : GeoDataFrame  Watershed polygons.
    windows_h : list           Accumulation windows (hours).
    exceedance_threshold_mm : float  Threshold for triggering exceedance flag.
    event_start, event_end : str, optional  Temporal slice for a specific event.

    Returns
    -------
    pd.DataFrame  One row per watershed with precip feature columns.
    """
    import rioxarray  # noqa

    if event_start or event_end:
        precip = precip.sel(time=slice(event_start, event_end))
        logger.info("Event window: %s → %s", event_start, event_end)

    windows = aggregate_precip_windows(precip, windows_h)

    # Ensure precipitation has spatial reference
    if not hasattr(precip, "rio") or precip.rio.crs is None:
        precip = precip.rio.write_crs("EPSG:4326")

    rows = []
    for huc_id, basin in watersheds.iterrows():
        geom = [basin.geometry.__geo_interface__]
        row: dict = {"huc_id": huc_id}

        for w_label, accum in windows.items():
            try:
                # Clip accumulated field to basin
                clipped = accum.rio.clip(geom, all_touched=True, from_disk=True)
                # Max over time → spatial field of per-pixel peak accumulation
                peak_spatial = clipped.max("time")
                vals = peak_spatial.values.ravel()
                vals = vals[~np.isnan(vals)]

                if len(vals) == 0:
                    row[f"mean_{w_label}_mm"] = np.nan
                    row[f"max_{w_label}_mm"]  = np.nan
                else:
                    row[f"mean_{w_label}_mm"] = float(vals.mean())
                    row[f"max_{w_label}_mm"]  = float(vals.max())
            except Exception as e:
                logger.warning("Precip clip failed for %s / %s: %s", huc_id, w_label, e)
                row[f"mean_{w_label}_mm"] = np.nan
                row[f"max_{w_label}_mm"]  = np.nan

        # Exceedance flag: did any 24h window exceed threshold?
        max_24h = row.get("max_accum_24h_mm", 0.0) or 0.0
        row["precip_exceedance_flag"] = int(max_24h >= exceedance_threshold_mm)

        rows.append(row)

    df = pd.DataFrame(rows).set_index("huc_id")
    logger.info("Precip stats: %d / %d watersheds processed", len(df), len(watersheds))
    return df
