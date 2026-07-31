"""
Burn Severity Module
====================
Compute dNBR (differenced Normalized Burn Ratio) from Landsat bands,
classify burn severity per the USGS MTBS scheme, and derive per-basin
summary statistics for downstream hazard modeling.

Reference: Key & Benson (2006) USGS FIREMON Landscape Assessment.
"""

import logging
from typing import Dict, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)

# USGS MTBS dNBR classification thresholds (scaled dNBR × 1000)
DNBR_THRESHOLDS: Dict[str, Tuple[float, float]] = {
    "enhanced_regrowth_high": (-500.0, -251.0),
    "enhanced_regrowth_low":  (-250.0, -101.0),
    "unburned":               (-100.0,   99.0),
    "low":                    ( 100.0,  269.0),
    "moderate_low":           ( 270.0,  439.0),
    "moderate_high":          ( 440.0,  659.0),
    "high":                   ( 660.0, 1300.0),
}

# Classes that contribute to "high-severity fraction"
HIGH_SEVERITY_CLASSES = {"moderate_high", "high"}

# Integer codes for each severity class (for raster encoding)
CLASS_CODES: Dict[str, int] = {
    "enhanced_regrowth_high": -2,
    "enhanced_regrowth_low":  -1,
    "unburned":                0,
    "low":                     1,
    "moderate_low":            2,
    "moderate_high":           3,
    "high":                    4,
}


# ---------------------------------------------------------------------------
# dNBR from Landsat reflectance bands
# ---------------------------------------------------------------------------

def compute_nbr(nir: xr.DataArray, swir: xr.DataArray) -> xr.DataArray:
    """
    Compute Normalized Burn Ratio: NBR = (NIR - SWIR2) / (NIR + SWIR2).

    Parameters
    ----------
    nir  : xr.DataArray  Landsat Band 5 (Landsat 8) or Band 4 (Landsat 7) TOA/SR
    swir : xr.DataArray  Landsat Band 7 (SWIR-2) TOA/SR

    Returns
    -------
    xr.DataArray  NBR in [-1, 1]
    """
    nbr = (nir - swir) / (nir + swir + 1e-10)
    nbr = nbr.clip(-1.0, 1.0)
    return nbr.rename("nbr")


def compute_dnbr(
    pre_nir: xr.DataArray,
    pre_swir: xr.DataArray,
    post_nir: xr.DataArray,
    post_swir: xr.DataArray,
    scale: float = 1000.0,
) -> xr.DataArray:
    """
    Compute differenced Normalized Burn Ratio (dNBR).

    dNBR = (pre-fire NBR − post-fire NBR) × scale

    Values are multiplied by 1000 to match USGS MTBS reporting convention.

    Parameters
    ----------
    pre_nir, pre_swir   : xr.DataArray  Pre-fire Landsat bands
    post_nir, post_swir : xr.DataArray  Post-fire Landsat bands
    scale : float  Multiplier (default 1000 for MTBS convention)

    Returns
    -------
    xr.DataArray  dNBR (unitless, scaled)
    """
    nbr_pre  = compute_nbr(pre_nir,  pre_swir)
    nbr_post = compute_nbr(post_nir, post_swir)
    dnbr = (nbr_pre - nbr_post) * scale
    dnbr.attrs["long_name"] = "Differenced Normalized Burn Ratio (× 1000)"
    dnbr.attrs["reference"] = "Key & Benson (2006) USGS FIREMON"
    logger.info(
        "dNBR computed: range=[%.1f, %.1f], mean=%.1f",
        float(dnbr.min()), float(dnbr.max()), float(dnbr.mean()),
    )
    return dnbr.rename("dnbr")


# ---------------------------------------------------------------------------
# Severity classification
# ---------------------------------------------------------------------------

def classify_burn_severity(
    dnbr: xr.DataArray,
    thresholds: Optional[Dict[str, Tuple[float, float]]] = None,
) -> xr.DataArray:
    """
    Classify dNBR into USGS MTBS severity categories.

    Parameters
    ----------
    dnbr : xr.DataArray  dNBR raster (scaled × 1000)
    thresholds : dict, optional  Override default DNBR_THRESHOLDS

    Returns
    -------
    xr.DataArray  Integer-coded severity class raster.
                  See CLASS_CODES for the mapping.
    """
    thresholds = thresholds or DNBR_THRESHOLDS
    classified = xr.full_like(dnbr, fill_value=0, dtype=np.int8)

    for class_name, (lo, hi) in thresholds.items():
        mask = (dnbr >= lo) & (dnbr < hi)
        classified = classified.where(~mask, other=CLASS_CODES[class_name])

    classified.attrs["long_name"] = "Burn Severity Class (USGS MTBS)"
    classified.attrs["class_codes"] = str(CLASS_CODES)
    return classified.rename("burn_severity_class")


# ---------------------------------------------------------------------------
# Per-basin summary statistics
# ---------------------------------------------------------------------------

def burn_severity_stats(
    dnbr: xr.DataArray,
    watersheds: gpd.GeoDataFrame,
    classified: Optional[xr.DataArray] = None,
) -> pd.DataFrame:
    """
    Compute per-basin burn severity statistics for use as model features.

    Statistics computed per HUC-12 watershed:
    - mean_dnbr, std_dnbr, max_dnbr
    - pct_high_severity     : % of burned pixels in moderate-high or high class
    - pct_unburned          : % of pixels classified as unburned
    - burn_area_km2         : total burned area above low threshold

    Parameters
    ----------
    dnbr : xr.DataArray      Scaled dNBR raster
    watersheds : GeoDataFrame  Watershed boundary polygons (indexed by HUC ID)
    classified : xr.DataArray, optional  Pre-computed severity class raster

    Returns
    -------
    pd.DataFrame  One row per watershed, columns = feature names.
    """
    import rioxarray  # noqa

    if classified is None:
        classified = classify_burn_severity(dnbr)

    # Pixel area in km²
    res_deg = abs(float(dnbr.rio.resolution()[0]))
    approx_km2_per_pixel = (res_deg * 111.0) ** 2  # rough at mid-latitudes

    rows = []
    for huc_id, basin in watersheds.iterrows():
        try:
            # Clip rasters to basin bounds
            geom = [basin.geometry.__geo_interface__]
            dnbr_clip = dnbr.rio.clip(geom, all_touched=True, from_disk=True)
            cls_clip  = classified.rio.clip(geom, all_touched=True, from_disk=True)
        except Exception as e:
            logger.warning("Clip failed for %s: %s", huc_id, e)
            continue

        dnbr_vals = dnbr_clip.values.ravel()
        cls_vals  = cls_clip.values.ravel()
        valid     = ~np.isnan(dnbr_vals)

        if valid.sum() == 0:
            logger.debug("No valid pixels for watershed %s — skipping", huc_id)
            continue

        dnbr_valid = dnbr_vals[valid]
        cls_valid  = cls_vals[valid]
        n = len(cls_valid)

        high_mask     = np.isin(cls_valid, [CLASS_CODES["moderate_high"], CLASS_CODES["high"]])
        unburned_mask = cls_valid <= CLASS_CODES["unburned"]
        burned_mask   = cls_valid >= CLASS_CODES["low"]

        rows.append({
            "huc_id":           huc_id,
            "mean_dnbr":        float(dnbr_valid.mean()),
            "std_dnbr":         float(dnbr_valid.std()),
            "max_dnbr":         float(dnbr_valid.max()),
            "pct_high_severity": float(high_mask.sum() / n * 100),
            "pct_unburned":      float(unburned_mask.sum() / n * 100),
            "burn_area_km2":     float(burned_mask.sum() * approx_km2_per_pixel),
        })

    df = pd.DataFrame(rows).set_index("huc_id")
    logger.info("Burn severity stats: %d / %d watersheds processed", len(df), len(watersheds))
    return df
