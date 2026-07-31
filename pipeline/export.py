"""
Export Module
=============
Serialize hazard pipeline outputs to cloud-native geospatial formats and
a stakeholder-facing HTML intelligence report.

Formats
-------
- Cloud Optimized GeoTIFF (COG)   — raster hazard surface for GIS/dashboard use
- GeoParquet                       — basin index table, columnar and cloud-queryable
- HTML Report                      — business-readable intelligence product
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from jinja2 import Environment, BaseLoader
from rasterio.enums import Resampling
from rasterio.shutil import copy as rio_copy

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cloud Optimized GeoTIFF
# ---------------------------------------------------------------------------

def to_cog(
    data: xr.DataArray,
    output_path: str,
    compress: str = "deflate",
    blocksize: int = 512,
    nodata: float = -9999.0,
    dtype: str = "float32",
) -> None:
    """
    Write an xarray DataArray to a Cloud Optimized GeoTIFF.

    COGs use internal tiling and overview levels so that clients can
    efficiently fetch partial extents without downloading the full file —
    essential for cloud-hosted hazard rasters.

    Parameters
    ----------
    data : xr.DataArray      2-D spatial array (must have rioxarray CRS).
    output_path : str        Destination path.
    compress : str           Compression codec (default: ``"deflate"``).
    blocksize : int          Tile size in pixels (default: 512).
    nodata : float           NoData value written to raster metadata.
    dtype : str              Output data type (default: ``"float32"``).
    """
    import tempfile, os

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    arr = data.values.astype(dtype)
    arr = np.where(np.isnan(arr), nodata, arr)

    crs = data.rio.crs
    transform = data.rio.transform()

    # Write to a temp GeoTIFF first, then convert to COG
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        with rasterio.open(
            tmp_path, "w",
            driver="GTiff",
            height=arr.shape[0],
            width=arr.shape[1],
            count=1,
            dtype=dtype,
            crs=crs,
            transform=transform,
            nodata=nodata,
        ) as src:
            src.write(arr, 1)
            # Build internal overviews
            src.build_overviews([2, 4, 8, 16, 32], Resampling.average)
            src.update_tags(ns="rio_overview", resampling="average")

        # Re-open and copy to COG layout
        with rasterio.open(tmp_path) as src:
            rio_copy(
                src, str(out),
                copy_src_overviews=True,
                driver="GTiff",
                compress=compress,
                tiled=True,
                blockxsize=blocksize,
                blockysize=blocksize,
            )
    finally:
        os.unlink(tmp_path)

    logger.info("COG written: %s (%.1f KB)", out, out.stat().st_size / 1024)


# ---------------------------------------------------------------------------
# GeoParquet
# ---------------------------------------------------------------------------

def to_geoparquet(
    gdf: gpd.GeoDataFrame,
    output_path: str,
    row_group_size: int = 50_000,
    crs: str = "EPSG:4326",
) -> None:
    """
    Write a GeoDataFrame to GeoParquet format.

    GeoParquet is a columnar format that enables efficient spatial queries
    and is natively readable by DuckDB, BigQuery, and modern cloud pipelines.

    Parameters
    ----------
    gdf : GeoDataFrame       Basin hazard index table with geometry.
    output_path : str        Destination .parquet file path.
    row_group_size : int     Row group size for partitioning.
    crs : str                Ensure output CRS (reproject if needed).
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if gdf.crs and gdf.crs.to_string() != crs:
        gdf = gdf.to_crs(crs)

    # Convert any non-serializable types
    for col in gdf.select_dtypes(include=["object"]).columns:
        gdf[col] = gdf[col].astype(str)

    gdf.to_parquet(str(out), row_group_size=row_group_size, index=True)
    logger.info("GeoParquet written: %s (%d rows, %.1f KB)",
                out, len(gdf), out.stat().st_size / 1024)


# ---------------------------------------------------------------------------
# HTML Intelligence Report
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Post-Wildfire Hazard Intelligence Report — {{ event_name }}</title>
<style>
  body { font-family: 'Segoe UI', Arial, sans-serif; margin: 0; background: #f7f9fc; color: #2c3e50; }
  header { background: #1a2533; color: white; padding: 24px 40px; }
  header h1 { margin: 0; font-size: 22px; }
  header p  { margin: 4px 0 0; color: #aab7c4; font-size: 13px; }
  main { max-width: 1100px; margin: 32px auto; padding: 0 24px; }
  h2 { color: #1a5276; border-bottom: 2px solid #d0dce8; padding-bottom: 6px; }
  .kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin: 24px 0; }
  .kpi-card { background: white; border-radius: 8px; padding: 18px 22px;
              box-shadow: 0 2px 8px rgba(0,0,0,0.08); text-align: center; }
  .kpi-card .label { font-size: 11px; text-transform: uppercase; color: #7f8c8d; letter-spacing: 0.5px; }
  .kpi-card .value { font-size: 28px; font-weight: 700; margin-top: 6px; }
  .critical { color: #c0392b; } .high { color: #d35400; }
  .moderate { color: #d4ac0d; } .low { color: #27ae60; }
  table { width: 100%; border-collapse: collapse; background: white;
          border-radius: 8px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
  th { background: #2471a3; color: white; padding: 12px 16px; text-align: left; font-size: 12px; text-transform: uppercase; }
  td { padding: 10px 16px; border-bottom: 1px solid #eaecef; font-size: 13px; }
  tr:last-child td { border-bottom: none; }
  tr:nth-child(even) { background: #f8fafc; }
  .tier-critical { background: #fdecea; font-weight: 700; color: #c0392b; }
  .tier-high     { background: #fef3e7; font-weight: 700; color: #d35400; }
  .tier-moderate { background: #fefde7; color: #7d6608; }
  .tier-low      { background: #eafaf1; color: #1e8449; }
  .basin-table   { margin-top: 32px; }
  .action-cell { font-size: 11px; color: #566573; font-style: italic; }
  footer { text-align: center; padding: 32px; color: #95a5a6; font-size: 11px; }
</style>
</head>
<body>
<header>
  <h1>Post-Wildfire Hazard Intelligence Report</h1>
  <p>{{ event_name }} &nbsp;|&nbsp; Generated {{ generated_at }} &nbsp;|&nbsp; post-wildfire-hazard-intelligence v1.0</p>
</header>
<main>

<h2>Event Summary</h2>
<div class="kpi-grid">
  <div class="kpi-card">
    <div class="label">Total Basins Assessed</div>
    <div class="value">{{ total_basins }}</div>
  </div>
  <div class="kpi-card">
    <div class="label">Critical Risk</div>
    <div class="value critical">{{ critical_count }}</div>
  </div>
  <div class="kpi-card">
    <div class="label">High Risk</div>
    <div class="value high">{{ high_count }}</div>
  </div>
  <div class="kpi-card">
    <div class="label">Burn Area (km²)</div>
    <div class="value">{{ total_burn_area }}</div>
  </div>
</div>

<h2>Risk Tier Summary</h2>
<table>
  <tr>
    <th>Risk Tier</th>
    <th>Basins</th>
    <th>Burn Area (km²)</th>
    <th>Mean Hazard Score</th>
    <th>Mean Debris Flow Prob</th>
    <th>Recommended Action</th>
  </tr>
  {% for row in summary_rows %}
  <tr>
    <td class="tier-{{ row.risk_tier | lower }}">{{ row.risk_tier }}</td>
    <td>{{ row.basin_count }}</td>
    <td>{{ row.burn_area_km2 }}</td>
    <td>{{ row.mean_hazard_score }}</td>
    <td>{{ "%.3f" | format(row.mean_debris_flow_prob) }}</td>
    <td class="action-cell">{{ row.recommended_action }}</td>
  </tr>
  {% endfor %}
</table>

<h2 class="basin-table">Basin-Level Hazard Index</h2>
<table>
  <tr>
    <th>Basin ID</th>
    <th>Risk Tier</th>
    <th>Hazard Score</th>
    <th>Debris Flow Prob</th>
    <th>% High Severity</th>
    <th>Slope P90 (°)</th>
    <th>Max 24h Precip (mm)</th>
    <th>Burn Area (km²)</th>
  </tr>
  {% for row in basin_rows %}
  <tr>
    <td><code>{{ row.huc_id }}</code></td>
    <td class="tier-{{ row.risk_tier | lower }}">{{ row.risk_tier }}</td>
    <td><strong>{{ row.hazard_score }}</strong></td>
    <td>{{ "%.3f" | format(row.debris_flow_prob) }}</td>
    <td>{{ "%.1f" | format(row.pct_high_severity) }}%</td>
    <td>{{ "%.1f" | format(row.slope_p90) if row.slope_p90 else "—" }}°</td>
    <td>{{ "%.1f" | format(row.max_accum_24h_mm) if row.max_accum_24h_mm else "—" }}</td>
    <td>{{ "%.1f" | format(row.burn_area_km2) if row.burn_area_km2 else "—" }}</td>
  </tr>
  {% endfor %}
</table>

</main>
<footer>
  Generated by post-wildfire-hazard-intelligence &nbsp;|&nbsp;
  Model outputs are probabilistic estimates. Do not use as sole basis for life-safety decisions.
</footer>
</body>
</html>
"""


def to_html_report(
    hazard_df: pd.DataFrame,
    insurance_summary: pd.DataFrame,
    output_path: str,
    event_name: str = "Wildfire Event",
) -> None:
    """
    Render an HTML intelligence report for non-scientist stakeholders.

    Parameters
    ----------
    hazard_df : DataFrame         Output of :func:`pipeline.index.compute_hazard_index`.
    insurance_summary : DataFrame Output of :func:`pipeline.index.generate_insurance_summary`.
    output_path : str             Destination .html file path.
    event_name : str              Event label shown in the report header.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    env = Environment(loader=BaseLoader())
    tmpl = env.from_string(_HTML_TEMPLATE)

    # Basin rows — sort by hazard score descending
    df_sorted = hazard_df.reset_index().sort_values("hazard_score", ascending=False)
    basin_rows = df_sorted.fillna(0).to_dict(orient="records")

    # Rename index column
    for row in basin_rows:
        if "huc_id" not in row and "index" in row:
            row["huc_id"] = row.pop("index")

    html = tmpl.render(
        event_name=event_name,
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M UTC"),
        total_basins=len(hazard_df),
        critical_count=int((hazard_df["risk_tier"] == "Critical").sum()),
        high_count=int((hazard_df["risk_tier"] == "High").sum()),
        total_burn_area=round(hazard_df.get("burn_area_km2", pd.Series([0])).sum(), 1),
        summary_rows=insurance_summary.to_dict(orient="records"),
        basin_rows=basin_rows,
    )

    out.write_text(html, encoding="utf-8")
    logger.info("HTML report written: %s", out)
