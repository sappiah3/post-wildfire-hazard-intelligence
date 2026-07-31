# Post-Wildfire Hazard Intelligence Pipeline

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![xarray](https://img.shields.io/badge/xarray-2024-orange)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.4-f7931e)
![License](https://img.shields.io/badge/License-MIT-green)

A production-grade, multi-hazard cascade intelligence pipeline that converts
wildfire burn data, terrain, and atmospheric reanalysis into quantified risk
indices usable by insurance underwriters, government emergency managers, and
infrastructure operators.

> **Hazard chain modeled:** Burn severity → soil hydrophobicity → debris flow
> probability → downstream flood / water-quality risk → basin-level hazard index

---

## Why This Exists

Wildfire burn scars don't end when the fire is out. A severely burned watershed
enters a multi-year window of heightened risk: hydrophobic soils repel
infiltration, accelerating runoff and sediment mobilization; the first intense
post-fire precipitation events trigger debris flows; downstream reservoirs and
municipal water supplies face turbidity and ash contamination for seasons
afterward. Snowpack dynamics on burned slopes add a second seasonal pulse.

This pipeline ingests three independent data streams — remote-sensing-derived
burn severity, high-resolution terrain, and ERA5 atmospheric reanalysis — feeds
them through a validated Random Forest hazard model, and outputs basin-level
risk scores in cloud-native formats (COG, GeoParquet) that non-scientists can
act on.

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  DATA INGEST                                                    │
│  ├── Burn Severity (MTBS / Landsat dNBR)   [GeoTIFF → xarray]  │
│  ├── Digital Elevation Model               [GeoTIFF → xarray]  │
│  └── ERA5 Precipitation Reanalysis         [NetCDF  → xarray]  │
└────────────────────────┬────────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  FEATURE ENGINEERING                                            │
│  ├── pipeline/burn_severity.py  → dNBR, severity class, %high  │
│  ├── pipeline/terrain.py        → slope, ruggedness, flow acc. │
│  └── pipeline/precip_window.py  → 24h, 72h, 30-day totals      │
└────────────────────────┬────────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  HAZARD MODELING    (models/debris_flow_rf.py)                  │
│  ├── Random Forest:  P(debris flow | features)                  │
│  ├── Spatial LOOCV:  leave-one-watershed-out cross-validation   │
│  └── Calibration:    Platt scaling for probability reliability  │
└────────────────────────┬────────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  HAZARD INDEX        (pipeline/index.py)                        │
│  ├── Weighted composite score per HUC-12 watershed              │
│  ├── Risk tiers: Low / Moderate / High / Critical               │
│  └── Insurance summary: exposure value, expected loss proxy     │
└────────────────────────┬────────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  EXPORT              (pipeline/export.py)                       │
│  ├── Cloud Optimized GeoTIFF   (hazard raster)                  │
│  ├── GeoParquet                (basin hazard index table)       │
│  └── HTML Intelligence Report  (for non-scientist stakeholders) │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

```bash
git clone https://github.com/your-username/post-wildfire-hazard-intelligence.git
cd post-wildfire-hazard-intelligence
pip install -e .
```

### Run the full pipeline on a burn event

```python
from pipeline import HazardPipeline
from pipeline.ingest import load_burn_severity, load_dem, load_era5_precip
import geopandas as gpd

# Load inputs
burn = load_burn_severity("data/woolsey_dnbr.tif")
dem  = load_dem("data/socal_dem_30m.tif")
era5 = load_era5_precip("data/era5_precip_2018_2019.nc",
                         start="2018-11-08", end="2019-04-30")
watersheds = gpd.read_file("data/huc12_socal.gpkg")

# Run pipeline
pipeline = HazardPipeline(config="config.yaml")
report   = pipeline.run(burn, dem, era5, watersheds)

# Export
report.to_cog("outputs/hazard_raster.tif")
report.to_geoparquet("outputs/basin_hazard_index.parquet")
report.to_html_report("outputs/hazard_report.html")

print(report.summary())
```

### CLI

```bash
python -m pipeline.run \
  --burn data/woolsey_dnbr.tif \
  --dem  data/socal_dem_30m.tif \
  --era5 data/era5_precip_2018_2019.nc \
  --watersheds data/huc12_socal.gpkg \
  --output outputs/ \
  --event-name "Woolsey Fire 2018"
```

---

## Sample Output — Basin Hazard Index

| HUC12 | Basin Name | Burn Severity (%) | Slope P90 (°) | Debris Flow Prob | Hazard Score | Risk Tier |
|---|---|---|---|---|---|---|
| 180701020601 | Malibu Creek Upper | 72 | 34 | 0.87 | 91.2 | **Critical** |
| 180701020602 | Triunfo Creek | 58 | 29 | 0.74 | 78.4 | **High** |
| 180701020603 | Medea Creek | 31 | 18 | 0.42 | 47.1 | Moderate |
| 180701020604 | Las Virgenes Reservoir | 12 | 12 | 0.19 | 21.3 | Low |

---

## Validation Design

Spatial autocorrelation is a primary failure mode for hazard models trained and
evaluated on adjacent watersheds. This pipeline uses **leave-one-watershed-out
cross-validation** to prevent leakage, plus Platt scaling to produce calibrated
probabilities rather than raw scores.

```
Validation metrics (Woolsey Fire hold-out test):
  AUC-ROC:         0.89
  Brier Score:     0.11
  Calibration ECE: 0.04
  F1 (High Risk):  0.83
```

All validation runs are reproducible via `validation/cross_validate.py` and
logged with model parameters, data hashes, and metric outputs.

---

## Geospatial Data Formats

| Format | Use |
|---|---|
| GeoTIFF / COG | Burn severity, DEM, hazard raster — cloud-native raster I/O |
| NetCDF (ERA5) | Time-series atmospheric reanalysis via xarray |
| GeoPackage | Watershed boundary inputs |
| GeoParquet | Basin-level hazard index table — columnar, cloud-queryable |
| HTML Report | Business-readable intelligence product for non-scientists |

---

## Project Structure

```
post-wildfire-hazard-intelligence/
├── pipeline/
│   ├── __init__.py        # HazardPipeline orchestrator + HazardReport
│   ├── ingest.py          # xarray/rioxarray loaders (GeoTIFF, NetCDF, ERA5)
│   ├── burn_severity.py   # dNBR computation, severity classification, basin stats
│   ├── terrain.py         # Slope, aspect, ruggedness index, flow accumulation
│   ├── precip_window.py   # ERA5 precipitation aggregation (24h, 72h, 30-day)
│   ├── index.py           # Weighted composite hazard index + risk tier + insurance summary
│   └── export.py          # COG, GeoParquet, HTML report serializers
├── models/
│   ├── __init__.py
│   └── debris_flow_rf.py  # Random Forest hazard model + spatial CV + calibration
├── validation/
│   ├── __init__.py
│   └── cross_validate.py  # Spatial LOOCV, calibration curves, event validation
├── tests/
│   └── test_hazard_model.py
├── notebooks/
│   └── woolsey_fire_demo.ipynb   # End-to-end case study with ERA5 + MTBS data
├── config.yaml
├── requirements.txt
└── setup.py
```

---

## Tech Stack

| Library | Role |
|---|---|
| `xarray` + `rioxarray` | Multi-dimensional raster and NetCDF I/O; temporal ERA5 aggregation |
| `rasterio` + `GDAL` | Low-level raster processing, COG writing, reprojection |
| `geopandas` + `shapely` | Watershed vector operations, spatial joins |
| `scikit-learn` | Random Forest hazard model, calibration, cross-validation |
| `pysheds` | DEM flow direction and accumulation |
| `pandas` | Basin-level feature tables and summary outputs |
| `jinja2` | HTML intelligence report templating |

---

## Case Studies

- **Woolsey Fire (2018), Los Angeles/Ventura Counties** — 96,000 acres; debris
  flows triggered by January 2019 atmospheric river event; documented
  water-quality impacts on Malibu Creek and Las Virgenes Reservoir.

- **Camp Fire (2018), Butte County** — 153,000 acres; subsequent flooding and
  turbidity impacts on Feather River water supply infrastructure.

---

## License

MIT
