"""
CLI Runner
==========
Command-line entry point for the post-wildfire hazard pipeline.

Usage
-----
  python -m pipeline.run \\
    --burn   data/woolsey_dnbr.tif \\
    --dem    data/socal_dem_30m.tif \\
    --era5   data/era5_precip_2018_2019.nc \\
    --watersheds data/huc12_socal.gpkg \\
    --output outputs/ \\
    --event-name "Woolsey Fire 2018" \\
    --event-start 2018-11-08 \\
    --event-end   2019-04-30

Or using the installed entry point:
  hazard-pipeline --burn ... --dem ... --era5 ... --watersheds ...
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Post-Wildfire Hazard Intelligence Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--burn",       required=True, help="dNBR / burn severity GeoTIFF")
    parser.add_argument("--dem",        required=True, help="Digital Elevation Model GeoTIFF")
    parser.add_argument("--era5",       required=True, help="ERA5 precipitation NetCDF")
    parser.add_argument("--watersheds", required=True, help="HUC-12 boundaries (GeoPackage/Shapefile)")
    parser.add_argument("--output",     default="outputs/", help="Output directory")
    parser.add_argument("--config",     default="config.yaml", help="Pipeline configuration YAML")
    parser.add_argument("--model",      default=None, help="Pre-trained model (.joblib); trains from scratch if not provided")
    parser.add_argument("--labels",     default=None, help="CSV with columns [huc_id, debris_flow] for training")
    parser.add_argument("--event-name", default="Wildfire Event", help="Event label for reports")
    parser.add_argument("--event-start", default=None, help="Post-fire period start date (YYYY-MM-DD)")
    parser.add_argument("--event-end",   default=None, help="Post-fire period end date (YYYY-MM-DD)")
    parser.add_argument("--run-cv",  action="store_true", help="Run spatial cross-validation (slow)")
    parser.add_argument("--formats", nargs="+", choices=["html", "parquet"], default=["html", "parquet"])
    return parser.parse_args()


def main():
    args = parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # ---- Imports ----
    from pipeline import HazardPipeline
    from pipeline.ingest import load_burn_severity, load_dem, load_era5_precip, load_watersheds
    import pandas as pd

    # ---- Load inputs ----
    logger.info("Loading inputs…")
    burn       = load_burn_severity(args.burn)
    dem        = load_dem(args.dem)
    era5       = load_era5_precip(args.era5, start=args.event_start, end=args.event_end)
    watersheds = load_watersheds(args.watersheds)

    # ---- Labels (optional, for training) ----
    labels = None
    if args.labels:
        ldf = pd.read_csv(args.labels).set_index("huc_id")["debris_flow"]
        labels = ldf
        logger.info("Labels loaded: %d basins (%d positive)", len(labels), int(labels.sum()))

    # ---- Run pipeline ----
    pipeline = HazardPipeline(config=args.config, model_path=args.model)
    report = pipeline.run(
        burn=burn,
        dem=dem,
        era5=era5,
        watersheds=watersheds,
        labels=labels,
        event_name=args.event_name,
        event_start=args.event_start,
        event_end=args.event_end,
        run_cv=args.run_cv,
    )

    print(report.summary())

    # ---- Export ----
    if "html" in args.formats:
        html_path = str(out / "hazard_report.html")
        report.to_html_report(html_path)
        logger.info("HTML report: %s", html_path)

    if "parquet" in args.formats:
        parquet_path = str(out / "basin_hazard_index.parquet")
        report.to_geoparquet(parquet_path)
        logger.info("GeoParquet: %s", parquet_path)

    # Always write CSV for easy inspection
    csv_path = str(out / "basin_hazard_index.csv")
    report.hazard_df.drop(columns=["geometry"], errors="ignore").to_csv(csv_path)
    logger.info("CSV: %s", csv_path)

    # Exit with non-zero code if any Critical basins found
    n_critical = (report.hazard_df["risk_tier"] == "Critical").sum()
    sys.exit(0)  # let calling systems decide on criticality


if __name__ == "__main__":
    main()
