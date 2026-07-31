"""
Pipeline Orchestrator
=====================
The HazardPipeline class wires together all pipeline stages and returns
a HazardReport object that encapsulates results and export methods.

This is the primary public API for the pipeline.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import geopandas as gpd
import pandas as pd
import xarray as xr
import yaml

from .burn_severity import burn_severity_stats, classify_burn_severity
from .export import to_cog, to_geoparquet, to_html_report
from .index import compute_hazard_index, generate_insurance_summary
from .precip_window import precip_basin_stats
from .terrain import compute_slope, compute_tri, terrain_stats
from models.debris_flow_rf import DebrisFlowModel

logger = logging.getLogger(__name__)


@dataclass
class HazardReport:
    """
    Container for all pipeline outputs.

    Attributes
    ----------
    hazard_df : DataFrame     Basin hazard index table.
    insurance_summary : DataFrame  Risk tier summary for stakeholders.
    debris_flow_proba : Series     Model probabilities per basin.
    burn_features : DataFrame      Per-basin burn severity statistics.
    terrain_features : DataFrame   Per-basin terrain statistics.
    precip_features : DataFrame    Per-basin precipitation statistics.
    cv_metrics : dict              Spatial cross-validation metrics.
    run_time_s : float             Total pipeline run time in seconds.
    event_name : str               Label for this fire event.
    """
    hazard_df: pd.DataFrame
    insurance_summary: pd.DataFrame
    debris_flow_proba: pd.Series
    burn_features: pd.DataFrame = field(default_factory=pd.DataFrame)
    terrain_features: pd.DataFrame = field(default_factory=pd.DataFrame)
    precip_features: pd.DataFrame = field(default_factory=pd.DataFrame)
    cv_metrics: Dict[str, Any] = field(default_factory=dict)
    run_time_s: float = 0.0
    event_name: str = "Wildfire Event"

    def summary(self) -> str:
        """Return a plain-text run summary."""
        tiers = self.hazard_df["risk_tier"].value_counts().to_dict()
        lines = [
            f"=== HAZARD INTELLIGENCE REPORT: {self.event_name} ===",
            f"  Total basins assessed:  {len(self.hazard_df)}",
            f"  Critical:               {tiers.get('Critical', 0)}",
            f"  High:                   {tiers.get('High', 0)}",
            f"  Moderate:               {tiers.get('Moderate', 0)}",
            f"  Low:                    {tiers.get('Low', 0)}",
        ]
        if self.cv_metrics:
            lines += [
                f"  AUC-ROC (spatial CV):   {self.cv_metrics.get('auc_roc', 'N/A')}",
                f"  Brier Score:            {self.cv_metrics.get('brier_score', 'N/A')}",
                f"  ECE:                    {self.cv_metrics.get('ece', 'N/A')}",
            ]
        lines.append(f"  Pipeline run time:      {self.run_time_s:.1f}s")
        return "\n".join(lines)

    def to_cog(self, path: str, data: Optional[xr.DataArray] = None, **kwargs) -> None:
        if data is None:
            raise ValueError("Pass a DataArray (e.g. hazard score raster) to to_cog().")
        to_cog(data, path, **kwargs)

    def to_geoparquet(self, path: str, geometry_col: Optional[str] = None) -> None:
        """Write hazard_df to GeoParquet. Requires geometry column."""
        if isinstance(self.hazard_df, gpd.GeoDataFrame):
            to_geoparquet(self.hazard_df, path)
        else:
            logger.warning("hazard_df has no geometry — writing plain Parquet")
            self.hazard_df.to_parquet(path)

    def to_html_report(self, path: str) -> None:
        to_html_report(
            self.hazard_df,
            self.insurance_summary,
            path,
            event_name=self.event_name,
        )


class HazardPipeline:
    """
    Full post-wildfire hazard intelligence pipeline.

    Parameters
    ----------
    config : str | dict
        Path to ``config.yaml`` or an already-parsed config dict.
    model_path : str, optional
        Path to a pre-trained DebrisFlowModel (.joblib). If None, the model
        is trained from scratch using the watershed labels in *watersheds*.
    """

    def __init__(
        self,
        config: Any = "config.yaml",
        model_path: Optional[str] = None,
    ):
        if isinstance(config, str):
            with open(config) as f:
                self.cfg = yaml.safe_load(f)
        else:
            self.cfg = config

        self.model: Optional[DebrisFlowModel] = None
        if model_path:
            self.model = DebrisFlowModel.load(model_path)
            logger.info("Pre-trained model loaded from %s", model_path)

    def run(
        self,
        burn: xr.DataArray,
        dem: xr.DataArray,
        era5: xr.DataArray,
        watersheds: gpd.GeoDataFrame,
        labels: Optional[pd.Series] = None,
        event_name: str = "Wildfire Event",
        event_start: Optional[str] = None,
        event_end: Optional[str] = None,
        run_cv: bool = False,
    ) -> HazardReport:
        """
        Execute the full pipeline end-to-end.

        Parameters
        ----------
        burn : xr.DataArray      Burn severity / dNBR raster.
        dem : xr.DataArray       Digital Elevation Model.
        era5 : xr.DataArray      ERA5 hourly precipitation (mm/h).
        watersheds : GeoDataFrame  HUC-12 boundary polygons.
        labels : pd.Series, optional
            Binary debris flow labels (huc_id index) for model training.
            Required if no pre-trained model is loaded.
        event_name : str
        event_start, event_end : str, optional  Temporal window for precip.
        run_cv : bool   Whether to run full spatial CV (slow on large datasets).

        Returns
        -------
        HazardReport
        """
        t0 = time.time()
        logger.info("=== HazardPipeline.run() started: %s ===", event_name)

        # Stage 1: Feature engineering — burn severity
        logger.info("[1/4] Computing burn severity statistics…")
        classified = classify_burn_severity(burn)
        burn_feat  = burn_severity_stats(burn, watersheds, classified=classified)

        # Stage 2: Terrain
        logger.info("[2/4] Computing terrain statistics…")
        slope = compute_slope(dem)
        tri   = compute_tri(dem, window=self.cfg.get("terrain", {}).get("ruggedness_window", 3))
        terr_feat = terrain_stats(
            slope, dem, watersheds,
            slope_percentiles=self.cfg.get("terrain", {}).get("slope_percentiles", [50, 75, 90]),
            tri=tri,
        )

        # Stage 3: Precipitation
        logger.info("[3/4] Computing precipitation statistics…")
        precip_feat = precip_basin_stats(
            era5, watersheds,
            windows_h=self.cfg.get("precipitation", {}).get("windows_hours", [6, 24, 72]),
            exceedance_threshold_mm=self.cfg.get("precipitation", {}).get("return_period_threshold_mm", 25.0),
            event_start=event_start,
            event_end=event_end,
        )

        # Merge all feature tables
        features = burn_feat.join(terr_feat, how="outer").join(precip_feat, how="outer")

        # Stage 4: Hazard model
        logger.info("[4/4] Running debris flow hazard model…")
        model_cfg = self.cfg.get("hazard_model", {})

        if self.model is None:
            if labels is None:
                raise ValueError(
                    "No pre-trained model and no training labels provided. "
                    "Pass labels= or model_path= to HazardPipeline."
                )
            self.model = DebrisFlowModel(
                n_estimators=model_cfg.get("n_estimators", 500),
                max_depth=model_cfg.get("max_depth", 8),
                min_samples_leaf=model_cfg.get("min_samples_leaf", 5),
                calibration=model_cfg.get("calibration", "sigmoid"),
            )
            self.model.train(features, labels.reindex(features.index).fillna(0))

        proba = self.model.predict_proba(features)

        # Optional: run spatial CV
        cv_metrics: Dict = {}
        if run_cv and labels is not None:
            cv_metrics = self.model.spatial_cross_validate(
                features,
                labels.reindex(features.index).fillna(0),
            )

        # Stage 5: Index and summary
        index_weights = self.cfg.get("hazard_index", {}).get("weights")
        hazard_df = compute_hazard_index(features, proba, weights=index_weights)
        insurance_summary = generate_insurance_summary(hazard_df, event_name=event_name)

        # Attach geometry for GeoParquet export
        if isinstance(watersheds, gpd.GeoDataFrame):
            hazard_gdf = gpd.GeoDataFrame(
                hazard_df.join(watersheds[["geometry"]], how="left"),
                geometry="geometry",
                crs=watersheds.crs,
            )
        else:
            hazard_gdf = hazard_df

        run_time = time.time() - t0
        logger.info("=== Pipeline complete in %.1fs ===", run_time)

        return HazardReport(
            hazard_df=hazard_gdf,
            insurance_summary=insurance_summary,
            debris_flow_proba=proba,
            burn_features=burn_feat,
            terrain_features=terr_feat,
            precip_features=precip_feat,
            cv_metrics=cv_metrics,
            run_time_s=run_time,
            event_name=event_name,
        )
