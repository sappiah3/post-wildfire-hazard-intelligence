"""
Hazard Index Module
===================
Compute a basin-level multi-hazard composite score from model outputs
and raw feature statistics. Classify basins into risk tiers and generate
an insurance-ready summary for non-scientist stakeholders.

The index design is intentionally transparent and configurable so that
insurance underwriters and government clients can audit and adjust
weights without black-box concerns.
"""

import logging
from typing import Dict, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default component weights — must sum to 1.0.
DEFAULT_WEIGHTS: Dict[str, float] = {
    "debris_flow_prob":  0.50,
    "pct_high_severity": 0.25,
    "slope_p90":         0.15,
    "max_accum_24h_mm":  0.10,
}

# Risk tier boundaries (hazard score 0–100)
RISK_TIERS: Dict[str, Tuple[float, float]] = {
    "Low":      (0,   25),
    "Moderate": (25,  50),
    "High":     (50,  75),
    "Critical": (75, 100),
}


# ---------------------------------------------------------------------------
# Component normalization
# ---------------------------------------------------------------------------

def _normalize(series: pd.Series, clip_upper: Optional[float] = None) -> pd.Series:
    """Min-max normalize to [0, 1]. Optionally clip extreme values first."""
    s = series.copy().astype(float)
    if clip_upper is not None:
        s = s.clip(upper=clip_upper)
    s_min, s_max = s.min(), s.max()
    if s_max == s_min:
        return pd.Series(0.0, index=series.index)
    return (s - s_min) / (s_max - s_min)


# ---------------------------------------------------------------------------
# Composite index
# ---------------------------------------------------------------------------

def compute_hazard_index(
    features: pd.DataFrame,
    debris_flow_prob: pd.Series,
    weights: Optional[Dict[str, float]] = None,
) -> gpd.GeoDataFrame:
    """
    Compute a weighted composite hazard score (0–100) per basin.

    Parameters
    ----------
    features : DataFrame
        Basin feature table containing at least:
        ``pct_high_severity``, ``slope_p90``, ``max_accum_24h_mm``.
    debris_flow_prob : Series
        Per-basin calibrated debris flow probability from the RF model.
    weights : dict, optional
        Component weights dict. Must sum to 1.0. Defaults to DEFAULT_WEIGHTS.

    Returns
    -------
    DataFrame  Basin index with hazard score, tier, and component columns.
    """
    weights = weights or DEFAULT_WEIGHTS
    w_total = sum(weights.values())
    if not np.isclose(w_total, 1.0):
        raise ValueError(f"Weights must sum to 1.0; got {w_total:.3f}")

    df = features.copy()
    df["debris_flow_prob"] = debris_flow_prob.reindex(df.index)

    # Normalize each component to [0, 1]
    norm: Dict[str, pd.Series] = {
        "debris_flow_prob":  _normalize(df["debris_flow_prob"]),
        "pct_high_severity": _normalize(df["pct_high_severity"], clip_upper=100.0),
        "slope_p90":         _normalize(df.get("slope_p90", pd.Series(0.0, index=df.index)), clip_upper=60.0),
        "max_accum_24h_mm":  _normalize(df.get("max_accum_24h_mm", pd.Series(0.0, index=df.index)), clip_upper=150.0),
    }

    # Weighted composite (scale to 0–100)
    score = sum(
        weights.get(k, 0.0) * v
        for k, v in norm.items()
        if k in weights
    ) * 100.0

    result = df.copy()
    result["hazard_score"] = score.round(2)
    result["risk_tier"]    = score.apply(_assign_tier)

    # Add normalized component columns for auditability
    for comp, vals in norm.items():
        result[f"norm_{comp}"] = (vals * 100).round(2)

    logger.info(
        "Hazard index computed: Critical=%d, High=%d, Moderate=%d, Low=%d",
        (result["risk_tier"] == "Critical").sum(),
        (result["risk_tier"] == "High").sum(),
        (result["risk_tier"] == "Moderate").sum(),
        (result["risk_tier"] == "Low").sum(),
    )
    return result


def _assign_tier(score: float) -> str:
    for tier, (lo, hi) in RISK_TIERS.items():
        if lo <= score < hi:
            return tier
    return "Critical"  # handles score == 100


# ---------------------------------------------------------------------------
# Insurance summary
# ---------------------------------------------------------------------------

def generate_insurance_summary(
    hazard_df: pd.DataFrame,
    event_name: str = "Wildfire Event",
    currency: str = "USD",
) -> pd.DataFrame:
    """
    Generate a business-readable risk summary for insurance underwriters.

    This output is deliberately structured for non-scientists: it presents
    exposure counts, tier distributions, and qualitative risk language
    rather than raw model metrics.

    Parameters
    ----------
    hazard_df : DataFrame   Output of :func:`compute_hazard_index`.
    event_name : str        Label for this event (e.g. "Woolsey Fire 2018").
    currency : str          Currency label for any monetary proxies.

    Returns
    -------
    DataFrame  Summary with one row per risk tier.
    """
    tiers = ["Critical", "High", "Moderate", "Low"]
    rows = []

    for tier in tiers:
        tier_df = hazard_df[hazard_df["risk_tier"] == tier]
        n = len(tier_df)
        burn_area = tier_df.get("burn_area_km2", pd.Series(dtype=float)).sum()
        mean_score = tier_df["hazard_score"].mean() if n else np.nan
        mean_prob  = tier_df.get("debris_flow_prob", pd.Series(dtype=float)).mean()

        rows.append({
            "event":             event_name,
            "risk_tier":         tier,
            "basin_count":       n,
            "burn_area_km2":     round(burn_area, 1) if not np.isnan(burn_area) else 0,
            "mean_hazard_score": round(mean_score, 1) if not np.isnan(mean_score) else 0,
            "mean_debris_flow_prob": round(mean_prob, 3) if not np.isnan(mean_prob) else 0,
            "recommended_action": _tier_action(tier),
        })

    summary = pd.DataFrame(rows)
    total_basins = summary["basin_count"].sum()
    logger.info(
        "Insurance summary — %s: %d basins total, %d Critical",
        event_name, total_basins, summary.loc[summary["risk_tier"] == "Critical", "basin_count"].values[0],
    )
    return summary


def _tier_action(tier: str) -> str:
    actions = {
        "Critical": "Immediate underwriting review; suspend new coverage pending field assessment",
        "High":     "Elevated premium adjustment; require debris-flow endorsement disclosure",
        "Moderate": "Standard review cycle; flag for next renewal assessment",
        "Low":      "Routine monitoring; no immediate underwriting action required",
    }
    return actions.get(tier, "Unknown")
