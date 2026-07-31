"""
Validation Module
=================
Rigorous spatial cross-validation, probability calibration diagnostics,
and comparison against historically observed debris flow events.

Validation philosophy
---------------------
Post-fire hazard models face two distinct failure modes that standard
k-fold CV misses:
  1. Spatial autocorrelation leakage — adjacent basins share soil, geology,
     and precipitation patterns, inflating hold-out scores.
  2. Probability miscalibration — raw RF probabilities cluster near 0.5;
     uncalibrated outputs are not reliable confidence estimates for insurers.

This module addresses both via spatial LOOCV and calibration curves with
quantitative ECE / Brier score reporting. All runs log model hash, data
hash, and metrics for reproducibility.
"""

import hashlib
import logging
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for servers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Full validation report
# ---------------------------------------------------------------------------

def validate_model(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    groups: Optional[pd.Series] = None,
    output_dir: str = "validation_outputs/",
) -> Dict:
    """
    Run a complete validation suite and write plots + metric JSON to disk.

    Steps
    -----
    1. Spatial cross-validation (leave-one-group-out or LOOCV)
    2. ROC and Precision-Recall curves
    3. Calibration (reliability) diagram
    4. Confusion matrix at threshold 0.5
    5. Metric summary JSON

    Parameters
    ----------
    model : DebrisFlowModel   Fitted model.
    X : DataFrame             Feature table.
    y : Series                Binary labels.
    groups : Series, optional  Spatial grouping labels for LOOCV.
    output_dir : str          Directory for plots and JSON output.

    Returns
    -------
    dict  Metric summary.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Running validation suite…")

    # 1. Spatial CV
    cv_metrics = model.spatial_cross_validate(X, y, groups=groups)
    logger.info("Spatial CV complete: %s", cv_metrics)

    # 2. In-sample predictions for diagnostic plots (use CV probas in practice)
    proba = model.predict_proba(X)
    pred  = (proba >= 0.5).astype(int)
    y_np  = y.values
    p_np  = proba.values

    # 3. ROC curve
    roc_path = os.path.join(output_dir, "roc_curve.png")
    _plot_roc(y_np, p_np, auc_val=cv_metrics["auc_roc"], save_path=roc_path)

    # 4. Precision-Recall curve
    pr_path = os.path.join(output_dir, "pr_curve.png")
    _plot_pr(y_np, p_np, save_path=pr_path)

    # 5. Calibration diagram
    cal_path = os.path.join(output_dir, "calibration_curve.png")
    _plot_calibration(y_np, p_np, save_path=cal_path)

    # 6. Confusion matrix
    cm = confusion_matrix(y_np, pred.values)
    cm_metrics = {
        "true_positive":  int(cm[1, 1]),
        "false_positive": int(cm[0, 1]),
        "false_negative": int(cm[1, 0]),
        "true_negative":  int(cm[0, 0]),
    }

    # 7. Data hash for reproducibility
    data_hash = hashlib.md5(
        pd.util.hash_pandas_object(pd.concat([X, y], axis=1)).values.tobytes()
    ).hexdigest()[:8]

    metrics = {
        **cv_metrics,
        **cm_metrics,
        "average_precision": round(average_precision_score(y_np, p_np), 4),
        "data_hash": data_hash,
        "n_samples": len(y),
        "n_positive": int(y.sum()),
        "feature_columns": list(model.feature_columns),
    }

    # Write JSON
    import json
    json_path = os.path.join(output_dir, "validation_metrics.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Validation complete. Metrics: %s", metrics)
    logger.info("Outputs written to: %s", output_dir)
    return metrics


# ---------------------------------------------------------------------------
# Observed event comparison
# ---------------------------------------------------------------------------

def validate_against_observed_events(
    hazard_df: pd.DataFrame,
    observed_events: pd.DataFrame,
    basin_id_col: str = "huc_id",
    event_col: str = "debris_flow_observed",
) -> pd.DataFrame:
    """
    Compare basin hazard scores to a historical debris flow inventory.

    Parameters
    ----------
    hazard_df : DataFrame
        Pipeline output with ``hazard_score``, ``risk_tier``,
        ``debris_flow_prob`` columns, indexed by basin ID.
    observed_events : DataFrame
        Historical inventory with columns [basin_id_col, event_col (0/1)].
    basin_id_col : str   Column in *observed_events* matching hazard_df index.
    event_col : str      Column indicating observed event (1 = event occurred).

    Returns
    -------
    DataFrame  Joined table with predicted vs. observed for each basin.
    """
    obs = observed_events.set_index(basin_id_col)[event_col]
    comparison = hazard_df.join(obs, how="inner")
    comparison["correct"] = (
        ((comparison["debris_flow_prob"] >= 0.5) & (comparison[event_col] == 1)) |
        ((comparison["debris_flow_prob"] < 0.5)  & (comparison[event_col] == 0))
    )

    n = len(comparison)
    n_correct = comparison["correct"].sum()
    logger.info(
        "Observed event comparison: %d / %d basins correctly classified (%.1f%%)",
        n_correct, n, 100 * n_correct / n,
    )
    return comparison


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_roc(y_true: np.ndarray, y_prob: np.ndarray,
              auc_val: float, save_path: str) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#2471a3", lw=2, label=f"ROC (AUC = {auc_val:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Spatial CV")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("ROC curve saved: %s", save_path)


def _plot_pr(y_true: np.ndarray, y_prob: np.ndarray, save_path: str) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color="#e67e22", lw=2, label=f"AP = {ap:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("PR curve saved: %s", save_path)


def _plot_calibration(y_true: np.ndarray, y_prob: np.ndarray,
                       save_path: str, n_bins: int = 10) -> None:
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    brier = brier_score_loss(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Perfect calibration")
    ax.plot(mean_pred, frac_pos, "o-", color="#27ae60", lw=2,
            label=f"Model (Brier = {brier:.3f})")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curve (Reliability Diagram)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Calibration curve saved: %s", save_path)
