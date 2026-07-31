"""
Tests for DebrisFlowModel and hazard index computation.
Run: pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box
import geopandas as gpd

from models.debris_flow_rf import DebrisFlowModel, FEATURE_COLUMNS
from pipeline.index import (
    _normalize,
    _assign_tier,
    compute_hazard_index,
    generate_insurance_summary,
    RISK_TIERS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_features() -> pd.DataFrame:
    """40 synthetic basins with realistic feature ranges."""
    rng = np.random.default_rng(42)
    n = 40
    return pd.DataFrame(
        {
            "pct_high_severity":     rng.uniform(0, 100, n),
            "mean_dnbr":             rng.uniform(-100, 800, n),
            "max_dnbr":              rng.uniform(200, 1200, n),
            "std_dnbr":              rng.uniform(50, 300, n),
            "slope_p50":             rng.uniform(5, 30, n),
            "slope_p75":             rng.uniform(10, 40, n),
            "slope_p90":             rng.uniform(20, 60, n),
            "slope_mean":            rng.uniform(8, 35, n),
            "tri_mean":              rng.uniform(2, 25, n),
            "elevation_range":       rng.uniform(50, 800, n),
            "max_accum_24h_mm":      rng.uniform(0, 120, n),
            "max_accum_72h_mm":      rng.uniform(0, 200, n),
            "precip_exceedance_flag": rng.integers(0, 2, n),
            "burn_area_km2":         rng.uniform(0.5, 50, n),
        },
        index=[f"HUC{i:04d}" for i in range(n)],
    )


@pytest.fixture
def synthetic_labels(synthetic_features) -> pd.Series:
    """Binary labels with ~30% positive rate."""
    rng = np.random.default_rng(42)
    n = len(synthetic_features)
    return pd.Series(
        rng.binomial(1, 0.3, n),
        index=synthetic_features.index,
        name="debris_flow",
    )


@pytest.fixture
def trained_model(synthetic_features, synthetic_labels) -> DebrisFlowModel:
    model = DebrisFlowModel(n_estimators=50, max_depth=4, random_state=42)
    return model.train(synthetic_features, synthetic_labels)


# ---------------------------------------------------------------------------
# DebrisFlowModel tests
# ---------------------------------------------------------------------------

class TestDebrisFlowModel:

    def test_train_returns_self(self, synthetic_features, synthetic_labels):
        model = DebrisFlowModel(n_estimators=50, max_depth=3, random_state=0)
        result = model.train(synthetic_features, synthetic_labels)
        assert result is model
        assert model.is_fitted_

    def test_predict_proba_shape(self, trained_model, synthetic_features):
        proba = trained_model.predict_proba(synthetic_features)
        assert len(proba) == len(synthetic_features)
        assert proba.index.equals(synthetic_features.index)

    def test_predict_proba_bounded(self, trained_model, synthetic_features):
        proba = trained_model.predict_proba(synthetic_features)
        assert (proba >= 0.0).all(), "Probabilities must be >= 0"
        assert (proba <= 1.0).all(), "Probabilities must be <= 1"

    def test_feature_importances_sum_to_one(self, trained_model):
        fi = trained_model.feature_importances_
        assert fi is not None
        assert np.isclose(fi.sum(), 1.0, atol=1e-6)

    def test_unfitted_model_raises(self, synthetic_features):
        model = DebrisFlowModel()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict_proba(synthetic_features)

    def test_missing_features_handled_gracefully(self, trained_model):
        """Model should handle missing feature columns with a warning."""
        partial_df = pd.DataFrame(
            {"pct_high_severity": [50.0], "slope_p90": [30.0]},
            index=["TEST_BASIN"],
        )
        proba = trained_model.predict_proba(partial_df)
        assert len(proba) == 1
        assert 0.0 <= float(proba.iloc[0]) <= 1.0

    def test_spatial_cv_returns_expected_keys(self, synthetic_features, synthetic_labels):
        model = DebrisFlowModel(n_estimators=20, max_depth=3, random_state=0)
        model.train(synthetic_features, synthetic_labels)
        # Use groups to speed up the test (leave-one-group-out instead of full LOOCV)
        groups = pd.Series(
            [f"G{i % 5}" for i in range(len(synthetic_features))],
            index=synthetic_features.index,
        )
        metrics = model.spatial_cross_validate(synthetic_features, synthetic_labels, groups=groups)
        for key in ("auc_roc", "brier_score", "f1", "ece"):
            assert key in metrics, f"Expected key '{key}' in metrics"

    def test_save_and_load_roundtrip(self, trained_model, synthetic_features, tmp_path):
        path = str(tmp_path / "model.joblib")
        trained_model.save(path)
        loaded = DebrisFlowModel.load(path)
        assert loaded.is_fitted_
        p1 = trained_model.predict_proba(synthetic_features)
        p2 = loaded.predict_proba(synthetic_features)
        pd.testing.assert_series_equal(p1, p2)


# ---------------------------------------------------------------------------
# Hazard Index tests
# ---------------------------------------------------------------------------

class TestHazardIndex:

    def test_normalize_range(self):
        s = pd.Series([0.0, 5.0, 10.0])
        norm = _normalize(s)
        assert float(norm.min()) == pytest.approx(0.0)
        assert float(norm.max()) == pytest.approx(1.0)

    def test_normalize_constant_series_returns_zeros(self):
        s = pd.Series([7.0, 7.0, 7.0])
        norm = _normalize(s)
        assert (norm == 0.0).all()

    def test_assign_tier_boundaries(self):
        assert _assign_tier(0.0)   == "Low"
        assert _assign_tier(24.9)  == "Low"
        assert _assign_tier(25.0)  == "Moderate"
        assert _assign_tier(50.0)  == "High"
        assert _assign_tier(75.0)  == "Critical"
        assert _assign_tier(100.0) == "Critical"

    def test_compute_hazard_index_columns(self, synthetic_features, trained_model):
        proba = trained_model.predict_proba(synthetic_features)
        hazard = compute_hazard_index(synthetic_features, proba)
        assert "hazard_score" in hazard.columns
        assert "risk_tier" in hazard.columns
        assert hazard["risk_tier"].isin(["Low", "Moderate", "High", "Critical"]).all()

    def test_hazard_score_bounded(self, synthetic_features, trained_model):
        proba = trained_model.predict_proba(synthetic_features)
        hazard = compute_hazard_index(synthetic_features, proba)
        assert (hazard["hazard_score"] >= 0).all()
        assert (hazard["hazard_score"] <= 100).all()

    def test_insurance_summary_tiers(self, synthetic_features, trained_model):
        proba = trained_model.predict_proba(synthetic_features)
        hazard = compute_hazard_index(synthetic_features, proba)
        summary = generate_insurance_summary(hazard, event_name="Test Fire")
        tiers_in_summary = set(summary["risk_tier"].tolist())
        assert tiers_in_summary == {"Low", "Moderate", "High", "Critical"}

    def test_custom_weights_respected(self, synthetic_features, trained_model):
        proba = trained_model.predict_proba(synthetic_features)
        w1 = {"debris_flow_prob": 1.0, "pct_high_severity": 0.0, "slope_p90": 0.0, "max_accum_24h_mm": 0.0}
        h1 = compute_hazard_index(synthetic_features, proba, weights=w1)
        w2 = {"debris_flow_prob": 0.0, "pct_high_severity": 1.0, "slope_p90": 0.0, "max_accum_24h_mm": 0.0}
        h2 = compute_hazard_index(synthetic_features, proba, weights=w2)
        # The two weight schemes should produce different orderings
        assert not (h1["hazard_score"] == h2["hazard_score"]).all()

    def test_invalid_weights_raise(self, synthetic_features, trained_model):
        proba = trained_model.predict_proba(synthetic_features)
        bad_weights = {"debris_flow_prob": 0.6, "pct_high_severity": 0.6}
        with pytest.raises(ValueError, match="sum to 1.0"):
            compute_hazard_index(synthetic_features, proba, weights=bad_weights)
