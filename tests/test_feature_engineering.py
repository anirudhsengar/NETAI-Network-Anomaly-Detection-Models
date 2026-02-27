"""Tests for feature engineering pipeline."""

import numpy as np
import pandas as pd
import pytest

from netai_anomaly.data.feature_engineering import (
    add_derived_features,
    add_lag_features,
    add_rolling_features,
    add_time_features,
    build_feature_matrix,
)


@pytest.fixture
def sample_df():
    """Create a sample DataFrame mimicking network telemetry."""
    n = 200
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "id": range(n),
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="5min"),
        "source_host": ["host-a"] * n,
        "destination_host": ["host-b"] * n,
        "test_type": ["throughput"] * n,
        "throughput_mbps": rng.normal(5000, 500, n),
        "latency_ms": rng.normal(20, 5, n),
        "packet_loss_pct": np.abs(rng.normal(0.05, 0.02, n)),
        "retransmits": rng.poisson(3, n),
        "jitter_ms": np.abs(rng.normal(1.5, 0.5, n)),
        "mtu": [1500] * n,
        "tcp_window_size": [65536] * n,
        "is_anomaly": [0] * (n - 20) + [1] * 20,
        "anomaly_type": [None] * (n - 20) + ["slow_link"] * 20,
    })
    return df


class TestTimeFeatures:
    def test_adds_expected_columns(self, sample_df):
        result = add_time_features(sample_df)
        assert "hour_sin" in result.columns
        assert "hour_cos" in result.columns
        assert "is_weekend" in result.columns

    def test_cyclical_encoding_range(self, sample_df):
        result = add_time_features(sample_df)
        assert result["hour_sin"].between(-1, 1).all()
        assert result["hour_cos"].between(-1, 1).all()

    def test_does_not_modify_original(self, sample_df):
        original_cols = set(sample_df.columns)
        add_time_features(sample_df)
        assert set(sample_df.columns) == original_cols


class TestRollingFeatures:
    def test_adds_rolling_columns(self, sample_df):
        result = add_rolling_features(sample_df, windows=[5], features=["throughput_mbps"])
        assert "throughput_mbps_rolling_mean_5" in result.columns
        assert "throughput_mbps_rolling_std_5" in result.columns

    def test_multiple_windows(self, sample_df):
        result = add_rolling_features(sample_df, windows=[5, 10])
        for feat in ["throughput_mbps", "latency_ms"]:
            for w in [5, 10]:
                assert f"{feat}_rolling_mean_{w}" in result.columns

    def test_no_nans(self, sample_df):
        result = add_rolling_features(sample_df)
        rolling_cols = [c for c in result.columns if "rolling" in c]
        for col in rolling_cols:
            assert not result[col].isna().any(), f"NaN found in {col}"


class TestLagFeatures:
    def test_adds_lag_columns(self, sample_df):
        result = add_lag_features(sample_df, lags=[1], features=["throughput_mbps"])
        assert "throughput_mbps_lag_1" in result.columns

    def test_no_nans(self, sample_df):
        result = add_lag_features(sample_df)
        lag_cols = [c for c in result.columns if "lag" in c]
        for col in lag_cols:
            assert not result[col].isna().any(), f"NaN found in {col}"


class TestDerivedFeatures:
    def test_adds_derived_columns(self, sample_df):
        result = add_derived_features(sample_df)
        assert "throughput_latency_ratio" in result.columns
        assert "loss_retransmit_product" in result.columns
        assert "jitter_latency_ratio" in result.columns
        assert "zero_throughput" in result.columns


class TestBuildFeatureMatrix:
    def test_output_shapes(self, sample_df):
        X, y, feature_names, scaler = build_feature_matrix(sample_df)
        assert X.shape[0] == len(sample_df)
        assert X.shape[1] == len(feature_names)
        assert y.shape[0] == len(sample_df)
        assert X.dtype == np.float32

    def test_normalized(self, sample_df):
        X, _, _, scaler = build_feature_matrix(sample_df, normalize=True)
        assert scaler is not None
        # After normalization, mean should be ~0, std ~1
        assert abs(X.mean()) < 0.5

    def test_unnormalized(self, sample_df):
        X, _, _, scaler = build_feature_matrix(sample_df, normalize=False)
        assert scaler is None

    def test_no_nans_or_infs(self, sample_df):
        X, _, _, _ = build_feature_matrix(sample_df)
        assert not np.isnan(X).any()
        assert not np.isinf(X).any()

    def test_labels_correct(self, sample_df):
        _, y, _, _ = build_feature_matrix(sample_df)
        assert y.sum() == 20  # 20 anomalies in fixture
