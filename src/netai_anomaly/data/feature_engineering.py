"""Feature engineering pipeline for network telemetry data."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

NUMERIC_FEATURES = [
    "throughput_mbps",
    "latency_ms",
    "packet_loss_pct",
    "retransmits",
    "jitter_ms",
]


def load_dataframe_from_db(db_path: str) -> pd.DataFrame:
    """Load network test data from SQLite into a DataFrame."""
    import sqlite3

    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        "SELECT * FROM network_tests ORDER BY timestamp", conn
    )
    conn.close()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract temporal features from timestamp."""
    df = df.copy()
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    # Cyclical encoding for hour
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    return df


def add_rolling_features(
    df: pd.DataFrame,
    windows: list[int] | None = None,
    features: list[str] | None = None,
) -> pd.DataFrame:
    """Add rolling window statistics for each numeric feature."""
    if windows is None:
        windows = [5, 15, 30]
    if features is None:
        features = NUMERIC_FEATURES

    df = df.copy()
    for feat in features:
        for w in windows:
            col = df[feat]
            df[f"{feat}_rolling_mean_{w}"] = col.rolling(window=w, min_periods=1).mean()
            df[f"{feat}_rolling_std_{w}"] = col.rolling(window=w, min_periods=1).std().fillna(0)

    return df


def add_lag_features(
    df: pd.DataFrame,
    lags: list[int] | None = None,
    features: list[str] | None = None,
) -> pd.DataFrame:
    """Add lag features for temporal dependency modeling."""
    if lags is None:
        lags = [1, 3, 5]
    if features is None:
        features = NUMERIC_FEATURES

    df = df.copy()
    for feat in features:
        for lag in lags:
            df[f"{feat}_lag_{lag}"] = df[feat].shift(lag).fillna(0)

    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived features capturing cross-metric relationships."""
    df = df.copy()

    # Throughput-to-latency ratio (efficiency indicator)
    df["throughput_latency_ratio"] = df["throughput_mbps"] / (df["latency_ms"] + 1e-6)

    # Loss-retransmit interaction
    df["loss_retransmit_product"] = df["packet_loss_pct"] * df["retransmits"]

    # Jitter relative to latency
    df["jitter_latency_ratio"] = df["jitter_ms"] / (df["latency_ms"] + 1e-6)

    # Binary flag for zero throughput (possible test failure)
    df["zero_throughput"] = (df["throughput_mbps"] == 0).astype(int)

    return df


def build_feature_matrix(
    df: pd.DataFrame,
    rolling_windows: list[int] | None = None,
    add_derived: bool = True,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str], StandardScaler | None]:
    """Build the full feature matrix from raw data.

    Args:
        df: Raw DataFrame with network test data.
        rolling_windows: Window sizes for rolling statistics.
        add_derived: Whether to add derived cross-metric features.
        normalize: Whether to standard-scale features.

    Returns:
        Tuple of (feature_matrix, labels, feature_names, scaler).
    """
    df = add_time_features(df)
    df = add_rolling_features(df, windows=rolling_windows)
    df = add_lag_features(df)
    if add_derived:
        df = add_derived_features(df)

    # Select all numeric feature columns (excluding metadata)
    exclude_cols = {
        "id", "timestamp", "source_host", "destination_host",
        "test_type", "is_anomaly", "anomaly_type", "created_at",
        "hour", "day_of_week",
    }
    feature_cols = [
        c for c in df.columns
        if c not in exclude_cols and np.issubdtype(df[c].dtype, np.number)
    ]

    X = df[feature_cols].values.astype(np.float32)
    y = (
        df["is_anomaly"].values.astype(np.int64)
        if "is_anomaly" in df.columns
        else np.zeros(len(df), dtype=np.int64)
    )

    # Handle NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = None
    if normalize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X).astype(np.float32)

    return X, y, feature_cols, scaler
