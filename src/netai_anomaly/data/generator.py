"""Synthetic data generator mimicking perfSONAR network telemetry.

Generates realistic network test data with injected anomalies for:
- Slow links (degraded throughput)
- High packet loss
- Excessive retransmits
- Failed network tests
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta

import numpy as np

from netai_anomaly.data.database import TelemetryDB

# Simulated NRP network hosts
HOSTS = [
    "sdsc-perfsonar.nrp.net",
    "ucsd-perfsonar.nrp.net",
    "uchicago-perfsonar.nrp.net",
    "nebraska-perfsonar.nrp.net",
    "florida-perfsonar.nrp.net",
    "mit-perfsonar.nrp.net",
    "stanford-perfsonar.nrp.net",
    "wisc-perfsonar.nrp.net",
    "tacc-perfsonar.nrp.net",
    "anl-perfsonar.nrp.net",
]

# Typical baseline parameters per link
LINK_PROFILES = {
    "high_bandwidth": {
        "throughput_mean": 8000,
        "throughput_std": 500,
        "latency_mean": 15,
        "latency_std": 3,
        "loss_mean": 0.01,
        "loss_std": 0.005,
        "retransmit_mean": 2,
        "retransmit_std": 1,
        "jitter_mean": 1.0,
        "jitter_std": 0.3,
    },
    "medium_bandwidth": {
        "throughput_mean": 3000,
        "throughput_std": 400,
        "latency_mean": 40,
        "latency_std": 8,
        "loss_mean": 0.05,
        "loss_std": 0.02,
        "retransmit_mean": 5,
        "retransmit_std": 3,
        "jitter_mean": 3.0,
        "jitter_std": 1.0,
    },
    "low_bandwidth": {
        "throughput_mean": 500,
        "throughput_std": 150,
        "latency_mean": 80,
        "latency_std": 15,
        "loss_mean": 0.1,
        "loss_std": 0.05,
        "retransmit_mean": 10,
        "retransmit_std": 5,
        "jitter_mean": 5.0,
        "jitter_std": 2.0,
    },
}

ANOMALY_TYPES = ["slow_link", "high_loss", "excessive_retransmits", "test_failure"]


def _generate_normal_sample(profile: dict, rng: np.random.Generator) -> dict:
    """Generate a normal (non-anomalous) data sample."""
    tp_mean = profile["throughput_mean"]
    tp_std = profile["throughput_std"]
    return {
        "throughput_mbps": max(10.0, rng.normal(tp_mean, tp_std)),
        "latency_ms": max(0.5, rng.normal(
            profile["latency_mean"], profile["latency_std"]
        )),
        "packet_loss_pct": max(0.0, min(100.0, rng.normal(
            profile["loss_mean"], profile["loss_std"]
        ))),
        "retransmits": max(0, int(rng.normal(
            profile["retransmit_mean"], profile["retransmit_std"]
        ))),
        "jitter_ms": max(0.0, rng.normal(
            profile["jitter_mean"], profile["jitter_std"]
        )),
    }


def _inject_anomaly(sample: dict, anomaly_type: str, rng: np.random.Generator) -> dict:
    """Inject an anomaly into a data sample."""
    sample = sample.copy()
    sample["is_anomaly"] = 1
    sample["anomaly_type"] = anomaly_type

    if anomaly_type == "slow_link":
        sample["throughput_mbps"] *= rng.uniform(0.05, 0.3)
        sample["latency_ms"] *= rng.uniform(2.0, 8.0)
    elif anomaly_type == "high_loss":
        sample["packet_loss_pct"] = rng.uniform(5.0, 50.0)
        sample["retransmits"] = int(sample["retransmits"] * rng.uniform(3.0, 15.0))
    elif anomaly_type == "excessive_retransmits":
        sample["retransmits"] = int(rng.uniform(50, 500))
        sample["throughput_mbps"] *= rng.uniform(0.3, 0.7)
    elif anomaly_type == "test_failure":
        sample["throughput_mbps"] = 0.0
        sample["latency_ms"] = 0.0
        sample["packet_loss_pct"] = 100.0
        sample["retransmits"] = 0
        sample["jitter_ms"] = 0.0

    return sample


def generate_synthetic_data(
    db: TelemetryDB,
    num_samples: int = 10000,
    anomaly_ratio: float = 0.08,
    seed: int = 42,
) -> dict[str, int]:
    """Generate synthetic perfSONAR-like network telemetry data.

    Args:
        db: Database instance to write to.
        num_samples: Total number of test records to generate.
        anomaly_ratio: Fraction of samples that are anomalous.
        seed: Random seed for reproducibility.

    Returns:
        Summary statistics of generated data.
    """
    rng = np.random.default_rng(seed)
    random.seed(seed)

    db.initialize()

    # Assign link profiles to host pairs
    host_pairs = []
    profiles_list = list(LINK_PROFILES.keys())
    for i, src in enumerate(HOSTS):
        for j, dst in enumerate(HOSTS):
            if i != j:
                profile_name = profiles_list[(i + j) % len(profiles_list)]
                host_pairs.append((src, dst, profile_name))

    num_anomalies = int(num_samples * anomaly_ratio)
    anomaly_indices = set(rng.choice(num_samples, size=num_anomalies, replace=False))

    start_time = datetime(2025, 1, 1)
    batch = []
    anomaly_counts = {t: 0 for t in ANOMALY_TYPES}

    for i in range(num_samples):
        src, dst, profile_name = host_pairs[i % len(host_pairs)]
        profile = LINK_PROFILES[profile_name]

        # Advance time with some jitter (tests run every ~5 min on average)
        timestamp = start_time + timedelta(
            minutes=int(i * 5 + rng.integers(-2, 3)),
            seconds=int(rng.integers(0, 60)),
        )

        sample = _generate_normal_sample(profile, rng)
        sample.update(
            {
                "timestamp": timestamp.isoformat(),
                "source_host": src,
                "destination_host": dst,
                "test_type": rng.choice(["throughput", "latency", "trace"]),
                "mtu": int(rng.choice([1500, 9000])),
                "tcp_window_size": int(rng.choice([65536, 131072, 262144, 524288])),
                "is_anomaly": 0,
                "anomaly_type": None,
            }
        )

        if i in anomaly_indices:
            atype = rng.choice(ANOMALY_TYPES)
            sample = _inject_anomaly(sample, atype, rng)
            anomaly_counts[atype] += 1

        batch.append(sample)

        # Batch insert every 1000 records
        if len(batch) >= 1000:
            db.insert_tests_batch(batch)
            batch = []

    if batch:
        db.insert_tests_batch(batch)

    return {
        "total_samples": num_samples,
        "total_anomalies": num_anomalies,
        "anomaly_breakdown": anomaly_counts,
    }
