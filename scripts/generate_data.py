#!/usr/bin/env python3
"""Generate synthetic perfSONAR-like network telemetry data."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from netai_anomaly.data.database import TelemetryDB
from netai_anomaly.data.generator import generate_synthetic_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic network telemetry data")
    parser.add_argument("--db-path", default="data/network_telemetry.db", help="SQLite database path")
    parser.add_argument("--num-samples", type=int, default=10000, help="Number of test records")
    parser.add_argument("--anomaly-ratio", type=float, default=0.08, help="Fraction of anomalies")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    db_path = Path(args.db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove existing DB for clean generation
    if db_path.exists():
        db_path.unlink()

    logger.info(f"Generating {args.num_samples} samples (anomaly ratio: {args.anomaly_ratio})")

    with TelemetryDB(db_path) as db:
        stats = generate_synthetic_data(
            db,
            num_samples=args.num_samples,
            anomaly_ratio=args.anomaly_ratio,
            seed=args.seed,
        )

    logger.info(f"Generated data summary:")
    logger.info(f"  Total samples:  {stats['total_samples']}")
    logger.info(f"  Total anomalies: {stats['total_anomalies']}")
    logger.info(f"  Breakdown: {stats['anomaly_breakdown']}")
    logger.info(f"  Database: {db_path}")


if __name__ == "__main__":
    main()
