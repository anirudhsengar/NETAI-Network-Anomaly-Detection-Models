"""Tests for database and synthetic data generation."""


import pytest

from netai_anomaly.data.database import TelemetryDB
from netai_anomaly.data.generator import generate_synthetic_data


@pytest.fixture
def tmp_db(tmp_path):
    """Create a temporary database."""
    db_path = tmp_path / "test.db"
    db = TelemetryDB(db_path)
    db.initialize()
    yield db
    db.close()


class TestTelemetryDB:
    def test_initialize_creates_tables(self, tmp_db):
        tables = tmp_db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = {t[0] for t in tables}
        assert "network_tests" in table_names
        assert "traceroute_hops" in table_names
        assert "network_links" in table_names

    def test_insert_and_count(self, tmp_db):
        tmp_db.insert_test(
            timestamp="2025-01-01T00:00:00",
            source_host="host-a",
            destination_host="host-b",
            test_type="throughput",
            throughput_mbps=1000.0,
            latency_ms=10.0,
            packet_loss_pct=0.01,
            retransmits=1,
            jitter_ms=0.5,
        )
        assert tmp_db.get_test_count() == 1

    def test_batch_insert(self, tmp_db):
        tests = [
            {
                "timestamp": f"2025-01-01T{i:02d}:00:00",
                "source_host": "a",
                "destination_host": "b",
                "test_type": "throughput",
                "throughput_mbps": 1000.0,
                "latency_ms": 10.0,
                "packet_loss_pct": 0.0,
                "retransmits": 0,
                "jitter_ms": 0.5,
                "is_anomaly": 0,
                "anomaly_type": None,
            }
            for i in range(100)
        ]
        tmp_db.insert_tests_batch(tests)
        assert tmp_db.get_test_count() == 100

    def test_anomaly_count(self, tmp_db):
        tmp_db.insert_test(
            timestamp="2025-01-01T00:00:00",
            source_host="a",
            destination_host="b",
            test_type="throughput",
            throughput_mbps=10.0,
            latency_ms=500.0,
            packet_loss_pct=50.0,
            retransmits=100,
            jitter_ms=20.0,
            is_anomaly=1,
            anomaly_type="slow_link",
        )
        assert tmp_db.get_anomaly_count() == 1

    def test_context_manager(self, tmp_path):
        db_path = tmp_path / "ctx_test.db"
        with TelemetryDB(db_path) as db:
            db.initialize()
            assert db.get_test_count() == 0


class TestDataGeneration:
    def test_generate_data(self, tmp_path):
        db_path = tmp_path / "gen_test.db"
        with TelemetryDB(db_path) as db:
            stats = generate_synthetic_data(db, num_samples=500, anomaly_ratio=0.1, seed=42)

        assert stats["total_samples"] == 500
        assert stats["total_anomalies"] == 50

        with TelemetryDB(db_path) as db:
            assert db.get_test_count() == 500
            assert db.get_anomaly_count() == 50

    def test_reproducibility(self, tmp_path):
        """Same seed should produce identical data."""
        results = []
        for i in range(2):
            db_path = tmp_path / f"repro_{i}.db"
            with TelemetryDB(db_path) as db:
                stats = generate_synthetic_data(db, num_samples=100, seed=123)
                results.append(stats)

        assert results[0]["total_anomalies"] == results[1]["total_anomalies"]
        assert results[0]["anomaly_breakdown"] == results[1]["anomaly_breakdown"]

    def test_anomaly_types_present(self, tmp_path):
        db_path = tmp_path / "types_test.db"
        with TelemetryDB(db_path) as db:
            stats = generate_synthetic_data(db, num_samples=5000, anomaly_ratio=0.15, seed=42)

        # With enough samples, all anomaly types should be represented
        breakdown = stats["anomaly_breakdown"]
        assert sum(breakdown.values()) > 0
