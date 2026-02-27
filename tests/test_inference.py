"""Tests for the inference service API."""

import pytest
from fastapi.testclient import TestClient

from netai_anomaly.inference.service import app, load_model


@pytest.fixture(autouse=True)
def setup_model():
    """Load a fresh untrained model for testing."""
    load_model(checkpoint_path="nonexistent.pt", input_dim=5, device="cpu")


@pytest.fixture
def client():
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_check(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True


class TestPredictEndpoint:
    def test_single_prediction(self, client):
        response = client.post("/predict", json={
            "throughput_mbps": 5000.0,
            "latency_ms": 15.0,
            "packet_loss_pct": 0.01,
            "retransmits": 2,
            "jitter_ms": 1.0,
        })
        assert response.status_code == 200
        data = response.json()
        assert "is_anomaly" in data
        assert "anomaly_score" in data
        assert "model_scores" in data
        assert isinstance(data["is_anomaly"], bool)
        assert isinstance(data["anomaly_score"], float)

    def test_anomalous_sample(self, client):
        # Extreme values that would be anomalous
        response = client.post("/predict", json={
            "throughput_mbps": 0.0,
            "latency_ms": 0.0,
            "packet_loss_pct": 100.0,
            "retransmits": 0,
            "jitter_ms": 0.0,
        })
        assert response.status_code == 200
        data = response.json()
        assert "anomaly_score" in data

    def test_invalid_input(self, client):
        response = client.post("/predict", json={
            "throughput_mbps": 5000.0,
            # Missing required fields
        })
        assert response.status_code == 422


class TestBatchEndpoint:
    def test_batch_prediction(self, client):
        samples = [
            {
                "throughput_mbps": 5000.0,
                "latency_ms": 15.0,
                "packet_loss_pct": 0.01,
                "retransmits": 2,
                "jitter_ms": 1.0,
            },
            {
                "throughput_mbps": 0.0,
                "latency_ms": 0.0,
                "packet_loss_pct": 100.0,
                "retransmits": 0,
                "jitter_ms": 0.0,
            },
        ]
        response = client.post("/predict/batch", json={"samples": samples})
        assert response.status_code == 200
        data = response.json()
        assert data["total_samples"] == 2
        assert len(data["results"]) == 2
        assert "anomalies_detected" in data

    def test_empty_batch(self, client):
        response = client.post("/predict/batch", json={"samples": []})
        assert response.status_code == 200
        data = response.json()
        assert data["total_samples"] == 0
