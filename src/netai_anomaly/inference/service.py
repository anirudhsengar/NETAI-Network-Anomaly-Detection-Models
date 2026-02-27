"""FastAPI-based real-time inference service for anomaly detection."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from netai_anomaly.models.autoencoder import Autoencoder
from netai_anomaly.models.ensemble import EnsembleDetector
from netai_anomaly.models.lstm_detector import LSTMDetector
from netai_anomaly.models.transformer_detector import TransformerDetector

logger = logging.getLogger(__name__)

app = FastAPI(
    title="NETAI Anomaly Detection API",
    description="Real-time network anomaly detection using deep learning ensemble",
    version="0.1.0",
)


class NetworkSample(BaseModel):
    """Single network telemetry sample for inference."""

    throughput_mbps: float = Field(..., description="Throughput in Mbps")
    latency_ms: float = Field(..., description="Latency in milliseconds")
    packet_loss_pct: float = Field(..., ge=0, le=100, description="Packet loss percentage")
    retransmits: int = Field(..., ge=0, description="Number of retransmissions")
    jitter_ms: float = Field(..., ge=0, description="Jitter in milliseconds")


class BatchRequest(BaseModel):
    """Batch of network telemetry samples."""

    samples: list[NetworkSample]


class AnomalyResult(BaseModel):
    """Anomaly detection result for a single sample."""

    is_anomaly: bool
    anomaly_score: float
    model_scores: dict[str, float]


class BatchResponse(BaseModel):
    """Batch anomaly detection results."""

    results: list[AnomalyResult]
    total_samples: int
    anomalies_detected: int


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    device: str


# Global model holder
_ensemble: EnsembleDetector | None = None
_input_dim: int = 5
_device: str = "cpu"


def load_model(checkpoint_path: str, input_dim: int = 5, device: str = "cpu") -> EnsembleDetector:
    """Load ensemble model from checkpoint."""
    global _ensemble, _input_dim, _device
    _device = device
    _input_dim = input_dim

    ae = Autoencoder(input_dim=input_dim)
    lstm = LSTMDetector(input_dim=input_dim)
    transformer = TransformerDetector(input_dim=input_dim)

    _ensemble = EnsembleDetector(
        autoencoder=ae,
        lstm=lstm,
        transformer=transformer,
        device=device,
    )

    path = Path(checkpoint_path)
    if path.exists():
        state = torch.load(path, map_location=device, weights_only=True)
        _ensemble.load_state_dict(state)
        logger.info(f"Loaded ensemble model from {checkpoint_path}")
    else:
        logger.warning(f"No checkpoint at {checkpoint_path}, using untrained model")

    return _ensemble


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=_ensemble is not None,
        device=_device,
    )


@app.post("/predict", response_model=AnomalyResult)
async def predict_single(sample: NetworkSample):
    """Predict anomaly for a single network sample."""
    if _ensemble is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    features = torch.FloatTensor([[
        sample.throughput_mbps,
        sample.latency_ms,
        sample.packet_loss_pct,
        sample.retransmits,
        sample.jitter_ms,
    ]])

    x_seq = features.unsqueeze(1)  # (1, 1, features)

    scores = _ensemble.compute_scores(features, x_seq)
    ensemble_score = _ensemble.ensemble_score(features, x_seq)

    return AnomalyResult(
        is_anomaly=bool(ensemble_score[0] >= _ensemble.ensemble_threshold),
        anomaly_score=float(ensemble_score[0]),
        model_scores={k: float(v[0]) for k, v in scores.items()},
    )


@app.post("/predict/batch", response_model=BatchResponse)
async def predict_batch(request: BatchRequest):
    """Predict anomalies for a batch of network samples."""
    if _ensemble is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not request.samples:
        return BatchResponse(results=[], total_samples=0, anomalies_detected=0)

    features_list = []
    for s in request.samples:
        features_list.append([
            s.throughput_mbps,
            s.latency_ms,
            s.packet_loss_pct,
            s.retransmits,
            s.jitter_ms,
        ])

    features = torch.FloatTensor(features_list)
    x_seq = features.unsqueeze(1)

    scores = _ensemble.compute_scores(features, x_seq)
    ensemble_scores = _ensemble.ensemble_score(features, x_seq)
    predictions = (ensemble_scores >= _ensemble.ensemble_threshold).astype(int)

    results = []
    for i in range(len(request.samples)):
        results.append(AnomalyResult(
            is_anomaly=bool(predictions[i]),
            anomaly_score=float(ensemble_scores[i]),
            model_scores={k: float(v[i]) for k, v in scores.items()},
        ))

    return BatchResponse(
        results=results,
        total_samples=len(results),
        anomalies_detected=int(predictions.sum()),
    )
