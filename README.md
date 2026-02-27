# NETAI: Network Anomaly Detection Models

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-66%20passed-brightgreen.svg)](#testing)

Deep learning models for **network anomaly detection** using perfSONAR and traceroute telemetry data from the [National Research Platform (NRP)](https://nrp.ai/). This project implements autoencoder, LSTM, and transformer architectures to automatically identify slow links, high packet loss, excessive retransmits, and failed network tests.

> **GSoC 2026 Prototype** — Developed as a practical demonstration for the NETAI / Network Anomaly Detection Models sub-project under the mentorship of Dmitry Mishin and Derek Weitzel.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Data Generation](#1-data-generation)
  - [Training](#2-training)
  - [Evaluation](#3-evaluation)
  - [Inference API](#4-inference-api)
- [Models](#models)
- [Feature Engineering](#feature-engineering)
- [Kubernetes Deployment](#kubernetes-deployment)
- [Testing](#testing)
- [Configuration](#configuration)
- [Technical Skills Demonstrated](#technical-skills-demonstrated)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                   Inference API (FastAPI)                │
│              POST /predict  POST /predict/batch          │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│                  Ensemble Detector                       │
│         (Weighted Average / Max Strategy)                │
├─────────────┬─────────────────┬─────────────────────────┤
│ Autoencoder │  LSTM Detector  │ Transformer Detector    │
│ (Recon MSE) │  (Seq Predict)  │ (Self-Attention)        │
└─────────────┴─────────────────┴─────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│              Feature Engineering Pipeline                │
│  Rolling Stats │ Lag Features │ Derived │ Normalization  │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│            SQLite Network Telemetry Database             │
│   perfSONAR throughput, latency, loss, retransmits,     │
│   jitter, traceroute hops, network link metadata         │
└─────────────────────────────────────────────────────────┘
```

## Features

- **Three anomaly detection architectures**: Autoencoder, Bidirectional LSTM, and Transformer encoder
- **Ensemble model** with configurable weighted averaging or max-score strategies
- **Comprehensive feature engineering**: rolling window statistics, lag features, temporal encoding, cross-metric derived features
- **SQLite database** schema matching perfSONAR telemetry structure
- **Synthetic data generator** producing realistic network test data with four anomaly types
- **FastAPI inference service** with single and batch prediction endpoints
- **Kubernetes-ready** with Deployment, Service, GPU training Job, PVC, and ConfigMap manifests
- **GPU-accelerated training** with automatic device detection (CUDA, MPS, CPU)
- **Early stopping, checkpointing**, and learning rate scheduling
- **66 comprehensive tests** covering data, features, models, training, inference, and configuration

## Project Structure

```
├── src/netai_anomaly/
│   ├── data/
│   │   ├── database.py            # SQLite schema & TelemetryDB interface
│   │   ├── feature_engineering.py # Feature pipeline (rolling, lag, derived)
│   │   ├── generator.py           # Synthetic perfSONAR data generator
│   │   └── loader.py              # PyTorch Dataset & DataLoader
│   ├── models/
│   │   ├── autoencoder.py         # Deep autoencoder (reconstruction-based)
│   │   ├── lstm_detector.py       # Bidirectional LSTM (sequence prediction)
│   │   ├── transformer_detector.py# Transformer encoder (self-attention)
│   │   └── ensemble.py            # Ensemble combining all three models
│   ├── training/
│   │   └── trainer.py             # Training loop, early stopping, evaluation
│   ├── inference/
│   │   └── service.py             # FastAPI REST API for real-time inference
│   └── utils/
│       ├── config.py              # YAML configuration management
│       └── metrics.py             # Anomaly detection evaluation metrics
├── tests/                         # 66 comprehensive unit tests
├── scripts/                       # CLI entry points (generate, train, evaluate, serve)
├── configs/default.yaml           # Full configuration file
├── k8s/                           # Kubernetes manifests
│   ├── deployment.yaml            # Inference deployment (2 replicas)
│   ├── service.yaml               # ClusterIP service
│   ├── gpu-training-job.yaml      # GPU-enabled training Job
│   ├── configmap.yaml             # Configuration ConfigMap
│   └── pvc.yaml                   # Persistent Volume Claims
├── docker/
│   ├── Dockerfile                 # CPU inference image
│   └── Dockerfile.gpu             # GPU training image (CUDA 12.1)
├── pyproject.toml                 # Package configuration
├── Makefile                       # Build automation
└── requirements.txt               # Dependencies
```

## Quick Start

```bash
# Clone the repository
git clone https://github.com/anirudh/NETAI-Network-Anomaly-Detection-Models.git
cd NETAI-Network-Anomaly-Detection-Models

# Create virtual environment and install
python3 -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu  # or full torch for GPU
pip install -e ".[dev]"

# Generate synthetic data, train, and serve
python scripts/generate_data.py
python scripts/train.py
python scripts/serve.py
```

## Usage

### 1. Data Generation

Generate synthetic perfSONAR-like network telemetry with configurable anomaly injection:

```bash
python scripts/generate_data.py --num-samples 10000 --anomaly-ratio 0.08
```

This creates a SQLite database (`data/network_telemetry.db`) with:
- **Network test records**: throughput, latency, packet loss, retransmits, jitter
- **Four anomaly types**: slow links, high packet loss, excessive retransmits, test failures
- **Realistic host pairs** mimicking NRP perfSONAR infrastructure

### 2. Training

Train all three models with early stopping and checkpointing:

```bash
python scripts/train.py --config configs/default.yaml
```

The training pipeline:
1. Loads data from SQLite and applies feature engineering (59 features)
2. Trains Autoencoder on point-wise features
3. Trains Bidirectional LSTM on temporal sequences (window=32)
4. Trains Transformer encoder on temporal sequences
5. Saves best checkpoints and ensemble model

### 3. Evaluation

Evaluate the trained ensemble on the held-out test set:

```bash
python scripts/evaluate.py --config configs/default.yaml --output results.json
```

### 4. Inference API

Launch the FastAPI server for real-time anomaly detection:

```bash
python scripts/serve.py --port 8000
```

**Endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check with model status |
| `/predict` | POST | Single sample anomaly detection |
| `/predict/batch` | POST | Batch anomaly detection |

**Example request:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "throughput_mbps": 5000.0,
    "latency_ms": 15.0,
    "packet_loss_pct": 0.01,
    "retransmits": 2,
    "jitter_ms": 1.0
  }'
```

**Example response:**

```json
{
  "is_anomaly": false,
  "anomaly_score": 0.12,
  "model_scores": {
    "autoencoder": 0.08,
    "lstm": 0.15,
    "transformer": 0.11
  }
}
```

## Models

### Autoencoder
Reconstruction-based anomaly detection. The encoder compresses input features into a low-dimensional latent space, and the decoder reconstructs them. Anomaly score = MSE between input and reconstruction.

| Layer | Dimensions |
|-------|-----------|
| Encoder | input → 64 → 32 → 16 → 8 (latent) |
| Decoder | 8 → 16 → 32 → 64 → input |

### LSTM Detector
Bidirectional LSTM that processes temporal sequences of network metrics. Predicts the next time step; anomaly score = prediction error of the last step in the sequence.

| Parameter | Value |
|-----------|-------|
| Hidden dim | 64 |
| Layers | 2 |
| Bidirectional | Yes |
| Sequence length | 32 |

### Transformer Detector
Transformer encoder with sinusoidal positional encoding and self-attention. Captures both local and long-range temporal dependencies in network traffic patterns.

| Parameter | Value |
|-----------|-------|
| d_model | 64 |
| Attention heads | 4 |
| Encoder layers | 3 |
| FFN dim | 128 |

### Ensemble
Combines all three models using normalized, weighted anomaly scores. Supports `weighted_average` and `max` strategies with configurable per-model weights.

## Feature Engineering

The pipeline transforms raw telemetry into 59 engineered features:

| Category | Features | Description |
|----------|----------|-------------|
| **Raw metrics** | 5 | throughput, latency, loss, retransmits, jitter |
| **Rolling statistics** | 30 | Mean and std over windows of 5, 15, 30 for each metric |
| **Lag features** | 15 | Values at t-1, t-3, t-5 for each metric |
| **Temporal** | 3 | hour_sin, hour_cos, is_weekend |
| **Derived** | 4 | throughput/latency ratio, loss×retransmits, jitter/latency, zero_throughput flag |
| **Categorical** | 2 | mtu, tcp_window_size |

## Kubernetes Deployment

Deploy on NRP's Kubernetes infrastructure:

```bash
# Create namespace
kubectl create namespace netai

# Deploy configuration and storage
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/pvc.yaml

# Run GPU training job
kubectl apply -f k8s/gpu-training-job.yaml

# Deploy inference service
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

**Docker images:**

```bash
# CPU inference
docker build -f docker/Dockerfile -t netai-anomaly:latest .

# GPU training (CUDA 12.1)
docker build -f docker/Dockerfile.gpu -t netai-anomaly:gpu .
```

## Testing

```bash
# Run all 66 tests
python -m pytest tests/ -v

# With coverage report
python -m pytest tests/ -v --cov=netai_anomaly --cov-report=term-missing

# Run specific test module
python -m pytest tests/test_models.py -v
```

**Test coverage:**
- `test_data.py` — Database operations, synthetic data generation, reproducibility
- `test_feature_engineering.py` — Time features, rolling stats, lag features, derived features
- `test_models.py` — All model architectures, shapes, scores, ensemble roundtrip
- `test_training.py` — Early stopping, datasets, training loop, evaluation
- `test_inference.py` — API endpoints (health, predict, batch, validation)
- `test_config.py` — YAML loading, defaults, overrides
- `test_metrics.py` — Precision, recall, F1, ROC-AUC, threshold optimization

## Configuration

All parameters are configurable via `configs/default.yaml`:

```yaml
data:
  database_path: "data/network_telemetry.db"
  sequence_length: 32
  batch_size: 64

models:
  autoencoder:
    hidden_dims: [64, 32, 16]
    latent_dim: 8
  lstm:
    hidden_dim: 64
    num_layers: 2
    bidirectional: true
  transformer:
    d_model: 64
    nhead: 4
    num_encoder_layers: 3

training:
  device: "auto"  # auto-detects CUDA/MPS/CPU
  early_stopping_patience: 10
```

## Technical Skills Demonstrated

| Skill | Implementation |
|-------|---------------|
| **Python** | Full package with `pyproject.toml`, type hints, dataclasses |
| **PyTorch** | Autoencoder, LSTM, Transformer architectures from scratch |
| **scikit-learn** | StandardScaler, evaluation metrics (precision, recall, F1, ROC-AUC) |
| **Pandas/NumPy** | Feature engineering pipeline, data manipulation |
| **SQLite** | Database schema, batch operations, telemetry storage |
| **FastAPI** | REST API with Pydantic models, health checks, batch inference |
| **Kubernetes** | Deployment, Service, GPU Job, ConfigMap, PVC manifests |
| **Docker** | Multi-stage builds, CPU and GPU images |
| **MLOps** | Early stopping, checkpointing, LR scheduling, model serialization |
| **Testing** | 66 pytest tests with fixtures, parametric coverage |
| **GPU Computing** | Auto device detection, CUDA/MPS support, GPU training jobs |

---

*Built for GSoC 2026 — NETAI / Network Anomaly Detection Models*
*National Research Platform (NRP) · perfSONAR · Kubernetes*
