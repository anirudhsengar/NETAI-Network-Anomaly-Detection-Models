# Architecture Documentation

## System Architecture

The NETAI anomaly detection system is designed as a modular pipeline that processes raw network telemetry data through feature engineering, trains multiple deep learning models, and serves predictions via a REST API deployable on Kubernetes.

### Data Flow

```
perfSONAR/Traceroute Data
        │
        ▼
┌─────────────────┐
│  SQLite Database │  ← Stores network_tests, traceroute_hops, network_links
│  (Telemetry DB)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Feature       │  ← Rolling stats, lag features, temporal encoding,
│    Engineering   │    derived cross-metric features, normalization
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  PyTorch Dataset │  ← Point-wise (autoencoder) and sequential (LSTM/Transformer)
│  & DataLoaders   │    with time-aware train/val/test splitting
└────────┬────────┘
         │
    ┌────┼────┐
    ▼    ▼    ▼
┌──────┐┌──────┐┌────────────┐
│  AE  ││ LSTM ││ Transformer│  ← Independent training with early stopping
└──┬───┘└──┬───┘└──────┬─────┘
   │       │           │
   └───────┼───────────┘
           ▼
┌─────────────────┐
│    Ensemble      │  ← Normalized weighted average of anomaly scores
│    Detector      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   FastAPI        │  ← /predict, /predict/batch, /health
│   Inference API  │
└─────────────────┘
```

### Database Schema

The SQLite database (`network_telemetry.db`) uses three tables:

- **`network_tests`**: Core telemetry data (throughput, latency, packet loss, retransmits, jitter) with anomaly labels and metadata
- **`traceroute_hops`**: Per-hop traceroute data linked to test records via foreign key
- **`network_links`**: Link capacity and baseline characteristics for host pairs

### Feature Engineering Pipeline

The pipeline creates 59 features from 5 raw metrics:

1. **Raw metrics** (5): Direct measurements from perfSONAR tests
2. **Rolling windows** (30): Mean and standard deviation for windows of 5, 15, 30 samples per metric
3. **Lag features** (15): Historical values at t-1, t-3, t-5 per metric
4. **Temporal** (3): Cyclical hour encoding (sin/cos) and weekend flag
5. **Derived** (4): Cross-metric ratios and interaction terms
6. **Categorical** (2): MTU and TCP window size

All features are standardized (zero mean, unit variance) before model input.

### Model Architecture Details

#### Autoencoder
- **Type**: Symmetric deep autoencoder with batch normalization
- **Anomaly detection**: Reconstruction error (MSE) as anomaly score
- **Rationale**: Normal traffic patterns are compressed efficiently; anomalous patterns have high reconstruction error

#### LSTM Detector
- **Type**: Bidirectional LSTM with FC prediction head
- **Anomaly detection**: Prediction error of the last time step in a sequence window
- **Rationale**: Temporal patterns in normal network behavior are predictable; deviations indicate anomalies

#### Transformer Detector
- **Type**: Transformer encoder with sinusoidal positional encoding
- **Anomaly detection**: Same prediction-error approach as LSTM
- **Rationale**: Self-attention captures both local jitter patterns and long-range periodic behaviors

### Deployment Architecture

The system is designed for NRP's Kubernetes infrastructure:

- **Training**: GPU-enabled Kubernetes Jobs using `nvidia.com/gpu` resource requests
- **Inference**: Stateless deployment (2 replicas) with liveness/readiness probes
- **Storage**: PersistentVolumeClaims for model checkpoints and telemetry data
- **Configuration**: ConfigMap-based YAML configuration injection
