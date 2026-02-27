# Model Documentation

## Anomaly Types Detected

| Anomaly Type | Description | Key Indicators |
|-------------|-------------|----------------|
| **Slow Link** | Degraded throughput with high latency | throughput ×0.05–0.3, latency ×2–8 |
| **High Packet Loss** | Excessive packet loss causing retransmissions | loss 5–50%, retransmits ×3–15 |
| **Excessive Retransmits** | TCP retransmission storms | retransmits 50–500, throughput ×0.3–0.7 |
| **Test Failure** | Complete test failure | throughput=0, loss=100% |

## Autoencoder

### Architecture
```
Input (59) → [Linear(64) → BN → ReLU → Dropout] → [Linear(32) → BN → ReLU → Dropout]
           → [Linear(16) → BN → ReLU → Dropout] → Linear(8) [Latent Space]
           → [Linear(16) → BN → ReLU → Dropout] → [Linear(32) → BN → ReLU → Dropout]
           → [Linear(64) → BN → ReLU → Dropout] → Linear(59) [Reconstruction]
```

### How It Works
The autoencoder is trained to reconstruct normal network traffic patterns. During training, it learns a compressed representation of typical metric distributions. At inference, anomalous samples produce higher reconstruction error because the model has not learned to encode their unusual patterns.

### Hyperparameters
- Hidden dimensions: [64, 32, 16]
- Latent dimension: 8
- Dropout: 0.1
- Learning rate: 0.001 (Adam, with ReduceLROnPlateau)

## LSTM Detector

### Architecture
```
Input (batch, 32, 59) → Bidirectional LSTM (2 layers, hidden=64)
                       → Last time step output (128-dim)
                       → Linear(64) → ReLU → Dropout → Linear(59) [Prediction]
```

### How It Works
The bidirectional LSTM processes sliding windows of 32 consecutive network measurements. It learns temporal patterns in normal traffic and predicts the final measurement in each window. Anomalous sequences produce predictions that deviate significantly from the actual values.

### Hyperparameters
- Hidden dimension: 64
- Layers: 2 (bidirectional)
- Dropout: 0.2
- Sequence length: 32
- Learning rate: 0.001

## Transformer Detector

### Architecture
```
Input (batch, 32, 59) → Linear(64) [Projection] → Positional Encoding
                       → Transformer Encoder (3 layers, 4 heads, FFN=128)
                       → Last position encoding
                       → Linear(128) → ReLU → Dropout → Linear(59) [Prediction]
```

### How It Works
The transformer encoder uses self-attention to capture dependencies across the full sequence window simultaneously. Unlike LSTM which processes sequentially, the transformer can directly model relationships between any two time steps. The `norm_first` architecture (pre-norm) provides more stable training.

### Hyperparameters
- Model dimension: 64
- Attention heads: 4
- Encoder layers: 3
- FFN dimension: 128
- Dropout: 0.1
- Learning rate: 0.0005

## Ensemble Strategy

The ensemble normalizes each model's anomaly scores to [0, 1] using min-max scaling, then combines them:

**Weighted Average** (default):
```
score = 0.33 × AE_score + 0.34 × LSTM_score + 0.33 × Transformer_score
```

**Max Strategy**:
```
score = max(AE_score, LSTM_score, Transformer_score)
```

The ensemble threshold is calibrated using the 95th percentile of scores on validation data.

## Evaluation Metrics

All models are evaluated using:
- **Precision**: Fraction of flagged samples that are truly anomalous
- **Recall**: Fraction of actual anomalies that are detected
- **F1 Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the receiver operating characteristic curve
- **Optimal threshold**: Determined by maximizing F1 on the precision-recall curve
