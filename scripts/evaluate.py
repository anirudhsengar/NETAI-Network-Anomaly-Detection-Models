#!/usr/bin/env python3
"""Evaluate trained models on test data."""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from netai_anomaly.data.feature_engineering import build_feature_matrix, load_dataframe_from_db
from netai_anomaly.data.loader import NetworkTelemetryDataset, create_data_splits
from netai_anomaly.models.autoencoder import Autoencoder
from netai_anomaly.models.ensemble import EnsembleDetector
from netai_anomaly.models.lstm_detector import LSTMDetector
from netai_anomaly.models.transformer_detector import TransformerDetector
from netai_anomaly.training.trainer import get_device
from netai_anomaly.utils.config import load_config
from netai_anomaly.utils.metrics import compute_anomaly_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate anomaly detection models")
    parser.add_argument("--config", default="configs/default.yaml", help="Config file path")
    parser.add_argument("--output", default=None, help="Output JSON path for metrics")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg.training.device)
    checkpoint_dir = Path(cfg.training.checkpoint_dir)

    # Load metadata
    metadata_path = checkpoint_dir / "metadata.pt"
    if not metadata_path.exists():
        logger.error("No metadata found. Run training first.")
        sys.exit(1)

    metadata = torch.load(metadata_path, weights_only=False)
    input_dim = metadata["input_dim"]
    seq_len = metadata["sequence_length"]

    # Load data
    df = load_dataframe_from_db(cfg.data.database_path)
    X, y, _, _ = build_feature_matrix(
        df,
        rolling_windows=cfg.features.rolling_windows,
        add_derived=cfg.features.derived_features,
        normalize=cfg.features.normalize,
    )

    # Build ensemble
    ae = Autoencoder(input_dim=input_dim, hidden_dims=cfg.models.autoencoder.hidden_dims,
                     latent_dim=cfg.models.autoencoder.latent_dim)
    lstm = LSTMDetector(input_dim=input_dim, hidden_dim=cfg.models.lstm.hidden_dim,
                        num_layers=cfg.models.lstm.num_layers, bidirectional=cfg.models.lstm.bidirectional)
    tf = TransformerDetector(input_dim=input_dim, d_model=cfg.models.transformer.d_model,
                             nhead=cfg.models.transformer.nhead,
                             num_encoder_layers=cfg.models.transformer.num_encoder_layers)

    ensemble = EnsembleDetector(ae, lstm, tf, device=str(device))

    ensemble_path = checkpoint_dir / "best_ensemble.pt"
    if ensemble_path.exists():
        state = torch.load(ensemble_path, map_location=device, weights_only=True)
        ensemble.load_state_dict(state)
    else:
        logger.error("No ensemble checkpoint found.")
        sys.exit(1)

    # Get test split
    seq_dataset = NetworkTelemetryDataset(X, y, sequence_length=seq_len)
    _, _, test_set = create_data_splits(seq_dataset, cfg.data.train_split, cfg.data.val_split)

    # Collect test data
    all_scores = []
    all_labels = []
    for i in range(len(test_set)):
        x_seq, label = test_set[i]
        x_seq = x_seq.unsqueeze(0)
        x_point = x_seq[:, -1, :]
        score = ensemble.ensemble_score(x_point, x_seq)
        all_scores.append(score[0])
        all_labels.append(label.item())

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    metrics = compute_anomaly_metrics(all_labels, all_scores)

    logger.info("=" * 60)
    logger.info("Ensemble Evaluation Results")
    logger.info("=" * 60)
    for k, v in metrics.items():
        logger.info(f"  {k:>15s}: {v:.4f}")

    # Also print per-model metrics from training
    logger.info("\nPer-model metrics (from training):")
    for model_name in ["autoencoder", "lstm", "transformer"]:
        m = metadata.get(f"{model_name}_metrics", {})
        logger.info(f"  {model_name}: F1={m.get('f1', 0):.4f}, Precision={m.get('precision', 0):.4f}, Recall={m.get('recall', 0):.4f}")

    if args.output:
        output = {
            "ensemble": metrics,
            "autoencoder": metadata.get("autoencoder_metrics", {}),
            "lstm": metadata.get("lstm_metrics", {}),
            "transformer": metadata.get("transformer_metrics", {}),
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        logger.info(f"Metrics saved to {args.output}")


if __name__ == "__main__":
    main()
