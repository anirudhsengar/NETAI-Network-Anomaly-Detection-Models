#!/usr/bin/env python3
"""Train anomaly detection models on network telemetry data."""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from netai_anomaly.data.feature_engineering import build_feature_matrix, load_dataframe_from_db
from netai_anomaly.data.loader import NetworkTelemetryDataset, create_dataloaders
from netai_anomaly.models.autoencoder import Autoencoder
from netai_anomaly.models.ensemble import EnsembleDetector
from netai_anomaly.models.lstm_detector import LSTMDetector
from netai_anomaly.models.transformer_detector import TransformerDetector
from netai_anomaly.training.trainer import Trainer, get_device
from netai_anomaly.utils.config import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train network anomaly detection models")
    parser.add_argument("--config", default="configs/default.yaml", help="Config file path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg.training.device)
    logger.info(f"Using device: {device}")

    torch.manual_seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)

    # Load and engineer features
    logger.info(f"Loading data from {cfg.data.database_path}")
    df = load_dataframe_from_db(cfg.data.database_path)
    logger.info(f"Loaded {len(df)} records ({df['is_anomaly'].sum()} anomalies)")

    X, y, feature_names, scaler = build_feature_matrix(
        df,
        rolling_windows=cfg.features.rolling_windows,
        add_derived=cfg.features.derived_features,
        normalize=cfg.features.normalize,
    )
    logger.info(f"Feature matrix: {X.shape} with {len(feature_names)} features")

    input_dim = X.shape[1]
    seq_len = cfg.data.sequence_length

    # Create datasets for both point-wise and sequence models
    point_dataset = NetworkTelemetryDataset(X, y, sequence_length=1)
    seq_dataset = NetworkTelemetryDataset(X, y, sequence_length=seq_len)

    point_train, point_val, point_test = create_dataloaders(
        point_dataset,
        train_ratio=cfg.data.train_split,
        val_ratio=cfg.data.val_split,
        batch_size=cfg.data.batch_size,
    )
    seq_train, seq_val, seq_test = create_dataloaders(
        seq_dataset,
        train_ratio=cfg.data.train_split,
        val_ratio=cfg.data.val_split,
        batch_size=cfg.data.batch_size,
    )

    checkpoint_dir = cfg.training.checkpoint_dir

    # --- Train Autoencoder ---
    logger.info("=" * 60)
    logger.info("Training Autoencoder")
    ae_model = Autoencoder(
        input_dim=input_dim,
        hidden_dims=cfg.models.autoencoder.hidden_dims,
        latent_dim=cfg.models.autoencoder.latent_dim,
        dropout=cfg.models.autoencoder.dropout,
    )
    ae_trainer = Trainer(ae_model, device, checkpoint_dir, cfg.training.log_interval)
    ae_history = ae_trainer.train(
        point_train, point_val,
        epochs=cfg.models.autoencoder.epochs,
        learning_rate=cfg.models.autoencoder.learning_rate,
        patience=cfg.training.early_stopping_patience,
        model_name="autoencoder",
    )
    ae_metrics = ae_trainer.evaluate(point_test, cfg.models.autoencoder.threshold_percentile)
    logger.info(f"Autoencoder metrics: {ae_metrics}")

    # --- Train LSTM ---
    logger.info("=" * 60)
    logger.info("Training LSTM Detector")
    lstm_model = LSTMDetector(
        input_dim=input_dim,
        hidden_dim=cfg.models.lstm.hidden_dim,
        num_layers=cfg.models.lstm.num_layers,
        dropout=cfg.models.lstm.dropout,
        bidirectional=cfg.models.lstm.bidirectional,
    )
    lstm_trainer = Trainer(lstm_model, device, checkpoint_dir, cfg.training.log_interval)
    lstm_history = lstm_trainer.train(
        seq_train, seq_val,
        epochs=cfg.models.lstm.epochs,
        learning_rate=cfg.models.lstm.learning_rate,
        patience=cfg.training.early_stopping_patience,
        model_name="lstm",
    )
    lstm_metrics = lstm_trainer.evaluate(seq_test, cfg.models.lstm.threshold_percentile)
    logger.info(f"LSTM metrics: {lstm_metrics}")

    # --- Train Transformer ---
    logger.info("=" * 60)
    logger.info("Training Transformer Detector")
    tf_model = TransformerDetector(
        input_dim=input_dim,
        d_model=cfg.models.transformer.d_model,
        nhead=cfg.models.transformer.nhead,
        num_encoder_layers=cfg.models.transformer.num_encoder_layers,
        dim_feedforward=cfg.models.transformer.dim_feedforward,
        dropout=cfg.models.transformer.dropout,
    )
    tf_trainer = Trainer(tf_model, device, checkpoint_dir, cfg.training.log_interval)
    tf_history = tf_trainer.train(
        seq_train, seq_val,
        epochs=cfg.models.transformer.epochs,
        learning_rate=cfg.models.transformer.learning_rate,
        patience=cfg.training.early_stopping_patience,
        model_name="transformer",
    )
    tf_metrics = tf_trainer.evaluate(seq_test, cfg.models.transformer.threshold_percentile)
    logger.info(f"Transformer metrics: {tf_metrics}")

    # --- Build and save ensemble ---
    logger.info("=" * 60)
    logger.info("Building Ensemble Detector")
    ensemble = EnsembleDetector(
        autoencoder=ae_model,
        lstm=lstm_model,
        transformer=tf_model,
        weights=cfg.models.ensemble.weights,
        strategy=cfg.models.ensemble.strategy,
        device=str(device),
    )

    # Save ensemble checkpoint
    ensemble_path = Path(checkpoint_dir) / "best_ensemble.pt"
    torch.save(ensemble.state_dict(), ensemble_path)
    logger.info(f"Ensemble saved to {ensemble_path}")

    # Save metadata
    metadata = {
        "input_dim": input_dim,
        "feature_names": feature_names,
        "sequence_length": seq_len,
        "autoencoder_metrics": ae_metrics,
        "lstm_metrics": lstm_metrics,
        "transformer_metrics": tf_metrics,
        "config": {
            "autoencoder": vars(cfg.models.autoencoder),
            "lstm": vars(cfg.models.lstm),
            "transformer": vars(cfg.models.transformer),
        },
    }
    torch.save(metadata, Path(checkpoint_dir) / "metadata.pt")

    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info(f"  Autoencoder F1: {ae_metrics['f1']:.4f}")
    logger.info(f"  LSTM F1:        {lstm_metrics['f1']:.4f}")
    logger.info(f"  Transformer F1: {tf_metrics['f1']:.4f}")


if __name__ == "__main__":
    main()
