"""Tests for training pipeline."""

import numpy as np
import pytest
import torch

from netai_anomaly.data.loader import NetworkTelemetryDataset, create_dataloaders
from netai_anomaly.models.autoencoder import Autoencoder
from netai_anomaly.models.lstm_detector import LSTMDetector
from netai_anomaly.models.transformer_detector import TransformerDetector
from netai_anomaly.training.trainer import EarlyStopping, Trainer, get_device

INPUT_DIM = 15


@pytest.fixture
def dummy_data():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((300, INPUT_DIM)).astype(np.float32)
    y = np.zeros(300, dtype=np.int64)
    y[-30:] = 1  # 10% anomalies
    return X, y


@pytest.fixture
def point_loaders(dummy_data):
    X, y = dummy_data
    ds = NetworkTelemetryDataset(X, y, sequence_length=1)
    return create_dataloaders(ds, batch_size=32)


@pytest.fixture
def seq_loaders(dummy_data):
    X, y = dummy_data
    ds = NetworkTelemetryDataset(X, y, sequence_length=8)
    return create_dataloaders(ds, batch_size=32)


class TestEarlyStopping:
    def test_no_improvement_stops(self):
        es = EarlyStopping(patience=3)
        for _ in range(3):
            assert not es.step(1.0)
        assert es.step(1.0)

    def test_improvement_resets(self):
        es = EarlyStopping(patience=3)
        es.step(1.0)
        es.step(1.0)
        es.step(0.5)  # improvement
        es.step(0.5)
        es.step(0.5)
        assert not es.should_stop

    def test_min_delta(self):
        es = EarlyStopping(patience=3, min_delta=0.1)
        es.step(1.0)      # best=1.0, counter=0
        es.step(0.95)     # not enough improvement (0.95 < 0.9 is False), counter=1
        es.step(0.95)     # counter=2
        assert not es.should_stop
        es.step(0.95)     # counter=3 → stop
        assert es.should_stop


class TestDataset:
    def test_point_dataset_length(self, dummy_data):
        X, y = dummy_data
        ds = NetworkTelemetryDataset(X, y, sequence_length=1)
        assert len(ds) == 300

    def test_seq_dataset_length(self, dummy_data):
        X, y = dummy_data
        ds = NetworkTelemetryDataset(X, y, sequence_length=8)
        assert len(ds) == 300 - 8 + 1

    def test_point_item_shape(self, dummy_data):
        X, y = dummy_data
        ds = NetworkTelemetryDataset(X, y, sequence_length=1)
        x, label = ds[0]
        assert x.shape == (INPUT_DIM,)
        assert label.dim() == 0

    def test_seq_item_shape(self, dummy_data):
        X, y = dummy_data
        ds = NetworkTelemetryDataset(X, y, sequence_length=8)
        x, label = ds[0]
        assert x.shape == (8, INPUT_DIM)
        assert label.dim() == 0


class TestTrainer:
    def test_train_autoencoder(self, point_loaders, tmp_path):
        train_loader, val_loader, _ = point_loaders
        model = Autoencoder(input_dim=INPUT_DIM, hidden_dims=[16, 8], latent_dim=4)
        trainer = Trainer(model, torch.device("cpu"), str(tmp_path))
        history = trainer.train(
            train_loader, val_loader, epochs=3, learning_rate=0.01, model_name="ae_test"
        )
        assert len(history["train_loss"]) == 3
        assert history["best_val_loss"] < float("inf")
        assert (tmp_path / "best_ae_test.pt").exists()

    def test_train_lstm(self, seq_loaders, tmp_path):
        train_loader, val_loader, _ = seq_loaders
        model = LSTMDetector(input_dim=INPUT_DIM, hidden_dim=16, num_layers=1)
        trainer = Trainer(model, torch.device("cpu"), str(tmp_path))
        history = trainer.train(
            train_loader, val_loader, epochs=3, learning_rate=0.01, model_name="lstm_test"
        )
        assert len(history["train_loss"]) == 3

    def test_train_transformer(self, seq_loaders, tmp_path):
        train_loader, val_loader, _ = seq_loaders
        model = TransformerDetector(
            input_dim=INPUT_DIM, d_model=16, nhead=4, num_encoder_layers=1
        )
        trainer = Trainer(model, torch.device("cpu"), str(tmp_path))
        history = trainer.train(
            train_loader, val_loader, epochs=3, learning_rate=0.01, model_name="tf_test"
        )
        assert len(history["train_loss"]) == 3

    def test_evaluate(self, point_loaders, tmp_path):
        _, _, test_loader = point_loaders
        model = Autoencoder(input_dim=INPUT_DIM, hidden_dims=[16, 8], latent_dim=4)
        trainer = Trainer(model, torch.device("cpu"), str(tmp_path))
        metrics = trainer.evaluate(test_loader)
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "threshold" in metrics

    def test_get_device(self):
        device = get_device("cpu")
        assert device == torch.device("cpu")
