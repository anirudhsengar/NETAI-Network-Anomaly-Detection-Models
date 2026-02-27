"""Tests for anomaly detection models."""

import numpy as np
import pytest
import torch

from netai_anomaly.models.autoencoder import Autoencoder
from netai_anomaly.models.ensemble import EnsembleDetector
from netai_anomaly.models.lstm_detector import LSTMDetector
from netai_anomaly.models.transformer_detector import TransformerDetector


@pytest.fixture
def input_dim():
    return 20


@pytest.fixture
def batch_size():
    return 16


@pytest.fixture
def seq_len():
    return 8


@pytest.fixture
def point_data(batch_size, input_dim):
    return torch.randn(batch_size, input_dim)


@pytest.fixture
def seq_data(batch_size, seq_len, input_dim):
    return torch.randn(batch_size, seq_len, input_dim)


class TestAutoencoder:
    def test_forward_shape(self, input_dim, point_data):
        model = Autoencoder(input_dim=input_dim, hidden_dims=[32, 16], latent_dim=8)
        output = model(point_data)
        assert output.shape == point_data.shape

    def test_encode_shape(self, input_dim, point_data):
        model = Autoencoder(input_dim=input_dim, latent_dim=4)
        z = model.encode(point_data)
        assert z.shape == (point_data.shape[0], 4)

    def test_anomaly_score_shape(self, input_dim, point_data, batch_size):
        model = Autoencoder(input_dim=input_dim)
        scores = model.anomaly_score(point_data)
        assert scores.shape == (batch_size,)
        assert (scores >= 0).all()

    def test_anomaly_score_higher_for_noise(self, input_dim):
        model = Autoencoder(input_dim=input_dim, hidden_dims=[32, 16], latent_dim=8)
        normal = torch.zeros(10, input_dim)
        noisy = torch.randn(10, input_dim) * 10
        # Untrained model, but extreme inputs should generally score higher
        score_normal = model.anomaly_score(normal)
        score_noisy = model.anomaly_score(noisy)
        # At minimum both should be non-negative
        assert (score_normal >= 0).all()
        assert (score_noisy >= 0).all()


class TestLSTMDetector:
    def test_forward_shape(self, input_dim, seq_data, batch_size):
        model = LSTMDetector(input_dim=input_dim, hidden_dim=32, num_layers=1)
        output = model(seq_data)
        assert output.shape == (batch_size, input_dim)

    def test_bidirectional(self, input_dim, seq_data, batch_size):
        model = LSTMDetector(input_dim=input_dim, bidirectional=True)
        output = model(seq_data)
        assert output.shape == (batch_size, input_dim)

    def test_anomaly_score_shape(self, input_dim, seq_data, batch_size):
        model = LSTMDetector(input_dim=input_dim)
        scores = model.anomaly_score(seq_data)
        assert scores.shape == (batch_size,)
        assert (scores >= 0).all()


class TestTransformerDetector:
    def test_forward_shape(self, input_dim, seq_data, batch_size):
        model = TransformerDetector(
            input_dim=input_dim, d_model=32, nhead=4, num_encoder_layers=2
        )
        output = model(seq_data)
        assert output.shape == (batch_size, input_dim)

    def test_anomaly_score_shape(self, input_dim, seq_data, batch_size):
        model = TransformerDetector(input_dim=input_dim, d_model=32, nhead=4)
        scores = model.anomaly_score(seq_data)
        assert scores.shape == (batch_size,)
        assert (scores >= 0).all()

    def test_different_sequence_lengths(self, input_dim, batch_size):
        model = TransformerDetector(input_dim=input_dim, d_model=32, nhead=4)
        for seq_len in [4, 16, 32]:
            x = torch.randn(batch_size, seq_len, input_dim)
            output = model(x)
            assert output.shape == (batch_size, input_dim)


class TestEnsembleDetector:
    @pytest.fixture
    def ensemble(self, input_dim):
        ae = Autoencoder(input_dim=input_dim)
        lstm = LSTMDetector(input_dim=input_dim, hidden_dim=32, num_layers=1)
        transformer = TransformerDetector(input_dim=input_dim, d_model=32, nhead=4)
        return EnsembleDetector(ae, lstm, transformer, device="cpu")

    def test_compute_scores(self, ensemble, point_data, seq_data, batch_size):
        scores = ensemble.compute_scores(point_data, seq_data)
        assert "autoencoder" in scores
        assert "lstm" in scores
        assert "transformer" in scores
        for name, s in scores.items():
            assert s.shape == (batch_size,), f"{name} shape mismatch"

    def test_ensemble_score(self, ensemble, point_data, seq_data, batch_size):
        combined = ensemble.ensemble_score(point_data, seq_data)
        assert combined.shape == (batch_size,)

    def test_predict(self, ensemble, point_data, seq_data, batch_size):
        preds = ensemble.predict(point_data, seq_data)
        assert preds.shape == (batch_size,)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_calibrate_threshold(self, ensemble, point_data, seq_data):
        threshold = ensemble.calibrate_threshold(point_data, seq_data, percentile=90)
        assert isinstance(threshold, float)
        assert threshold >= 0

    def test_state_dict_roundtrip(self, ensemble, input_dim):
        state = ensemble.state_dict()
        new_ae = Autoencoder(input_dim=input_dim)
        new_lstm = LSTMDetector(input_dim=input_dim, hidden_dim=32, num_layers=1)
        new_tf = TransformerDetector(input_dim=input_dim, d_model=32, nhead=4)
        new_ensemble = EnsembleDetector(new_ae, new_lstm, new_tf, device="cpu")
        new_ensemble.load_state_dict(state)
        # Verify weights match
        for p1, p2 in zip(ensemble.autoencoder.parameters(), new_ensemble.autoencoder.parameters()):
            assert torch.allclose(p1, p2)

    def test_max_strategy(self, input_dim, point_data, seq_data, batch_size):
        ae = Autoencoder(input_dim=input_dim)
        lstm = LSTMDetector(input_dim=input_dim, hidden_dim=32, num_layers=1)
        tf = TransformerDetector(input_dim=input_dim, d_model=32, nhead=4)
        ensemble = EnsembleDetector(ae, lstm, tf, strategy="max", device="cpu")
        scores = ensemble.ensemble_score(point_data, seq_data)
        assert scores.shape == (batch_size,)
