"""Tests for configuration loading."""

import yaml

from netai_anomaly.utils.config import Config, load_config


class TestConfig:
    def test_default_config(self):
        cfg = Config()
        assert cfg.data.batch_size == 64
        assert cfg.models.autoencoder.latent_dim == 8
        assert cfg.training.device == "auto"

    def test_load_yaml_config(self, tmp_path):
        config_data = {
            "data": {"batch_size": 128, "sequence_length": 16},
            "models": {
                "autoencoder": {"latent_dim": 16, "hidden_dims": [128, 64, 32],
                                "dropout": 0.1, "learning_rate": 0.001,
                                "epochs": 50, "threshold_percentile": 95},
                "lstm": {"hidden_dim": 128, "num_layers": 3, "dropout": 0.2,
                         "bidirectional": True, "learning_rate": 0.001,
                         "epochs": 50, "threshold_percentile": 95},
                "transformer": {"d_model": 128, "nhead": 8, "num_encoder_layers": 4,
                                "dim_feedforward": 256, "dropout": 0.1,
                                "learning_rate": 0.0005, "epochs": 50,
                                "threshold_percentile": 95},
                "ensemble": {"weights": {"autoencoder": 0.4, "lstm": 0.3, "transformer": 0.3},
                             "strategy": "weighted_average"},
            },
            "training": {"seed": 123},
            "features": {"rolling_windows": [5, 15, 30],
                         "numeric_features": ["throughput_mbps", "latency_ms",
                                              "packet_loss_pct", "retransmits", "jitter_ms"],
                         "derived_features": True, "normalize": True},
            "inference": {"host": "0.0.0.0", "port": 8000,
                          "model_path": "checkpoints/best_ensemble.pt",
                          "batch_inference": True},
        }
        config_path = tmp_path / "test_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        cfg = load_config(config_path)
        assert cfg.data.batch_size == 128
        assert cfg.data.sequence_length == 16
        assert cfg.models.autoencoder.latent_dim == 16
        assert cfg.models.lstm.hidden_dim == 128
        assert cfg.training.seed == 123

    def test_load_nonexistent_returns_defaults(self):
        cfg = load_config("nonexistent.yaml")
        assert cfg.data.batch_size == 64

    def test_load_actual_config(self):
        cfg = load_config("configs/default.yaml")
        assert cfg.data.batch_size == 64
        assert cfg.models.transformer.nhead == 4
