"""Ensemble anomaly detector combining multiple model scores.

Supports weighted averaging and max-score strategies to combine
anomaly scores from autoencoder, LSTM, and transformer models.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch

from netai_anomaly.models.autoencoder import Autoencoder
from netai_anomaly.models.lstm_detector import LSTMDetector
from netai_anomaly.models.transformer_detector import TransformerDetector


class EnsembleDetector:
    """Ensemble combining autoencoder, LSTM, and transformer anomaly detectors."""

    def __init__(
        self,
        autoencoder: Autoencoder,
        lstm: LSTMDetector,
        transformer: TransformerDetector,
        weights: dict[str, float] | None = None,
        strategy: Literal["weighted_average", "max"] = "weighted_average",
        device: str = "cpu",
    ):
        self.autoencoder = autoencoder.to(device)
        self.lstm = lstm.to(device)
        self.transformer = transformer.to(device)
        self.device = device

        if weights is None:
            weights = {"autoencoder": 0.33, "lstm": 0.34, "transformer": 0.33}
        self.weights = weights
        self.strategy = strategy

        # Thresholds set during calibration
        self.thresholds: dict[str, float] = {}
        self.ensemble_threshold: float = 0.5

    def compute_scores(
        self,
        x_point: torch.Tensor,
        x_seq: torch.Tensor,
    ) -> dict[str, np.ndarray]:
        """Compute anomaly scores from all models.

        Args:
            x_point: Point-wise features (batch, features) for autoencoder.
            x_seq: Sequential features (batch, seq_len, features) for LSTM/transformer.

        Returns:
            Dictionary mapping model name to anomaly scores.
        """
        x_point = x_point.to(self.device)
        x_seq = x_seq.to(self.device)

        scores = {
            "autoencoder": self.autoencoder.anomaly_score(x_point).cpu().numpy(),
            "lstm": self.lstm.anomaly_score(x_seq).cpu().numpy(),
            "transformer": self.transformer.anomaly_score(x_seq).cpu().numpy(),
        }
        return scores

    def ensemble_score(
        self,
        x_point: torch.Tensor,
        x_seq: torch.Tensor,
    ) -> np.ndarray:
        """Compute combined ensemble anomaly score."""
        scores = self.compute_scores(x_point, x_seq)

        # Normalize each score to [0, 1] using min-max
        normalized = {}
        for name, s in scores.items():
            s_min, s_max = s.min(), s.max()
            if s_max - s_min > 1e-8:
                normalized[name] = (s - s_min) / (s_max - s_min)
            else:
                normalized[name] = np.zeros_like(s)

        if self.strategy == "weighted_average":
            combined = sum(
                self.weights[name] * normalized[name] for name in normalized
            )
        elif self.strategy == "max":
            combined = np.max(
                np.stack(list(normalized.values()), axis=0), axis=0
            )
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        return combined

    def predict(
        self,
        x_point: torch.Tensor,
        x_seq: torch.Tensor,
    ) -> np.ndarray:
        """Predict anomaly labels (0 = normal, 1 = anomaly)."""
        scores = self.ensemble_score(x_point, x_seq)
        return (scores >= self.ensemble_threshold).astype(int)

    def calibrate_threshold(
        self,
        x_point: torch.Tensor,
        x_seq: torch.Tensor,
        percentile: float = 95.0,
    ) -> float:
        """Set the ensemble threshold based on score distribution."""
        scores = self.ensemble_score(x_point, x_seq)
        self.ensemble_threshold = float(np.percentile(scores, percentile))
        return self.ensemble_threshold

    def state_dict(self) -> dict:
        """Get full ensemble state for serialization."""
        return {
            "autoencoder": self.autoencoder.state_dict(),
            "lstm": self.lstm.state_dict(),
            "transformer": self.transformer.state_dict(),
            "weights": self.weights,
            "strategy": self.strategy,
            "ensemble_threshold": self.ensemble_threshold,
            "thresholds": self.thresholds,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore ensemble state from serialized form."""
        self.autoencoder.load_state_dict(state["autoencoder"])
        self.lstm.load_state_dict(state["lstm"])
        self.transformer.load_state_dict(state["transformer"])
        self.weights = state["weights"]
        self.strategy = state["strategy"]
        self.ensemble_threshold = state["ensemble_threshold"]
        self.thresholds = state.get("thresholds", {})
