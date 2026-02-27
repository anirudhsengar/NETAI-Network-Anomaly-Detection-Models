"""LSTM-based anomaly detection model.

Uses a sequence-to-one architecture that learns temporal patterns in
network telemetry. Anomalies are detected as sequences whose prediction
error deviates significantly from the learned normal behaviour.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LSTMDetector(nn.Module):
    """Bidirectional LSTM for sequence-based network anomaly detection.

    Architecture:
        LSTM encoder → FC layers → reconstruction of last time step
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)

        self.fc = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim).

        Returns:
            Reconstruction of the last time step (batch, input_dim).
        """
        lstm_out, _ = self.lstm(x)
        # Use the output at the last time step
        last_output = lstm_out[:, -1, :]
        return self.fc(last_output)

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Compute anomaly score as MSE between prediction and actual last step."""
        self.eval()
        with torch.no_grad():
            predicted = self.forward(x)
            actual = x[:, -1, :]
            scores = torch.mean((actual - predicted) ** 2, dim=-1)
        return scores
