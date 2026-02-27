"""Transformer-based anomaly detection model.

Uses a transformer encoder to learn complex temporal dependencies in
network telemetry sequences. Self-attention allows the model to capture
both local and long-range patterns in network behaviour.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence position information."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerDetector(nn.Module):
    """Transformer encoder for sequence-based network anomaly detection.

    Architecture:
        Linear projection → Positional encoding → Transformer encoder
        → Mean pooling → FC → reconstruction
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 3,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model

        # Project input features to model dimension
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        self.fc = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim).

        Returns:
            Reconstruction of the last time step (batch, input_dim).
        """
        # Project to d_model dimension
        x_proj = self.input_projection(x)
        x_proj = self.pos_encoder(x_proj)

        # Transformer encoding
        encoded = self.transformer_encoder(x_proj)

        # Use the last position's encoding for prediction
        last_encoded = encoded[:, -1, :]
        return self.fc(last_encoded)

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Compute anomaly score as MSE between prediction and actual last step."""
        self.eval()
        with torch.no_grad():
            predicted = self.forward(x)
            actual = x[:, -1, :]
            scores = torch.mean((actual - predicted) ** 2, dim=-1)
        return scores
