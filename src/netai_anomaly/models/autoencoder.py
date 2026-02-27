"""Autoencoder-based anomaly detection model.

Uses reconstruction error as the anomaly score: samples that are poorly
reconstructed are likely anomalous since the model is trained on
predominantly normal network traffic.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    """Deep autoencoder for reconstruction-based anomaly detection.

    Architecture:
        Encoder: input_dim → hidden_dims → latent_dim
        Decoder: latent_dim → hidden_dims(reversed) → input_dim
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | None = None,
        latent_dim: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32, 16]

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Build encoder
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Build decoder (mirror of encoder)
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-sample reconstruction error (MSE)."""
        self.eval()
        with torch.no_grad():
            x_recon = self.forward(x)
            scores = torch.mean((x - x_recon) ** 2, dim=-1)
        return scores
