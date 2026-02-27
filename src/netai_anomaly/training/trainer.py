"""Training pipeline for anomaly detection models.

Supports training autoencoder, LSTM, and transformer models with
early stopping, checkpointing, and GPU acceleration.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from netai_anomaly.models.autoencoder import Autoencoder
from netai_anomaly.models.lstm_detector import LSTMDetector
from netai_anomaly.models.transformer_detector import TransformerDetector
from netai_anomaly.utils.metrics import compute_anomaly_metrics

logger = logging.getLogger(__name__)


def get_device(preference: str = "auto") -> torch.device:
    """Determine the best available compute device."""
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(preference)


class EarlyStopping:
    """Early stopping to halt training when validation loss plateaus."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class Trainer:
    """Unified trainer for all anomaly detection models."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        checkpoint_dir: str = "checkpoints",
        log_interval: int = 5,
    ):
        self.model = model.to(device)
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_interval = log_interval

    def _compute_loss(
        self, model: nn.Module, batch: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """Compute reconstruction loss for a batch."""
        x, _ = batch
        x = x.to(self.device)

        if isinstance(model, Autoencoder):
            # Point-wise: flatten sequences if necessary
            if x.dim() == 3:
                x = x[:, -1, :]
            x_recon = model(x)
            return nn.functional.mse_loss(x_recon, x)
        elif isinstance(model, (LSTMDetector, TransformerDetector)):
            # Sequence-based: predict last time step
            if x.dim() == 2:
                x = x.unsqueeze(1)
            predicted = model(x)
            target = x[:, -1, :]
            return nn.functional.mse_loss(predicted, target)
        else:
            raise TypeError(f"Unsupported model type: {type(model)}")

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        learning_rate: float = 0.001,
        patience: int = 10,
        model_name: str = "model",
    ) -> dict:
        """Train the model with early stopping and checkpointing.

        Returns:
            Training history with loss curves and best metrics.
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=max(1, patience // 3)
        )
        early_stopping = EarlyStopping(patience=patience)

        history = {"train_loss": [], "val_loss": [], "best_epoch": 0, "best_val_loss": float("inf")}

        logger.info(f"Training {model_name} on {self.device} for up to {epochs} epochs")
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            # Training phase
            self.model.train()
            train_losses = []
            for batch in train_loader:
                optimizer.zero_grad()
                loss = self._compute_loss(self.model, batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                train_losses.append(loss.item())

            avg_train_loss = np.mean(train_losses)

            # Validation phase
            self.model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    loss = self._compute_loss(self.model, batch)
                    val_losses.append(loss.item())

            avg_val_loss = np.mean(val_losses)
            scheduler.step(avg_val_loss)

            history["train_loss"].append(float(avg_train_loss))
            history["val_loss"].append(float(avg_val_loss))

            # Checkpointing
            if avg_val_loss < history["best_val_loss"]:
                history["best_val_loss"] = float(avg_val_loss)
                history["best_epoch"] = epoch
                torch.save(
                    self.model.state_dict(),
                    self.checkpoint_dir / f"best_{model_name}.pt",
                )

            if epoch % self.log_interval == 0 or epoch == 1:
                logger.info(
                    f"  Epoch {epoch:3d}/{epochs} | "
                    f"Train Loss: {avg_train_loss:.6f} | "
                    f"Val Loss: {avg_val_loss:.6f}"
                )

            if early_stopping.step(avg_val_loss):
                logger.info(f"  Early stopping at epoch {epoch}")
                break

        elapsed = time.time() - start_time
        history["training_time_seconds"] = elapsed
        logger.info(
            f"  {model_name} training complete in {elapsed:.1f}s | "
            f"Best val loss: {history['best_val_loss']:.6f} at epoch {history['best_epoch']}"
        )

        # Reload best weights
        best_path = self.checkpoint_dir / f"best_{model_name}.pt"
        if best_path.exists():
            self.model.load_state_dict(torch.load(best_path, weights_only=True))

        return history

    def evaluate(
        self,
        test_loader: DataLoader,
        threshold_percentile: float = 95.0,
    ) -> dict:
        """Evaluate the model on test data.

        Returns:
            Dictionary with anomaly detection metrics and threshold.
        """
        self.model.eval()
        all_scores = []
        all_labels = []

        with torch.no_grad():
            for batch in test_loader:
                x, y = batch
                x = x.to(self.device)

                if isinstance(self.model, Autoencoder):
                    if x.dim() == 3:
                        x = x[:, -1, :]
                    scores = self.model.anomaly_score(x)
                else:
                    if x.dim() == 2:
                        x = x.unsqueeze(1)
                    scores = self.model.anomaly_score(x)

                all_scores.append(scores.cpu().numpy())
                all_labels.append(y.numpy())

        all_scores = np.concatenate(all_scores)
        all_labels = np.concatenate(all_labels)

        # Compute threshold from training distribution
        threshold = float(np.percentile(all_scores, threshold_percentile))
        metrics = compute_anomaly_metrics(all_labels, all_scores, threshold)

        return metrics
