"""PyTorch dataset and data loaders for network telemetry."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset


class NetworkTelemetryDataset(Dataset):
    """Dataset for network telemetry data.

    Supports both point-wise and sequence-based (sliding window) access.
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        sequence_length: int = 1,
    ):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.sequence_length = sequence_length
        self.num_features = features.shape[1]

        if sequence_length > 1:
            # Compute valid sequence start indices
            self._length = max(0, len(features) - sequence_length + 1)
        else:
            self._length = len(features)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.sequence_length > 1:
            seq = self.features[idx : idx + self.sequence_length]
            # Label is the anomaly status of the last element in the window
            label = self.labels[idx + self.sequence_length - 1]
            return seq, label
        return self.features[idx], self.labels[idx]


def create_data_splits(
    dataset: NetworkTelemetryDataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[Subset, Subset, Subset]:
    """Split dataset into train/val/test sets (time-aware, no shuffle)."""
    n = len(dataset)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_indices = list(range(0, train_end))
    val_indices = list(range(train_end, val_end))
    test_indices = list(range(val_end, n))

    return (
        Subset(dataset, train_indices),
        Subset(dataset, val_indices),
        Subset(dataset, test_indices),
    )


def create_dataloaders(
    dataset: NetworkTelemetryDataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    batch_size: int = 64,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test DataLoaders."""
    train_set, val_set, test_set = create_data_splits(
        dataset, train_ratio, val_ratio, seed
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
