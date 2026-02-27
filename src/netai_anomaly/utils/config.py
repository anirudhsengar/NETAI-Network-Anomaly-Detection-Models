"""Configuration management for NETAI anomaly detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    database_path: str = "data/network_telemetry.db"
    sequence_length: int = 32
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    batch_size: int = 64


@dataclass
class FeatureConfig:
    rolling_windows: list[int] = field(default_factory=lambda: [5, 15, 30])
    numeric_features: list[str] = field(
        default_factory=lambda: [
            "throughput_mbps",
            "latency_ms",
            "packet_loss_pct",
            "retransmits",
            "jitter_ms",
        ]
    )
    derived_features: bool = True
    normalize: bool = True


@dataclass
class AutoencoderConfig:
    hidden_dims: list[int] = field(default_factory=lambda: [64, 32, 16])
    latent_dim: int = 8
    dropout: float = 0.1
    learning_rate: float = 0.001
    epochs: int = 50
    threshold_percentile: float = 95.0


@dataclass
class LSTMConfig:
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    bidirectional: bool = True
    learning_rate: float = 0.001
    epochs: int = 50
    threshold_percentile: float = 95.0


@dataclass
class TransformerConfig:
    d_model: int = 64
    nhead: int = 4
    num_encoder_layers: int = 3
    dim_feedforward: int = 128
    dropout: float = 0.1
    learning_rate: float = 0.0005
    epochs: int = 50
    threshold_percentile: float = 95.0


@dataclass
class EnsembleConfig:
    weights: dict[str, float] = field(
        default_factory=lambda: {"autoencoder": 0.33, "lstm": 0.34, "transformer": 0.33}
    )
    strategy: str = "weighted_average"


@dataclass
class ModelsConfig:
    autoencoder: AutoencoderConfig = field(default_factory=AutoencoderConfig)
    lstm: LSTMConfig = field(default_factory=LSTMConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)


@dataclass
class TrainingConfig:
    device: str = "auto"
    early_stopping_patience: int = 10
    checkpoint_dir: str = "checkpoints"
    log_interval: int = 5
    seed: int = 42


@dataclass
class InferenceConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    model_path: str = "checkpoints/best_ensemble.pt"
    batch_inference: bool = True


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)


def _build_dataclass(cls: type, data: dict[str, Any]) -> Any:
    """Recursively build a dataclass from a dictionary."""
    if data is None:
        return cls()
    field_types = {f.name: f.type for f in cls.__dataclass_fields__.values()}
    kwargs = {}
    for key, value in data.items():
        if key in field_types and isinstance(value, dict):
            # Check if the field type is itself a dataclass
            ft = field_types[key]
            if isinstance(ft, str):
                ft = globals().get(ft) or locals().get(ft)
            if ft and hasattr(ft, "__dataclass_fields__"):
                value = _build_dataclass(ft, value)
        kwargs[key] = value
    return cls(**kwargs)


def load_config(path: str | Path) -> Config:
    """Load configuration from a YAML file."""
    path = Path(path)
    if not path.exists():
        return Config()

    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    data_cfg = DataConfig(**raw.get("data", {}))
    features_cfg = FeatureConfig(**raw.get("features", {}))

    models_raw = raw.get("models", {})
    models_cfg = ModelsConfig(
        autoencoder=AutoencoderConfig(**models_raw.get("autoencoder", {})),
        lstm=LSTMConfig(**models_raw.get("lstm", {})),
        transformer=TransformerConfig(**models_raw.get("transformer", {})),
        ensemble=EnsembleConfig(**models_raw.get("ensemble", {})),
    )

    training_cfg = TrainingConfig(**raw.get("training", {}))
    inference_cfg = InferenceConfig(**raw.get("inference", {}))

    return Config(
        data=data_cfg,
        features=features_cfg,
        models=models_cfg,
        training=training_cfg,
        inference=inference_cfg,
    )
