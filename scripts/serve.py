#!/usr/bin/env python3
"""Launch the anomaly detection inference API server."""

import argparse
import logging
import sys
from pathlib import Path

import torch
import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from netai_anomaly.inference.service import app, load_model
from netai_anomaly.training.trainer import get_device
from netai_anomaly.utils.config import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Start the anomaly detection API server")
    parser.add_argument("--config", default="configs/default.yaml", help="Config file path")
    parser.add_argument("--host", default=None, help="Override host")
    parser.add_argument("--port", type=int, default=None, help="Override port")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg.training.device)

    # Load model metadata to get input_dim
    metadata_path = Path(cfg.training.checkpoint_dir) / "metadata.pt"
    input_dim = 5
    if metadata_path.exists():
        metadata = torch.load(metadata_path, weights_only=False)
        input_dim = metadata["input_dim"]

    load_model(
        checkpoint_path=cfg.inference.model_path,
        input_dim=input_dim,
        device=str(device),
    )

    host = args.host or cfg.inference.host
    port = args.port or cfg.inference.port
    logger.info(f"Starting inference server on {host}:{port}")

    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
