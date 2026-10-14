"""AASIST model implementation for anti-spoofing."""

from __future__ import annotations

import logging
import os

import torch
import torch.nn as nn

from app.audio.AASIST import Model as AASISTModel
from app.core.config import get_settings

logger = logging.getLogger(__name__)

_settings = get_settings()

_model_instance: nn.Module | None = None

AASIST_CONFIG = {
    "nb_samp": 64600,
    "first_conv": 128,
    "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
    "gat_dims": [64, 32],
    "pool_ratios": [0.5, 0.7, 0.5, 0.5],
    "temperatures": [2.0, 2.0, 100.0, 100.0],
}


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_aasist_model(model_name: str = None) -> nn.Module:
    global _model_instance
    if _model_instance is not None:
        return _model_instance

    device = get_device()
    logger.info("Loading AASIST model on %s", device)

    model = AASISTModel(AASIST_CONFIG)

    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        weights_path = os.path.join(current_dir, "..", "..", "models", "AASIST.pth")
        weights_path = os.path.normpath(weights_path)

        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict, strict=True)
        
        param_count = sum(p.numel() for p in model.parameters())
        logger.info("Loaded AASIST weights from %s (strict=True)", weights_path)
        logger.info("Model parameter count: %d (expected ~299568)", param_count)
    except Exception as e:
        logger.error("Failed to load AASIST weights: %s", e)
        raise

    model.to(device)
    model.eval()
    _model_instance = model
    return model


def get_aasist_model() -> nn.Module:
    global _model_instance
    if _model_instance is None:
        return load_aasist_model()
    return _model_instance


def unload_model() -> None:
    global _model_instance
    if _model_instance is not None:
        del _model_instance
        _model_instance = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()