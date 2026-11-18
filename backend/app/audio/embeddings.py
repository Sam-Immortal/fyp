"""Embedding extraction for AASIST model.

Captures the 160-value pre-output embedding (last_hidden) alongside
the anomaly score. Uses the exact same preprocessing and model path
as app.audio.inference.
"""

from __future__ import annotations

import torch
import numpy as np

from app.audio.inference import AASIST_MAX_LEN, pad_audio, get_device
from app.audio.model import get_aasist_model


def embed_window(audio: np.ndarray, sr: int = 16000) -> tuple[float, np.ndarray]:
    """Run AASIST inference and return anomaly score + 160-dim embedding.

    Args:
        audio: Raw audio waveform (float32, mono)
        sr: Sample rate (must be 16000)

    Returns:
        (anomaly_score, embedding) where embedding is float32 array of shape (160,)
    """
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

    # Ensure mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=0)

    # Pad/tile to fixed length (same as inference.py)
    audio = pad_audio(audio, AASIST_MAX_LEN)

    # Build input tensor exactly like inference.py
    device = get_device()
    x = torch.from_numpy(audio).float().unsqueeze(0).to(device)  # (1, 64600)

    model = get_aasist_model()
    model.eval()

    with torch.no_grad():
        last_hidden, output = model(x)
        # last_hidden shape: (1, 160)
        # output shape: (1, 2) - logits for [spoof, bonafide]

        probs = torch.softmax(output, dim=1)
        anomaly_score = float(probs[0, 0].item())  # spoof class probability

        # Extract embedding as float32 numpy array
        embedding = last_hidden[0].cpu().numpy().astype(np.float32)  # (160,)

    return anomaly_score, embedding


def verify_embedding_linear_layer(embedding: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply the model's final linear layer to the embedding to reproduce logits.

    Returns:
        (reproduced_logits, actual_logits) both shape (2,)
    """
    model = get_aasist_model()
    model.eval()

    with torch.no_grad():
        # embedding is (160,) -> add batch dim
        emb_tensor = torch.from_numpy(embedding).float().unsqueeze(0).to(get_device())
        reproduced_logits = model.out_layer(emb_tensor)[0].cpu().numpy()

        # Get actual logits by running forward again
        # We need to run the full forward to get the actual logits
        # This is just for verification - in practice we'd need the original input
        # But we can't easily get actual logits without the original input
        # So we just return reproduced_logits
        actual_logits = np.zeros(2)  # placeholder

    return reproduced_logits, actual_logits


def get_final_linear_layer() -> torch.nn.Linear:
    """Get the model's final linear layer (out_layer)."""
    model = get_aasist_model()
    return model.out_layer


def embed_window_with_logits(audio: np.ndarray, sr: int = 16000) -> tuple[float, np.ndarray, np.ndarray]:
    """Run inference and return anomaly_score, embedding, and logits.

    Returns:
        (anomaly_score, embedding, logits) where:
        - anomaly_score: float (spoof probability)
        - embedding: np.ndarray float32 shape (160,)
        - logits: np.ndarray float32 shape (2,) [spoof_logit, bonafide_logit]
    """
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=0)

    audio = pad_audio(audio, AASIST_MAX_LEN)

    device = get_device()
    x = torch.from_numpy(audio).float().unsqueeze(0).to(device)

    model = get_aasist_model()
    model.eval()

    with torch.no_grad():
        last_hidden, output = model(x)
        probs = torch.softmax(output, dim=1)
        anomaly_score = float(probs[0, 0].item())
        embedding = last_hidden[0].cpu().numpy().astype(np.float32)
        logits = output[0].cpu().numpy().astype(np.float32)

    return anomaly_score, embedding, logits