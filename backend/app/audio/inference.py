"""Audio inference service for real-time deepfake detection."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
import torch

from app.audio.model import get_aasist_model, get_device
from app.core.config import get_settings

logger = logging.getLogger(__name__)

_settings = get_settings()

AASIST_SAMPLE_RATE = 16000
AASIST_MAX_LEN = 64600


def pad_audio(x: np.ndarray, max_len: int = AASIST_MAX_LEN) -> np.ndarray:
    """Pad or tile audio to fixed length (official AASIST behavior).
    
    Does NOT zero-pad. Instead repeats/tiles the audio to reach max_len.
    """
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x_len) + 1
    return np.tile(x, num_repeats)[:max_len]


@dataclass
class InferenceResult:
    """Result of audio anomaly inference."""

    anomaly_score: float
    is_high_risk: bool
    processing_time_ms: float
    threshold: float


class AudioInferenceService:
    """Real-time audio anomaly detection service."""

    def __init__(self, threshold: float | None = None) -> None:
        self.threshold = threshold or _settings.safety_threshold
        self.model = None
        self._initialized = False

    def initialize(self) -> None:
        """Initialize model."""
        if self._initialized:
            return

        logger.info("Initializing audio inference service...")
        self.model = get_aasist_model()
        self._initialized = True
        logger.info("Audio inference service initialized")

    def infer(self, audio: np.ndarray, sr: int = AASIST_SAMPLE_RATE) -> InferenceResult:
        """Run inference on audio chunk.

        Args:
            audio: Raw audio waveform (float32, mono)
            sr: Sample rate (must be 16000)

        Returns:
            InferenceResult with anomaly score and metadata
        """
        if not self._initialized:
            self.initialize()

        if sr != AASIST_SAMPLE_RATE:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=AASIST_SAMPLE_RATE)

        # Ensure mono
        if audio.ndim > 1:
            audio = np.mean(audio, axis=0)

        # Pad/tile to fixed length (64600 samples = ~4.04s at 16kHz)
        audio = pad_audio(audio, AASIST_MAX_LEN)

        start_time = time.perf_counter()

        # Convert to tensor: (batch=1, time=64600) - model expects (batch, time) and adds channel dim internally
        device = get_device()
        x = torch.from_numpy(audio).float().unsqueeze(0).to(device)

        # Run inference
        with torch.no_grad():
            _, output = self.model(x)
            # output shape: (1, 2) - index 0 = spoof, index 1 = bonafide
            # (confirmed against the official repo's own label convention
            # and evaluation scoring code)
            probs = torch.softmax(output, dim=1)
            anomaly_score = float(probs[0, 0].item())  # spoof class probability

        processing_time_ms = (time.perf_counter() - start_time) * 1000

        is_high_risk = anomaly_score >= self.threshold

        return InferenceResult(
            anomaly_score=anomaly_score,
            is_high_risk=is_high_risk,
            processing_time_ms=processing_time_ms,
            threshold=self.threshold,
        )

    def infer_batch(self, audio_batch: list[np.ndarray], sr: int = AASIST_SAMPLE_RATE) -> list[InferenceResult]:
        """Run inference on batch of audio chunks.

        Args:
            audio_batch: List of raw audio waveforms
            sr: Sample rate (must be 16000)

        Returns:
            List of InferenceResult
        """
        if not self._initialized:
            self.initialize()

        if sr != AASIST_SAMPLE_RATE:
            import librosa
            audio_batch = [librosa.resample(a, orig_sr=sr, target_sr=AASIST_SAMPLE_RATE) for a in audio_batch]

        # Ensure mono and pad each
        processed_batch = []
        for audio in audio_batch:
            if audio.ndim > 1:
                audio = np.mean(audio, axis=0)
            audio = pad_audio(audio, AASIST_MAX_LEN)
            processed_batch.append(audio)

        start_time = time.perf_counter()

        # Stack into batch tensor: (batch, 64600) - model expects (batch, time) and adds channel dim internally
        device = get_device()
        x = torch.from_numpy(np.stack(processed_batch)).float().to(device)

        # Run inference
        with torch.no_grad():
            _, outputs = self.model(x)
            probs = torch.softmax(outputs, dim=1)
            anomaly_scores = probs[:, 0].cpu().numpy()

        processing_time_ms = (time.perf_counter() - start_time) * 1000

        return [
            InferenceResult(
                anomaly_score=float(score),
                is_high_risk=score >= self.threshold,
                processing_time_ms=processing_time_ms / len(audio_batch),
                threshold=self.threshold,
            )
            for score in anomaly_scores
        ]


# Global service instance
_inference_service: AudioInferenceService | None = None


def get_inference_service(threshold: float | None = None) -> AudioInferenceService:
    """Get singleton inference service."""
    global _inference_service
    if _inference_service is None:
        _inference_service = AudioInferenceService(threshold=threshold)
    elif threshold is not None:
        _inference_service.threshold = threshold
    return _inference_service