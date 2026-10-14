"""Mel spectrogram extraction for audio anti-spoofing."""

from __future__ import annotations

import logging

import librosa
import numpy as np
import torch

from app.core.config import get_settings

logger = logging.getLogger(__name__)

_settings = get_settings()


class MelSpectrogramExtractor:
    """Extract log-Mel spectrograms from raw audio."""

    def __init__(
        self,
        sample_rate: int | None = None,
        n_mels: int | None = None,
        n_fft: int | None = None,
        hop_length: int | None = None,
        win_length: int | None = None,
        window: str | None = None,
        power: float | None = None,
        fmin: float | None = None,
        fmax: float | None = None,
        norm: str | None = None,
        mel_scale: str | None = None,
    ) -> None:
        self.sample_rate = sample_rate or _settings.audio_sample_rate
        self.n_mels = n_mels or _settings.audio_n_mels
        self.n_fft = n_fft or _settings.audio_n_fft
        self.hop_length = hop_length or _settings.audio_hop_length
        self.win_length = win_length or _settings.audio_win_length
        self.window = window or _settings.audio_window
        self.power = power or _settings.audio_power
        self.fmin = fmin or _settings.audio_fmin
        self.fmax = fmax or _settings.audio_fmax
        self.norm = norm or _settings.audio_norm
        self.mel_scale = mel_scale or _settings.audio_mel_scale

    def extract(self, audio: np.ndarray, sr: int | None = None) -> torch.Tensor:
        """Extract log-Mel spectrogram from audio.

        Args:
            audio: Raw audio waveform as numpy array (float32, mono)
            sr: Sample rate (if None, uses config default)

        Returns:
            Log-Mel spectrogram as torch.Tensor of shape (1, n_mels, time)
        """
        if sr is None:
            sr = self.sample_rate
        elif sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)

        # Ensure mono
        if audio.ndim > 1:
            audio = librosa.to_mono(audio)

        # Extract Mel spectrogram (mel_scale param added in librosa 0.10.0)
        melspec_kwargs = dict(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            n_mels=self.n_mels,
            power=self.power,
            fmin=self.fmin,
            fmax=self.fmax,
            norm=self.norm,
        )
        # mel_scale parameter added in librosa 0.10.0
        import librosa

        if hasattr(librosa.feature.melspectrogram, "__call__"):
            try:
                import inspect

                sig = inspect.signature(librosa.feature.melspectrogram)
                if "mel_scale" in sig.parameters:
                    melspec_kwargs["mel_scale"] = self.mel_scale
            except Exception:
                pass

        mel_spec = librosa.feature.melspectrogram(**melspec_kwargs)

        # Convert to log scale
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)

        # Normalize to [0, 1] range for model input
        log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-8)

        # Convert to tensor: (1, n_mels, time)
        tensor = torch.from_numpy(log_mel).float().unsqueeze(0)

        return tensor

    def extract_batch(self, audio_batch: list[np.ndarray]) -> torch.Tensor:
        """Extract spectrograms for a batch of audio.

        Args:
            audio_batch: List of raw audio waveforms

        Returns:
            Batch tensor of shape (batch, 1, n_mels, time)
        """
        specs = [self.extract(audio) for audio in audio_batch]
        # Pad to same time dimension
        max_time = max(s.shape[-1] for s in specs)
        padded = []
        for spec in specs:
            if spec.shape[-1] < max_time:
                pad = max_time - spec.shape[-1]
                spec = torch.nn.functional.pad(spec, (0, pad))
            padded.append(spec)
        return torch.stack(padded)


# Singleton extractor instance
_extractor_instance: MelSpectrogramExtractor | None = None


def get_spectrogram_extractor() -> MelSpectrogramExtractor:
    """Get singleton spectrogram extractor."""
    global _extractor_instance
    if _extractor_instance is None:
        _extractor_instance = MelSpectrogramExtractor()
    return _extractor_instance
