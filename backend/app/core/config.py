"""Application configuration loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated, Any, Literal

from pydantic import BeforeValidator, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def parse_optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


OptionalFloat = Annotated[float | None, BeforeValidator(parse_optional_float)]


class Settings(BaseSettings):
    """Runtime configuration for the ScamGuard API.

    All values are populated from environment variables or a `.env` file.
    No static IPs or secrets are baked into the image.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = Field(default="ScamGuard API", description="Service display name")
    environment: Literal["development", "staging", "production"] = "development"
    debug: bool = False

    host: str = Field(default="0.0.0.0", description="Bind host")
    port: int = Field(default=8000, ge=1, le=65535, description="Bind port")

    database_url: str = Field(
        default="postgresql+asyncpg://scamguard:scamguard@postgres:5432/scamguard",
        description="Async SQLAlchemy DSN",
    )
    redis_url: str = Field(
        default="redis://redis:6379/0",
        description="Redis connection URL (also Taskiq broker)",
    )
    minio_endpoint: str = Field(default="minio:9000", description="MinIO endpoint")
    minio_access_key: str = Field(default="scamguard")
    minio_secret_key: str = Field(default="scamguard-secret")
    minio_bucket: str = Field(default="ci-artifacts")

    jwt_secret: str = Field(default="change-me-in-production", min_length=16)
    jwt_algorithm: str = Field(default="HS256")
    jwt_access_expire_minutes: int = Field(default=15, ge=1)
    jwt_refresh_expire_days: int = Field(default=7, ge=1)

    safety_threshold: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Anomaly score above which an alert is raised",
    )

    cors_origins_raw: str = Field(default="*", alias="CORS_ORIGINS")

    @property
    def cors_origins(self) -> list[str]:
        if self.cors_origins_raw == "*":
            return ["*"]
        return [origin.strip() for origin in self.cors_origins_raw.split(",")]

    hf_token: str | None = Field(default=None, description="Hugging Face token for Inference API")
    hf_llm_model: str = Field(default="mistralai/Mistral-7B-Instruct-v0.3")
    hf_inference_base_url: str = Field(
        default="https://router.huggingface.co/v1",
        description="Hugging Face Inference Providers base URL (OpenAI-compatible)",
    )
    audio_model_name: str = Field(default="stapes-ai/aasist")

    # Audio processing settings
    audio_sample_rate: int = Field(
        default=16000, ge=8000, le=48000, description="Audio sample rate in Hz"
    )
    audio_n_mels: int = Field(default=80, ge=40, le=128, description="Number of Mel filterbanks")
    audio_n_fft: int = Field(default=512, ge=256, le=2048, description="FFT window size")
    audio_hop_length: int = Field(default=160, ge=80, le=320, description="Hop length (samples)")
    audio_win_length: int = Field(
        default=400, ge=200, le=800, description="Window length (samples)"
    )
    audio_window: str = Field(default="hann", description="Window function type")
    audio_power: float = Field(default=2.0, description="Power for Mel spectrogram")
    audio_fmin: float = Field(default=0.0, description="Minimum frequency for Mel scale")
    audio_fmax: OptionalFloat = Field(
        default=None, description="Maximum frequency for Mel scale (None = sr/2)"
    )
    audio_norm: str = Field(default="slaney", description="Mel filterbank normalization")
    audio_mel_scale: str = Field(default="htk", description="Mel scale type")

    nlp_pending_timeout_minutes: int = Field(
        default=10,
        ge=1,
        description="Minutes after which a pending NLP event is marked as failed",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings accessor."""
    return Settings()
