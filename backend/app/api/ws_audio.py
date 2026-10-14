"""WebSocket endpoint for real-time audio streaming and deepfake detection."""

from __future__ import annotations

import json
import logging
import time

import jwt
import numpy as np
from fastapi import (
    APIRouter,
    Depends,
    Query,
    WebSocket,
    WebSocketDisconnect,
    WebSocketException,
    status,
)
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.audio.inference import get_inference_service, AASIST_MAX_LEN, AASIST_SAMPLE_RATE
from app.core.config import get_settings
from app.core.security import decode_token
from app.db.models import AudioAnomalyEvent, User
from app.db.session import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ws", tags=["audio"])

_settings = get_settings()


async def authenticate_websocket(
    websocket: WebSocket,
    token: str,
    db: AsyncSession,
) -> User:
    """Authenticate WebSocket connection using JWT token."""
    try:
        payload = decode_token(token)
    except jwt.exceptions.ExpiredSignatureError:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Token expired")
        raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION, reason="Token expired")
    except jwt.exceptions.InvalidTokenError:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Invalid token")
        raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION, reason="Invalid token")

    if payload.get("type") != "access":
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Not an access token")
        raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION, reason="Not an access token")

    user_id = int(payload["sub"])
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user or not user.is_active:
        await websocket.close(
            code=status.WS_1008_POLICY_VIOLATION, reason="User not found or inactive"
        )
        raise WebSocketException(
            code=status.WS_1008_POLICY_VIOLATION, reason="User not found or inactive"
        )

    return user


@router.websocket("/audio")
async def websocket_audio(
    websocket: WebSocket,
    token: str = Query(..., description="JWT access token"),
    session_id: str = Query(..., description="Unique session identifier"),
    sample_rate: int = Query(16000, ge=8000, le=48000, description="Audio sample rate"),
    db: AsyncSession = Depends(get_db),
) -> None:
    """WebSocket endpoint for real-time audio streaming and deepfake detection.

    Client sends raw PCM audio chunks (binary), server returns anomaly scores
    when a fixed 64600-sample (~4.04s at 16kHz) window is accumulated.

    Query Parameters:
        token: JWT access token for authentication
        session_id: Unique session identifier for this call/stream
        sample_rate: Audio sample rate (8000-48000)

    Message Format (Server -> Client):
        {
            "type": "score",
            "chunk_index": 0,
            "anomaly_score": 0.123,
            "is_high_risk": false,
            "threshold": 0.75,
            "processing_time_ms": 12.5,
            "timestamp": "2025-01-15T10:30:00Z"
        }
    """
    user = await authenticate_websocket(websocket, token, db)
    await websocket.accept()

    inference_service = get_inference_service()
    chunk_index = 0
    audio_buffer = np.array([], dtype=np.float32)

    logger.info(
        "Audio WebSocket connected: user=%s, session=%s, sr=%d, window_samples=%d",
        user.id,
        session_id,
        sample_rate,
        AASIST_MAX_LEN,
    )

    try:
        while True:
            # Receive binary audio chunk (int16 PCM)
            data = await websocket.receive_bytes()

            # Convert to float32 [-1, 1]
            chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

            # Resample if needed
            if sample_rate != AASIST_SAMPLE_RATE:
                import librosa
                chunk = librosa.resample(chunk, orig_sr=sample_rate, target_sr=AASIST_SAMPLE_RATE)

            # Accumulate audio data
            audio_buffer = np.concatenate([audio_buffer, chunk])

            # Process when we have enough samples for a full window
            while len(audio_buffer) >= AASIST_MAX_LEN:
                # Take exactly one window
                window = audio_buffer[:AASIST_MAX_LEN]
                audio_buffer = audio_buffer[AASIST_MAX_LEN:]

                # Run inference
                result = inference_service.infer(window, sr=AASIST_SAMPLE_RATE)

                # Persist to database
                event = AudioAnomalyEvent(
                    user_id=user.id,
                    session_id=session_id,
                    chunk_index=chunk_index,
                    anomaly_score=result.anomaly_score,
                    threshold=result.threshold,
                    is_high_risk=result.is_high_risk,
                )
                db.add(event)
                await db.commit()

                # Send score back to client
                response = {
                    "type": "score",
                    "chunk_index": chunk_index,
                    "anomaly_score": result.anomaly_score,
                    "is_high_risk": result.is_high_risk,
                    "threshold": result.threshold,
                    "processing_time_ms": result.processing_time_ms,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                await websocket.send_text(json.dumps(response))

                chunk_index += 1

    except WebSocketDisconnect:
        logger.info("Audio WebSocket disconnected: user=%s, session=%s", user.id, session_id)
    except Exception as e:
        logger.exception("Audio WebSocket error: %s", e)
        try:
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR, reason=str(e))
        except Exception:
            pass
    finally:
        # Process any remaining audio in buffer if it's a full window
        if len(audio_buffer) >= AASIST_MAX_LEN:
            window = audio_buffer[:AASIST_MAX_LEN]
            result = inference_service.infer(window, sr=AASIST_SAMPLE_RATE)
            event = AudioAnomalyEvent(
                user_id=user.id,
                session_id=session_id,
                chunk_index=chunk_index,
                anomaly_score=result.anomaly_score,
                threshold=result.threshold,
                is_high_risk=result.is_high_risk,
            )
            db.add(event)
            await db.commit()