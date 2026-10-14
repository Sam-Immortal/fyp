"""NLP scam classification endpoint."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.security import decode_token
from app.db.models import NlpScoreEvent, User
from app.db.session import get_db
from app.nlp.worker import queue_nlp_analysis
from app.schemas.nlp import NlpScoreRequest, NlpScoreResponse, ScamPattern

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/nlp", tags=["nlp"])

_settings = get_settings()


async def get_current_user(request: Request, db: AsyncSession = Depends(get_db)) -> User:
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token")

    token = auth_header.split(" ", 1)[1]
    try:
        payload = decode_token(token)
    except Exception:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

    if payload.get("type") != "access":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not an access token")

    user_id = int(payload["sub"])
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user or not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found or inactive")

    return user


@router.post("/score", response_model=NlpScoreResponse, status_code=status.HTTP_202_ACCEPTED)
async def score_sms(
    payload: NlpScoreRequest,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> NlpScoreResponse:
    """
    Submit SMS text for asynchronous scam classification.

    Returns immediately with a placeholder response; actual analysis runs in background worker.
    Client should poll or use WebSocket for result notification.
    """
    # Create placeholder event
    event = NlpScoreEvent(
        user_id=current_user.id,
        sender=payload.sender or "unknown",
        message_preview=payload.text[:500],
        score=0.0,
        threshold=_settings.safety_threshold,
        is_high_risk=False,
        raw_response=None,
    )
    db.add(event)
    await db.commit()
    await db.refresh(event)

    # Queue background analysis
    await queue_nlp_analysis(event.id, payload.text, payload.sender)

    return NlpScoreResponse(
        scam_probability=0.0,
        risk_level="low",
        patterns=ScamPattern(),
        explanation="Analysis queued",
        model=_settings.hf_llm_model,
    )


@router.get("/events", response_model=list[dict])
async def list_nlp_events(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    limit: int = 50,
    offset: int = 0,
) -> list[dict]:
    """List recent NLP score events for the current user."""
    result = await db.execute(
        select(NlpScoreEvent)
        .where(NlpScoreEvent.user_id == current_user.id)
        .order_by(NlpScoreEvent.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    events = result.scalars().all()
    return [
        {
            "id": e.id,
            "sender": e.sender,
            "message_preview": e.message_preview,
            "score": e.score,
            "threshold": e.threshold,
            "is_high_risk": e.is_high_risk,
            "created_at": e.created_at.isoformat(),
        }
        for e in events
    ]