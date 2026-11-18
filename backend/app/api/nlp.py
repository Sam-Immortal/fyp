"""NLP scam classification endpoint."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth import get_current_user
from app.core.config import get_settings
from app.db.models import NlpScoreEvent
from app.db.session import get_db
from app.nlp.tasks import nlp_analysis_task
from app.schemas.nlp import (
    NlpScoreEventResponse,
    NlpScoreRequest,
    NlpScoreResponse,
    ScamPattern,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/nlp", tags=["nlp"])

_settings = get_settings()


@router.post("/score", response_model=NlpScoreResponse, status_code=status.HTTP_202_ACCEPTED)
async def score_sms(
    payload: NlpScoreRequest,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> NlpScoreResponse:
    """
    Submit SMS text for asynchronous scam classification.

    Acknowledged immediately; verdict available shortly after via GET /api/v1/nlp/events.
    """
    # Create pending event
    event = NlpScoreEvent(
        user_id=current_user.id,
        sender=payload.sender or "unknown",
        message_preview=payload.text[:500],
        score=0.0,
        threshold=_settings.safety_threshold,
        is_high_risk=False,
        status="pending",
        risk_level=None,
        patterns=None,
        explanation=None,
        error=None,
    )
    db.add(event)
    await db.commit()
    await db.refresh(event)

    # Queue background analysis
    await nlp_analysis_task.kiq(event.id, payload.text, payload.sender)

    return NlpScoreResponse(
        event_id=event.id,
        status="pending",
        scam_probability=0.0,
        risk_level="pending",
        patterns=ScamPattern(),
        explanation="Analysis acknowledged; verdict available shortly",
        model=_settings.hf_llm_model,
    )


@router.get("/events", response_model=list[NlpScoreEventResponse])
async def list_nlp_events(
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    limit: int = 50,
    offset: int = 0,
    since_id: int | None = None,
) -> list[NlpScoreEventResponse]:
    """List recent NLP score events for the current user."""
    query = select(NlpScoreEvent).where(NlpScoreEvent.user_id == current_user.id)
    if since_id:
        query = query.where(NlpScoreEvent.id > since_id)
    query = query.order_by(NlpScoreEvent.created_at.desc()).limit(limit).offset(offset)

    result = await db.execute(query)
    events = result.scalars().all()
    return [NlpScoreEventResponse.model_validate(e) for e in events]


@router.get("/events/{event_id}", response_model=NlpScoreEventResponse)
async def get_nlp_event(
    event_id: int,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> NlpScoreEventResponse:
    """Get a specific NLP score event by ID (owner only)."""
    event = await db.get(NlpScoreEvent, event_id)
    if not event or event.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Event not found")
    return NlpScoreEventResponse.model_validate(event)