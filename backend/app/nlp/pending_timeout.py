"""Periodic task to handle stale pending NLP events."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

from app.core.config import get_settings
from app.db.models import NlpScoreEvent
from app.db.session import async_session_factory
from app.workers.broker import broker

logger = logging.getLogger(__name__)

_settings = get_settings()


async def mark_stale_pending_failed(session_factory=None) -> int:
    """Mark NLP events stuck in pending status as failed.

    Args:
        session_factory: Optional session factory for testing. Uses default if not provided.

    Returns:
        Number of events marked as failed.
    """
    if session_factory is None:
        session_factory = async_session_factory

    timeout_minutes = _settings.nlp_pending_timeout_minutes
    cutoff = datetime.now(UTC) - timedelta(minutes=timeout_minutes)

    async with session_factory() as session:
        from sqlalchemy import select

        # Find events that have been pending for too long
        stmt = select(NlpScoreEvent).where(
            NlpScoreEvent.status == "pending",
            NlpScoreEvent.created_at < cutoff,
        )
        result = await session.execute(stmt)
        stale_events = result.scalars().all()

        for event in stale_events:
            logger.warning(
                "NLP event %d has been pending for >%d minutes, marking as failed",
                event.id,
                timeout_minutes,
            )
            event.status = "failed"
            event.risk_level = None
            event.patterns = None
            event.error = f"Analysis timed out after {timeout_minutes} minutes"
            event.explanation = "Analysis timed out - no response from AI service"
            event.is_high_risk = False

        if stale_events:
            await session.commit()
            logger.info("Marked %d stale NLP events as failed", len(stale_events))

        return len(stale_events)


@broker.task(task_name="nlp.pending_timeout", schedule=[{"cron": "* * * * *"}])
async def check_pending_timeout() -> None:
    """Check for NLP events stuck in pending status and mark them as failed."""
    logger.debug("Checking for stale pending NLP events")
    await mark_stale_pending_failed()