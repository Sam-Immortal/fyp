"""Taskiq tasks for NLP processing."""

from __future__ import annotations

from app.nlp.worker import run_nlp_analysis
from app.workers.broker import broker

# Import to register the periodic task
from app.nlp import pending_timeout  # noqa: F401


@broker.task(task_name="nlp.analysis")
async def nlp_analysis_task(event_id: int, text: str, sender: str | None) -> None:
    """Background task for NLP scam classification."""
    await run_nlp_analysis(event_id, text, sender)