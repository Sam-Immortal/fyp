"""Tests for NLP scam classification endpoint."""

from __future__ import annotations

import pytest_asyncio
from datetime import UTC, datetime, timedelta
from httpx import ASGITransport, AsyncClient
from unittest.mock import AsyncMock, patch

from app.core.security import create_access_token, hash_password
from app.db.models import NlpScoreEvent, User
from app.db.session import get_db
from app.main import create_app
from app.nlp.worker import run_nlp_analysis
from app.nlp.pending_timeout import mark_stale_pending_failed
from tests.conftest import test_engine, test_session_factory


@pytest_asyncio.fixture(autouse=True)
async def setup_db():
    from app.db.base import Base

    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture
async def test_user():
    async with test_session_factory() as session:
        user = User(
            email="test@example.com",
            password_hash=hash_password("Str0ngP@ss"),
            full_name="Test User",
        )
        session.add(user)
        await session.commit()
        await session.refresh(user)
        yield user


@pytest_asyncio.fixture
async def client():
    app = create_app()
    app.dependency_overrides[get_db] = lambda: test_session_factory()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac
    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def auth_token(test_user: User):
    return create_access_token(test_user.id, test_user.email)


async def test_nlp_score_endpoint_exists(client: AsyncClient, auth_token: str):
    """NLP score endpoint should be registered."""
    app = create_app()
    routes = [route.path for route in app.routes]
    assert "/api/v1/nlp/score" in routes


async def test_nlp_score_requires_auth(client: AsyncClient):
    """NLP score endpoint should require authentication."""
    resp = await client.post("/api/v1/nlp/score", json={"text": "test message"})
    # HTTPBearer(auto_error=False) with explicit raise returns 401 with WWW-Authenticate
    assert resp.status_code == 401
    assert resp.headers.get("www-authenticate") == "Bearer"


async def test_nlp_score_accepts_request(client: AsyncClient, auth_token: str):
    """NLP score endpoint should accept valid requests."""
    headers = {"Authorization": f"Bearer {auth_token}"}
    with patch("app.nlp.tasks.nlp_analysis_task.kiq", new_callable=AsyncMock) as mock_queue:
        resp = await client.post(
            "/api/v1/nlp/score",
            json={"text": "URGENT: Verify your account now", "sender": "+15551234567"},
            headers=headers,
        )
        assert resp.status_code == 202
        data = resp.json()
        assert "event_id" in data
        assert data["status"] == "pending"
        assert "scam_probability" in data
        assert "risk_level" in data
        assert "patterns" in data
        assert "explanation" in data
        mock_queue.assert_called_once()


async def test_nlp_events_list(client: AsyncClient, auth_token: str):
    """Should be able to list NLP events."""
    headers = {"Authorization": f"Bearer {auth_token}"}
    with patch("app.nlp.tasks.nlp_analysis_task.kiq", new_callable=AsyncMock):
        # First create an event
        await client.post(
            "/api/v1/nlp/score",
            json={"text": "test message"},
            headers=headers,
        )
        # Then list events
        resp = await client.get("/api/v1/nlp/events", headers=headers)
        assert resp.status_code == 200
        events = resp.json()
        assert isinstance(events, list)


async def test_nlp_score_event_persistence(test_user: User):
    """NlpScoreEvent should be persisted correctly."""
    async with test_session_factory() as session:
        event = NlpScoreEvent(
            user_id=test_user.id,
            sender="+15551234567",
            message_preview="Test message",
            score=0.85,
            threshold=0.75,
            is_high_risk=True,
            status="completed",
            raw_response='{"scam_probability": 0.85}',
        )
        session.add(event)
        await session.commit()
        await session.refresh(event)

        assert event.id is not None
        assert event.user_id == test_user.id
        assert event.score == 0.85
        assert event.is_high_risk is True


async def test_run_nlp_analysis_failure_raises(test_user: User):
    """run_nlp_analysis should store risk_level=None and patterns=None when call_hf_inference raises."""
    # Create a pending event first
    async with test_session_factory() as session:
        event = NlpScoreEvent(
            user_id=test_user.id,
            sender="+15551234567",
            message_preview="Test message",
            score=0.0,
            threshold=0.75,
            is_high_risk=False,
            status="pending",
        )
        session.add(event)
        await session.commit()
        await session.refresh(event)
        event_id = event.id

    # Mock call_hf_inference to raise an exception
    with patch("app.nlp.worker.call_hf_inference", side_effect=RuntimeError("Service unavailable")):
        await run_nlp_analysis(event_id, "test message", "+15551234567", session_factory=test_session_factory)

    # Verify the event was updated with failure status
    async with test_session_factory() as session:
        event = await session.get(NlpScoreEvent, event_id)
        assert event is not None
        assert event.status == "failed"
        assert event.risk_level is None
        assert event.patterns is None
        assert event.score == 0.0
        assert event.is_high_risk is False
        assert "Analysis failed" in event.explanation
        assert event.error is not None


async def test_run_nlp_analysis_failure_invalid_json(test_user: User):
    """run_nlp_analysis should store risk_level=None and patterns=None when LLM returns invalid JSON."""
    # Create a pending event first
    async with test_session_factory() as session:
        event = NlpScoreEvent(
            user_id=test_user.id,
            sender="+15551234567",
            message_preview="Test message",
            score=0.0,
            threshold=0.75,
            is_high_risk=False,
            status="pending",
        )
        session.add(event)
        await session.commit()
        await session.refresh(event)
        event_id = event.id

    # Mock call_hf_inference to return invalid JSON (empty object missing required fields)
    with patch("app.nlp.worker.call_hf_inference", return_value="{}"):
        await run_nlp_analysis(event_id, "test message", "+15551234567", session_factory=test_session_factory)

    # Verify the event was updated with failure status
    async with test_session_factory() as session:
        event = await session.get(NlpScoreEvent, event_id)
        assert event is not None
        assert event.status == "failed"
        assert event.risk_level is None
        assert event.patterns is None
        assert event.score == 0.0
        assert event.is_high_risk is False
        assert "Analysis failed" in event.explanation
        assert event.error is not None


async def test_pending_timeout_marks_stale_pending_as_failed(test_user: User):
    """Pending events older than timeout become failed with risk_level=None, patterns=None, error set."""
    old_time = datetime.now(UTC) - timedelta(minutes=15)  # Older than default 10 min timeout
    async with test_session_factory() as session:
        event = NlpScoreEvent(
            user_id=test_user.id,
            sender="+15551234567",
            message_preview="Old pending message",
            score=0.0,
            threshold=0.75,
            is_high_risk=False,
            status="pending",
            created_at=old_time,
        )
        session.add(event)
        await session.commit()
        await session.refresh(event)
        event_id = event.id

    # Run the timeout check
    count = await mark_stale_pending_failed(session_factory=test_session_factory)
    assert count == 1

    # Verify the event was updated
    async with test_session_factory() as session:
        event = await session.get(NlpScoreEvent, event_id)
        assert event is not None
        assert event.status == "failed"
        assert event.risk_level is None
        assert event.patterns is None
        assert event.error is not None
        assert "timed out" in event.error.lower()
        assert event.explanation is not None
        assert event.is_high_risk is False


async def test_pending_timeout_fresh_pending_untouched(test_user: User):
    """Fresh pending events (within timeout) are not modified."""
    recent_time = datetime.now(UTC) - timedelta(minutes=5)  # Within default 10 min timeout
    async with test_session_factory() as session:
        event = NlpScoreEvent(
            user_id=test_user.id,
            sender="+15551234567",
            message_preview="Fresh pending message",
            score=0.0,
            threshold=0.75,
            is_high_risk=False,
            status="pending",
            created_at=recent_time,
        )
        session.add(event)
        await session.commit()
        await session.refresh(event)
        event_id = event.id

    # Run the timeout check
    count = await mark_stale_pending_failed(session_factory=test_session_factory)
    assert count == 0

    # Verify the event was NOT modified
    async with test_session_factory() as session:
        event = await session.get(NlpScoreEvent, event_id)
        assert event is not None
        assert event.status == "pending"
        assert event.risk_level is None  # Still None for pending
        assert event.patterns is None
        assert event.error is None
        assert event.explanation is None


async def test_pending_timeout_completed_untouched(test_user: User):
    """Completed events are not modified by timeout check."""
    old_time = datetime.now(UTC) - timedelta(minutes=15)
    async with test_session_factory() as session:
        event = NlpScoreEvent(
            user_id=test_user.id,
            sender="+15551234567",
            message_preview="Completed message",
            score=0.85,
            threshold=0.75,
            is_high_risk=True,
            status="completed",
            risk_level="high",
            patterns={"urgency": True},
            explanation="Detected urgency",
            created_at=old_time,
        )
        session.add(event)
        await session.commit()
        await session.refresh(event)
        event_id = event.id

    # Run the timeout check
    count = await mark_stale_pending_failed(session_factory=test_session_factory)
    assert count == 0

    # Verify the event was NOT modified
    async with test_session_factory() as session:
        event = await session.get(NlpScoreEvent, event_id)
        assert event is not None
        assert event.status == "completed"
        assert event.risk_level == "high"
        assert event.patterns == {"urgency": True}
        assert event.explanation == "Detected urgency"


async def test_pending_timeout_already_failed_untouched(test_user: User):
    """Already failed events are not modified by timeout check."""
    old_time = datetime.now(UTC) - timedelta(minutes=15)
    async with test_session_factory() as session:
        event = NlpScoreEvent(
            user_id=test_user.id,
            sender="+15551234567",
            message_preview="Failed message",
            score=0.0,
            threshold=0.75,
            is_high_risk=False,
            status="failed",
            risk_level=None,
            patterns=None,
            error="Previous error",
            explanation="Previous explanation",
            created_at=old_time,
        )
        session.add(event)
        await session.commit()
        await session.refresh(event)
        event_id = event.id

    # Run the timeout check
    count = await mark_stale_pending_failed(session_factory=test_session_factory)
    assert count == 0

    # Verify the event was NOT modified
    async with test_session_factory() as session:
        event = await session.get(NlpScoreEvent, event_id)
        assert event is not None
        assert event.status == "failed"
        assert event.risk_level is None
        assert event.patterns is None
        assert event.error == "Previous error"
        assert event.explanation == "Previous explanation"


async def test_api_returns_failed_event_with_null_risk_and_patterns(test_user: User, client: AsyncClient, auth_token: str):
    """GET /events and /events/{id} return failed events with risk_level=null, patterns=null."""
    headers = {"Authorization": f"Bearer {auth_token}"}

    # Create a failed event directly in DB
    async with test_session_factory() as session:
        event = NlpScoreEvent(
            user_id=test_user.id,
            sender="+15551234567",
            message_preview="Failed message",
            score=0.0,
            threshold=0.75,
            is_high_risk=False,
            status="failed",
            risk_level=None,
            patterns=None,
            error="Analysis failed: RuntimeError",
            explanation="Analysis timed out - no response from AI service",
        )
        session.add(event)
        await session.commit()
        await session.refresh(event)
        event_id = event.id

    # Test list endpoint
    resp = await client.get("/api/v1/nlp/events", headers=headers)
    assert resp.status_code == 200
    events = resp.json()
    assert len(events) >= 1
    failed_event = next(e for e in events if e["id"] == event_id)
    assert failed_event["status"] == "failed"
    assert failed_event["risk_level"] is None
    assert failed_event["patterns"] is None
    assert failed_event["error"] == "Analysis failed: RuntimeError"

    # Test single event endpoint
    resp = await client.get(f"/api/v1/nlp/events/{event_id}", headers=headers)
    assert resp.status_code == 200
    event_data = resp.json()
    assert event_data["status"] == "failed"
    assert event_data["risk_level"] is None
    assert event_data["patterns"] is None
    assert event_data["error"] == "Analysis failed: RuntimeError"