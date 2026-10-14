"""Tests for NLP scam classification endpoint."""

from __future__ import annotations

import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from unittest.mock import AsyncMock, patch

from app.core.security import create_access_token, hash_password
from app.db.models import NlpScoreEvent, User
from app.db.session import get_db
from app.main import create_app
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
    assert resp.status_code == 401


async def test_nlp_score_accepts_request(client: AsyncClient, auth_token: str):
    """NLP score endpoint should accept valid requests."""
    headers = {"Authorization": f"Bearer {auth_token}"}
    with patch("app.api.nlp.queue_nlp_analysis", new_callable=AsyncMock) as mock_queue:
        resp = await client.post(
            "/api/v1/nlp/score",
            json={"text": "URGENT: Verify your account now", "sender": "+15551234567"},
            headers=headers,
        )
        assert resp.status_code == 202
        data = resp.json()
        assert "scam_probability" in data
        assert "risk_level" in data
        assert "patterns" in data
        assert "explanation" in data
        mock_queue.assert_called_once()


async def test_nlp_events_list(client: AsyncClient, auth_token: str):
    """Should be able to list NLP events."""
    headers = {"Authorization": f"Bearer {auth_token}"}
    with patch("app.api.nlp.queue_nlp_analysis", new_callable=AsyncMock):
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
            raw_response='{"scam_probability": 0.85}',
        )
        session.add(event)
        await session.commit()
        await session.refresh(event)

        assert event.id is not None
        assert event.user_id == test_user.id
        assert event.score == 0.85
        assert event.is_high_risk is True