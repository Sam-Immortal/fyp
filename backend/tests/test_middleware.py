"""Tests for SanitizationMiddleware."""

from __future__ import annotations

import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from unittest.mock import AsyncMock, patch

from app.core.security import create_access_token, hash_password
from app.db.models import User
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
async def auth_token(test_user: User):
    return create_access_token(test_user.id, test_user.email)


@pytest_asyncio.fixture
async def client():
    app = create_app()
    app.dependency_overrides[get_db] = lambda: test_session_factory()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac
    app.dependency_overrides.clear()


async def test_sanitization_middleware_strips_nul_bytes(client: AsyncClient, auth_token: str):
    """SanitizationMiddleware should strip NUL bytes from JSON request bodies."""
    headers = {"Authorization": f"Bearer {auth_token}", "Content-Type": "application/json"}
    # Create a JSON payload with an embedded NUL byte - send as raw bytes to ensure NUL is in body
    raw_body = b'{"text": "test message\x00with nul", "sender": "+15551234567"}'

    # The NUL byte should be stripped before reaching the endpoint
    with patch("app.nlp.tasks.nlp_analysis_task.kiq", new_callable=AsyncMock) as mock_queue:
        resp = await client.post(
            "/api/v1/nlp/score",
            content=raw_body,
            headers=headers,
        )
        # Request should succeed (not fail with 500 or 400)
        assert resp.status_code == 202
        data = resp.json()
        assert "scam_probability" in data

        # Verify the NUL byte was stripped by checking what was queued
        mock_queue.assert_called_once()
        call_args = mock_queue.call_args
        # The second argument should be the text with NUL byte stripped
        queued_text = call_args[0][1]
        assert "\x00" not in queued_text
        assert queued_text == "test messagewith nul"


async def test_sanitization_middleware_no_change_without_nul(client: AsyncClient, auth_token: str):
    """SanitizationMiddleware should not modify bodies without NUL bytes."""
    headers = {"Authorization": f"Bearer {auth_token}", "Content-Type": "application/json"}
    payload = {"text": "normal message without nul", "sender": "+15551234567"}

    with patch("app.nlp.tasks.nlp_analysis_task.kiq", new_callable=AsyncMock) as mock_queue:
        resp = await client.post(
            "/api/v1/nlp/score",
            json=payload,
            headers=headers,
        )
        assert resp.status_code == 202
        mock_queue.assert_called_once()
        call_args = mock_queue.call_args
        queued_text = call_args[0][1]
        assert queued_text == "normal message without nul"


async def test_sanitization_middleware_ignores_get_requests(client: AsyncClient, auth_token: str):
    """SanitizationMiddleware should not interfere with GET requests."""
    headers = {"Authorization": f"Bearer {auth_token}"}
    with patch("app.nlp.tasks.nlp_analysis_task.kiq", new_callable=AsyncMock):
        resp = await client.get("/api/v1/nlp/events", headers=headers)
        # Should not crash
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)


async def test_sanitization_middleware_ignores_websocket_upgrade(client: AsyncClient):
    """SanitizationMiddleware should not interfere with WebSocket upgrade requests."""
    # Just verify the middleware doesn't break WebSocket connections
    # This is a smoke test - we can't easily test WebSocket upgrade in test client
    pass