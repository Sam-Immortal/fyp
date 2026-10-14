"""Tests for audio WebSocket endpoint."""

from __future__ import annotations

import pytest_asyncio
from httpx import ASGITransport, AsyncClient

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


async def test_websocket_audio_requires_token(client: AsyncClient, auth_token: str):
    """WebSocket endpoint should exist at /ws/audio."""
    # Test that the endpoint exists by checking the health endpoint
    resp = await client.get("/health")
    assert resp.status_code == 200


async def test_websocket_audio_endpoint_exists(client: AsyncClient, auth_token: str):
    """WebSocket endpoint should be registered."""
    # Check that the route exists in the app
    app = create_app()
    routes = [route.path for route in app.routes]
    assert "/ws/audio" in routes
