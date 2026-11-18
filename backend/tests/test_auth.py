from __future__ import annotations

import jwt
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from datetime import UTC, datetime, timedelta

from app.core.security import create_access_token, create_refresh_token, decode_token, hash_password
from app.db.models import User
from app.db.session import get_db
from app.main import create_app

TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

test_engine = create_async_engine(TEST_DATABASE_URL, echo=False, future=True)
test_session_factory = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)


async def override_get_db():
    async with test_session_factory() as session:
        yield session


@pytest_asyncio.fixture(autouse=True)
async def setup_db():
    from app.db.base import Base

    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture
async def test_db_session() -> AsyncSession:
    async with test_session_factory() as session:
        yield session


@pytest_asyncio.fixture
async def client():
    app = create_app()
    app.dependency_overrides[get_db] = override_get_db
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac
    app.dependency_overrides.clear()


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


async def test_register(client: AsyncClient):
    resp = await client.post(
        "/auth/register",
        json={"email": "new@example.com", "password": "Str0ngP@ss", "full_name": "New User"},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["email"] == "new@example.com"
    assert "id" in data


async def test_register_duplicate_email(client: AsyncClient, test_user: User):
    resp = await client.post(
        "/auth/register",
        json={"email": "test@example.com", "password": "Str0ngP@ss", "full_name": "Dup"},
    )
    assert resp.status_code == 409


async def test_login_success(client: AsyncClient, test_user: User):
    resp = await client.post(
        "/auth/login",
        json={"email": "test@example.com", "password": "Str0ngP@ss"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["token_type"] == "bearer"
    assert data["expires_in"] == 900


async def test_login_wrong_password(client: AsyncClient, test_user: User):
    resp = await client.post(
        "/auth/login",
        json={"email": "test@example.com", "password": "WrongPass"},
    )
    assert resp.status_code == 401


async def test_me_endpoint(client: AsyncClient, test_user: User):
    login = await client.post(
        "/auth/login",
        json={"email": "test@example.com", "password": "Str0ngP@ss"},
    )
    access = login.json()["access_token"]
    resp = await client.get("/auth/me", headers={"Authorization": f"Bearer {access}"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["user"]["email"] == "test@example.com"
    assert data["user"]["full_name"] == "Test User"


async def test_me_invalid_token(client: AsyncClient):
    resp = await client.get("/auth/me", headers={"Authorization": "Bearer invalid.token.here"})
    assert resp.status_code == 401


async def test_refresh_rotation(client: AsyncClient, test_user: User):
    login = await client.post(
        "/auth/login",
        json={"email": "test@example.com", "password": "Str0ngP@ss"},
    )
    tokens = login.json()
    old_refresh = tokens["refresh_token"]

    resp = await client.post("/auth/refresh", json={"refresh_token": old_refresh})
    assert resp.status_code == 200
    new_tokens = resp.json()
    assert new_tokens["refresh_token"] != old_refresh

    resp2 = await client.post("/auth/refresh", json={"refresh_token": old_refresh})
    assert resp2.status_code == 401


async def test_refresh_invalid_token_returns_401_not_500(client: AsyncClient):
    """Regression test: invalid/unknown refresh token should return 401, not crash with 500."""
    resp = await client.post("/auth/refresh", json={"refresh_token": "not.a.real.token"})
    assert resp.status_code == 401
    data = resp.json()
    assert data["detail"] == "Refresh token revoked or expired"


async def test_logout_revokes_refresh(client: AsyncClient, test_user: User):
    login = await client.post(
        "/auth/login",
        json={"email": "test@example.com", "password": "Str0ngP@ss"},
    )
    tokens = login.json()
    refresh = tokens["refresh_token"]

    resp = await client.post("/auth/logout", json={"refresh_token": refresh})
    assert resp.status_code == 204

    resp2 = await client.post("/auth/refresh", json={"refresh_token": refresh})
    assert resp2.status_code == 401


async def test_protected_route_without_auth(client: AsyncClient):
    resp = await client.get("/auth/me")
    assert resp.status_code == 401
    assert resp.headers.get("www-authenticate") == "Bearer"


async def test_malformed_token_string(client: AsyncClient):
    """Malformed token string returns 401 with WWW-Authenticate header."""
    resp = await client.get("/auth/me", headers={"Authorization": "Bearer not.a.valid.token"})
    assert resp.status_code == 401
    assert resp.headers.get("www-authenticate") == "Bearer"


async def test_expired_token(client: AsyncClient, test_user: User):
    """Expired token returns 401 with WWW-Authenticate header."""
    # Create an expired token (exp 1 hour ago) manually
    from app.core.config import get_settings
    settings = get_settings()
    payload = {
        "sub": str(test_user.id),
        "email": test_user.email,
        "type": "access",
        "iat": int((datetime.now(UTC) - timedelta(hours=2)).timestamp()),
        "exp": int((datetime.now(UTC) - timedelta(hours=1)).timestamp()),
    }
    expired_token = jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)
    resp = await client.get("/auth/me", headers={"Authorization": f"Bearer {expired_token}"})
    assert resp.status_code == 401
    assert resp.headers.get("www-authenticate") == "Bearer"


async def test_refresh_token_used_as_access_token(client: AsyncClient, test_user: User):
    """Raw refresh token (non-JWT) used as access token returns 401 with WWW-Authenticate header."""
    login = await client.post(
        "/auth/login",
        json={"email": "test@example.com", "password": "Str0ngP@ss"},
    )
    refresh_token = login.json()["refresh_token"]
    resp = await client.get("/auth/me", headers={"Authorization": f"Bearer {refresh_token}"})
    assert resp.status_code == 401
    assert resp.headers.get("www-authenticate") == "Bearer"


async def test_refresh_token_jwt_used_as_access_token(client: AsyncClient, test_user: User):
    """Properly signed JWT with type=refresh used as access token returns 401 'Not an access token'."""
    from app.core.config import get_settings
    settings = get_settings()
    payload = {
        "sub": str(test_user.id),
        "email": test_user.email,
        "type": "refresh",
        "iat": int(datetime.now(UTC).timestamp()),
        "exp": int((datetime.now(UTC) + timedelta(hours=1)).timestamp()),
    }
    refresh_jwt = jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)
    resp = await client.get("/auth/me", headers={"Authorization": f"Bearer {refresh_jwt}"})
    assert resp.status_code == 401
    assert resp.headers.get("www-authenticate") == "Bearer"
    assert resp.json()["detail"] == "Not an access token"


async def test_missing_sub_claim(client: AsyncClient):
    """Token with missing 'sub' claim returns 401 with WWW-Authenticate header."""
    from app.core.config import get_settings
    settings = get_settings()
    # Create a token without 'sub' claim
    payload = {"type": "access", "email": "test@example.com", "exp": int((datetime.now(UTC) + timedelta(hours=1)).timestamp())}
    token = jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)
    resp = await client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 401
    assert resp.headers.get("www-authenticate") == "Bearer"


async def test_non_numeric_sub_claim(client: AsyncClient):
    """Token with non-numeric 'sub' claim returns 401 with WWW-Authenticate header."""
    from app.core.config import get_settings
    settings = get_settings()
    # Create a token with non-numeric 'sub'
    payload = {"type": "access", "sub": "not-a-number", "exp": int((datetime.now(UTC) + timedelta(hours=1)).timestamp())}
    token = jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)
    resp = await client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 401
    assert resp.headers.get("www-authenticate") == "Bearer"
