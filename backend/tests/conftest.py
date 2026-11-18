"""Test configuration for using SQLite in-memory database."""

from __future__ import annotations

import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.security import create_access_token, hash_password
from app.db.models import User
from app.db.session import get_db
from app.main import create_app

# Override to use SQLite for tests
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

test_engine = create_async_engine(TEST_DATABASE_URL, echo=False, future=True)
test_session_factory = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)


@pytest_asyncio.fixture(autouse=True)
async def setup_db():
    async with test_engine.begin() as conn:
        from app.db.base import Base

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
    app.dependency_overrides[get_db] = lambda: test_session_factory()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac
    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def test_user(test_db_session: AsyncSession):
    user = User(
        email="test@example.com",
        password_hash=hash_password("StrongPass123!"),
        full_name="Test User",
    )
    test_db_session.add(user)
    await test_db_session.commit()
    await test_db_session.refresh(user)
    yield user


@pytest_asyncio.fixture
async def auth_token(test_user: User):
    return create_access_token(test_user.id, test_user.email)
