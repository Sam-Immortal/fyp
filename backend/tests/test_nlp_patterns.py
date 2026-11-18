"""Tests for NLP patterns column (JSON/JSONB) type handling."""
import os
import pytest
from urllib.parse import urlparse
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy import select, delete

from app.db.models import NlpScoreEvent, User, Base
from app.db.session import get_db
from app.main import create_app
from app.schemas.nlp import ScamPattern
from app.core.security import hash_password, create_access_token
from httpx import ASGITransport, AsyncClient
from tests.conftest import test_session_factory


async def test_patterns_column_sqlite():
    """Test that patterns column works correctly with SQLite (JSON stored as text)."""
    # Use SQLite in-memory for testing
    engine = create_async_engine('sqlite+aiosqlite:///:memory:', echo=False)
    async_session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with async_session_factory() as session:
        # Create event with a patterns dict
        event = NlpScoreEvent(
            user_id=1,
            sender='+15551234567',
            message_preview='Test message',
            score=0.85,
            threshold=0.75,
            is_high_risk=True,
            status='completed',
            patterns={'urgency': True, 'authority_impersonation': False}
        )
        session.add(event)
        await session.commit()
        await session.refresh(event)
        
        # Verify patterns is a dict when assigned
        assert isinstance(event.patterns, dict)
        assert event.patterns == {'urgency': True, 'authority_impersonation': False}
        
        # Now read it back
        result = await session.execute(select(NlpScoreEvent).where(NlpScoreEvent.id == event.id))
        event_read = result.scalar_one()
        
        # Verify patterns is read back as dict (not string)
        assert isinstance(event_read.patterns, dict), f"Expected dict, got {type(event_read.patterns)}"
        assert event_read.patterns == {'urgency': True, 'authority_impersonation': False}


@pytest.mark.skipif(
    not os.getenv("POSTGRES_TEST_URL"),
    reason="PostgreSQL not available for testing"
)
async def test_patterns_column_postgres():
    """Test that patterns column works correctly with PostgreSQL (native JSONB)."""
    pg_url = os.getenv("POSTGRES_TEST_URL")
    if not pg_url:
        pytest.skip("POSTGRES_TEST_URL not set")
    
    # Guard: ensure we're only running against a test database
    parsed = urlparse(pg_url)
    db_name = parsed.path.lstrip('/')
    if not db_name.endswith('_test'):
        pytest.fail(f"Refusing to run against non-test database '{db_name}' (must end with '_test')")
    
    engine = create_async_engine(pg_url, echo=False)
    async_session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    user_id = None
    event_id = None
    
    try:
        async_session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        
        async with async_session_factory() as session:
            # Create a user first
            user = User(
                email="test@example.com",
                password_hash=hash_password("testpassword"),
                full_name="Test User"
            )
            session.add(user)
            await session.commit()
            await session.refresh(user)
            user_id = user.id
            
            # Create event with a patterns dict
            event = NlpScoreEvent(
                user_id=user_id,
                sender='+15551234567',
                message_preview='Test message',
                score=0.85,
                threshold=0.75,
                is_high_risk=True,
                status='completed',
                patterns={'urgency': True, 'authority_impersonation': False, 'credential_request': True}
            )
            session.add(event)
            await session.commit()
            await session.refresh(event)
            event_id = event.id
            
            # Verify patterns is a dict when assigned
            assert isinstance(event.patterns, dict)
            
            # Now read it back
            result = await session.execute(select(NlpScoreEvent).where(NlpScoreEvent.id == event.id))
            event_read = result.scalar_one()
            
            # Verify patterns is read back as dict (native JSONB)
            assert isinstance(event_read.patterns, dict)
            assert event_read.patterns == {'urgency': True, 'authority_impersonation': False, 'credential_request': True}
    finally:
        # Cleanup: delete test data
        if event_id:
            async with async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)() as session:
                await session.execute(delete(NlpScoreEvent).where(NlpScoreEvent.id == event_id))
                await session.commit()
        if user_id:
            async with async_session_factory() as session:
                await session.execute(delete(User).where(User.id == user_id))
                await session.commit()
        
        # Drop tables
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)


@pytest.mark.asyncio
async def test_patterns_api_read_back(client: AsyncClient, test_user: User, auth_token: str):
    """Test reading patterns via API endpoint on SQLite (default)."""
    # Create an NlpScoreEvent with patterns directly in the database
    async with test_session_factory() as session:
        event = NlpScoreEvent(
            user_id=test_user.id,
            sender='+15551234567',
            message_preview='Test message',
            score=0.85,
            threshold=0.75,
            is_high_risk=True,
            status='completed',
            patterns={'urgency': True, 'authority_impersonation': False, 'credential_request': True}
        )
        session.add(event)
        await session.commit()
        await session.refresh(event)
        event_id = event.id

    headers = {"Authorization": f"Bearer {auth_token}"}
    resp = await client.get("/api/v1/nlp/events", headers=headers)
    assert resp.status_code == 200
    events = resp.json()
    assert len(events) >= 1

    # Find our event
    found = False
    for e in events:
        if e['id'] == event_id:
            found = True
            assert 'patterns' in e
            assert isinstance(e['patterns'], dict)
            assert e['patterns']['urgency'] is True
            assert e['patterns']['authority_impersonation'] is False
            assert e['patterns']['credential_request'] is True
            break
    assert found, "Created event not found in API response"


@pytest.mark.skipif(
    not os.getenv("POSTGRES_TEST_URL"),
    reason="PostgreSQL not available for testing"
)
@pytest.mark.asyncio
async def test_patterns_api_read_back_postgres():
    """Test reading patterns via API endpoint on PostgreSQL (native JSONB)."""
    pg_url = os.getenv("POSTGRES_TEST_URL")
    if not pg_url:
        pytest.skip("POSTGRES_TEST_URL not set")

    # Guard: ensure we're only running against a test database
    parsed = urlparse(pg_url)
    db_name = parsed.path.lstrip('/')
    if not db_name.endswith('_test'):
        pytest.fail(f"Refusing to run against non-test database '{db_name}' (must end with '_test')")

    # Create a separate engine for PostgreSQL test (inside the event loop)
    pg_engine = create_async_engine(pg_url, echo=False, pool_pre_ping=True)
    pg_session_factory = async_sessionmaker(pg_engine, class_=AsyncSession, expire_on_commit=False)

    # Use unique identifiers to avoid conflicts with previous runs
    import uuid
    unique_id = uuid.uuid4().hex[:8]
    test_email = f"test_pg_{unique_id}@example.com"

    try:
        # Create tables
        async with pg_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        # Create app with PostgreSQL database
        app = create_app()

        # Async generator for dependency override - creates session in the correct event loop
        async def override_get_db_pg():
            async with pg_session_factory() as session:
                yield session

        app.dependency_overrides[get_db] = override_get_db_pg

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as ac:
            # Create user
            async with pg_session_factory() as session:
                user = User(
                    email=test_email,
                    password_hash=hash_password("testpassword"),
                    full_name="Test PG User"
                )
                session.add(user)
                await session.commit()
                await session.refresh(user)
                user_id = user.id
                user_email = user.email

            # Create event with patterns
            async with pg_session_factory() as session:
                event = NlpScoreEvent(
                    user_id=user_id,
                    sender='+15551234567',
                    message_preview='Test message PG',
                    score=0.9,
                    threshold=0.75,
                    is_high_risk=True,
                    status='completed',
                    patterns={'urgency': True, 'threat': True, 'reward': False}
                )
                session.add(event)
                await session.commit()
                await session.refresh(event)
                event_id = event.id

            # Get auth token
            token = create_access_token(user_id, user_email)
            headers = {"Authorization": f"Bearer {token}"}

            # Call API
            resp = await ac.get("/api/v1/nlp/events", headers=headers)
            assert resp.status_code == 200
            events = resp.json()
            assert len(events) >= 1

            # Find our event
            found = False
            for e in events:
                if e['id'] == event_id:
                    found = True
                    assert 'patterns' in e
                    assert isinstance(e['patterns'], dict)
                    assert e['patterns']['urgency'] is True
                    assert e['patterns']['threat'] is True
                    assert e['patterns']['reward'] is False
                    break
            assert found, "Created event not found in API response"

    finally:
        # Cleanup
        async with pg_engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        await pg_engine.dispose()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])