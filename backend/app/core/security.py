from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import jwt
from passlib.context import CryptContext
from pydantic import BaseModel

from app.core.config import get_settings

# Use pbkdf2_sha256 instead of bcrypt to avoid 72-byte limit and Windows compatibility issues
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")

_SETTINGS = get_settings()


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def generate_refresh_token() -> str:
    return secrets.token_urlsafe(48)


def hash_refresh_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def create_access_token(user_id: int, email: str) -> str:
    now = datetime.now(UTC)
    expire = now + timedelta(minutes=_SETTINGS.jwt_access_expire_minutes)
    payload = {
        "sub": str(user_id),
        "email": email,
        "type": "access",
        "iat": int(now.timestamp()),
        "exp": int(expire.timestamp()),
    }
    return jwt.encode(payload, _SETTINGS.jwt_secret, algorithm=_SETTINGS.jwt_algorithm)


def create_refresh_token(user_id: int) -> tuple[str, str]:
    raw = generate_refresh_token()
    hashed = hash_refresh_token(raw)
    return raw, hashed


def decode_token(token: str) -> dict:
    return jwt.decode(token, _SETTINGS.jwt_secret, algorithms=[_SETTINGS.jwt_algorithm])


@dataclass
class TokenPair:
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = _SETTINGS.jwt_access_expire_minutes * 60


def make_token_pair(user_id: int, email: str) -> TokenPair:
    access = create_access_token(user_id, email)
    refresh_raw, _ = create_refresh_token(user_id)
    return TokenPair(access_token=access, refresh_token=refresh_raw)


class JwtClaims(BaseModel):
    sub: str
    email: str
    type: str
    iat: int
    exp: int
    jti: str | None = None


class RefreshTokenData(BaseModel):
    jti: str
    user_id: int
    expires_at: int
