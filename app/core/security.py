from datetime import datetime, timedelta, timezone
import hashlib
import uuid
from typing import Any
from uuid import UUID

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import jwt, JWTError
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import get_db
from app.repositories.user_repository import UserRepository
from app.shared.models.user import ADMIN_ROLE

pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
http_bearer = HTTPBearer(auto_error=True)


def get_password_hash(password: str) -> str:
    password_bytes = password.encode("utf-8")
    sha256_hash = hashlib.sha256(password_bytes).digest()
    return pwd_context.hash(sha256_hash)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    password_bytes = plain_password.encode("utf-8")
    sha256_hash = hashlib.sha256(password_bytes).digest()
    return pwd_context.verify(sha256_hash, hashed_password)


def _create_token(data: dict[str, Any], expires_delta: timedelta) -> tuple[str, str]:
    """Create a JWT token and return (token, jti)."""
    jti = str(uuid.uuid4())
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + expires_delta
    to_encode.update({"exp": expire, "jti": jti})
    token = jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    return token, jti


def create_access_token(user_id: UUID) -> tuple[str, int]:
    token, _ = _create_token(
        {"sub": str(user_id), "type": "access"},
        timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    expires_in = settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
    return token, expires_in


def create_refresh_token(user_id: UUID) -> tuple[str, str, datetime]:
    """Create refresh token and return (token, jti, expires_at)."""
    token, jti = _create_token(
        {"sub": str(user_id), "type": "refresh"},
        timedelta(minutes=settings.REFRESH_TOKEN_EXPIRE_MINUTES),
    )
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=settings.REFRESH_TOKEN_EXPIRE_MINUTES)
    return token, jti, expires_at


def create_password_reset_token(user_id: UUID) -> tuple[str, str, datetime]:
    """Create password reset token and return (token, jti, expires_at) for DB storage."""
    token, jti = _create_token(
        {"sub": str(user_id), "type": "reset"},
        timedelta(minutes=settings.PASSWORD_RESET_TOKEN_EXPIRE_MINUTES),
    )
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=settings.PASSWORD_RESET_TOKEN_EXPIRE_MINUTES)
    return token, jti, expires_at


def decode_token(token: str) -> dict[str, Any]:
    return jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(http_bearer),
    db: Session = Depends(get_db),
):
    token = credentials.credentials
    try:
        payload = decode_token(token)
        if payload.get("type") != "access":
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token type")
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token payload")
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")

    repo = UserRepository(db)
    user = repo.get_user(UUID(user_id))
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user


def require_admin_user(current_user=Depends(get_current_user)):
    if getattr(current_user, "role", None) != ADMIN_ROLE:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return current_user
