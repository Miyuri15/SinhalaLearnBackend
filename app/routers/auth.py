# app/routers/auth.py

from uuid import UUID
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
import logging

logger = logging.getLogger(__name__)

from app.schemas.auth import (
    BootstrapAdminRequest,
    SignUpRequest,
    SignInRequest,
    ForgotPasswordRequest,
    ResetPasswordRequest,
    RefreshTokenRequest,
    AuthTokensResponse,
    SignOutResponse,
)
from app.core.config import settings
from app.core.database import get_db
from app.core.security import (
    verify_password,
    create_access_token,
    create_refresh_token,
    decode_token,
    create_password_reset_token,
    get_current_user,
)
from app.services.user_service import UserService
from app.repositories.refresh_token_repository import RefreshTokenRepository
from app.repositories.password_reset_token_repository import PasswordResetTokenRepository
from app.services.email_service import EmailService

router = APIRouter()


def _auth_tokens_for_user(user, db: Session) -> AuthTokensResponse:
    access_token, expires_in = create_access_token(user.id)
    refresh_token, jti, expires_at = create_refresh_token(user.id)

    token_repo = RefreshTokenRepository(db)
    token_repo.create_token(user.id, jti, expires_at)

    return AuthTokensResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=expires_in,
        user_id=str(user.id),
        tier=getattr(user, "tier", None),
        role=getattr(user, "role", None),
    )


@router.post("/signup", response_model=AuthTokensResponse)
def signup(payload: SignUpRequest, db: Session = Depends(get_db)):
    """Register a new user account."""
    service = UserService(db)
    existing = service.get_user_by_email(payload.email)
    if existing:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered")

    user = service.create_user(
        email=payload.email,
        full_name=payload.full_name,
        password=payload.password,
    )
    
    return _auth_tokens_for_user(user, db)


@router.post("/signin", response_model=AuthTokensResponse)
def signin(payload: SignInRequest, db: Session = Depends(get_db)):
    """Sign in with email and password."""
    service = UserService(db)
    user = service.get_user_by_email(payload.email)
    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Account is inactive")

    # Update last login timestamp
    user.last_login_at = datetime.now(timezone.utc)
    db.add(user)
    db.commit()

    return _auth_tokens_for_user(user, db)


@router.post("/bootstrap-admin", response_model=AuthTokensResponse)
def bootstrap_admin(payload: BootstrapAdminRequest, db: Session = Depends(get_db)):
    """
    Create the first admin account.

    This endpoint is intentionally guarded by ADMIN_BOOTSTRAP_TOKEN and is only
    usable while no admin exists.
    """
    if not settings.ADMIN_BOOTSTRAP_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin bootstrap is not configured",
        )
    if payload.bootstrap_token != settings.ADMIN_BOOTSTRAP_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid admin bootstrap token",
        )

    service = UserService(db)
    if service.admin_exists():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An admin user already exists",
        )

    user = service.bootstrap_admin(
        email=payload.email,
        full_name=payload.full_name,
        password=payload.password,
    )
    return _auth_tokens_for_user(user, db)


@router.post("/forgot-password")
def forgot_password(payload: ForgotPasswordRequest, db: Session = Depends(get_db)):
    """Start password reset flow: generate reset token and send email.
    
    Always returns success message to prevent email enumeration attacks.
    In development, also returns the token for testing.
    """
    service = UserService(db)
    user = service.get_user_by_email(payload.email)

    # Always return generic message
    response = {"detail": "If an account exists with this email, password reset instructions have been sent."}
    
    if user:
        # Revoke any existing active reset tokens for this user
        reset_token_repo = PasswordResetTokenRepository(db)
        reset_token_repo.revoke_all_user_tokens(user.id)
        
        # Generate and store new reset token
        reset_token, jti, expires_at = create_password_reset_token(user.id)
        reset_token_repo.create_token(user.id, jti, expires_at)
        
        # Send email
        email_service = EmailService()
        email_sent = email_service.send_password_reset_email(
            to_email=user.email,
            reset_token=reset_token,
            user_name=user.full_name
        )
        
        # In development, include token in response for testing
        from app.core.config import settings
        if settings.ENV == "development":
            response["reset_token"] = reset_token
            response["email_sent"] = email_sent
    
    return response


@router.post("/reset-password")
def reset_password(payload: ResetPasswordRequest, db: Session = Depends(get_db)):
    """Complete password reset: verify token and update password."""
    # Validate token
    try:
        data = decode_token(payload.token)
        if data.get("type") != "reset":
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token type")
        user_id = data.get("sub")
        jti = data.get("jti")
        if not user_id or not jti:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token payload")
    except Exception:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")

    # Check if token is valid in DB (not used, not revoked, not expired)
    reset_token_repo = PasswordResetTokenRepository(db)
    if not reset_token_repo.is_token_valid(jti):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, 
            detail="Token has already been used, revoked, or expired"
        )

    service = UserService(db)
    user = service.get_user(UUID(user_id))
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    # Update password
    service.update_user(user, password=payload.new_password)
    
    # Mark token as used
    reset_token_repo.mark_token_as_used(jti)
    
    return {"detail": "Password reset successfully."}


@router.post("/refresh", response_model=AuthTokensResponse)
def refresh_token(payload: RefreshTokenRequest, db: Session = Depends(get_db)):
    """Issue a new access token using a refresh token."""
    try:
        data = decode_token(payload.refresh_token)
        if data.get("type") != "refresh":
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token type")
        user_id = data.get("sub")
        jti = data.get("jti")
        if not user_id or not jti:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token payload")
    except Exception:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")

    # Verify token is still active in DB
    token_repo = RefreshTokenRepository(db)
    if not token_repo.is_token_valid(jti):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token revoked or expired")

    service = UserService(db)
    user = service.get_user(UUID(user_id))
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")

    # Issue new tokens
    access_token, expires_in = create_access_token(user.id)
    refresh_token_str, new_jti, expires_at = create_refresh_token(user.id)
    
    # Store new refresh token
    token_repo.create_token(user.id, new_jti, expires_at)

    logger.info("✓ Refresh token used to issue new access token for user %s", user_id)

    return AuthTokensResponse(
        access_token=access_token,
        refresh_token=refresh_token_str,
        expires_in=expires_in,
        user_id=str(user.id),
        tier=getattr(user, "tier", None),
        role=getattr(user, "role", None),
    )


@router.post("/signout", response_model=SignOutResponse)
def signout(current_user=Depends(get_current_user), db: Session = Depends(get_db)):
    """Sign out: revoke all refresh tokens for the user."""
    token_repo = RefreshTokenRepository(db)
    token_repo.revoke_all_user_tokens(current_user.id)
    return SignOutResponse(success=True)
