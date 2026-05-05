# app/routers/users.py

from fastapi import APIRouter, Depends, HTTPException, Query, status
import logging
from sqlalchemy.exc import IntegrityError
from uuid import UUID
from typing import List, Optional
from sqlalchemy.orm import Session

from app.schemas.user import (
    UserCreate,
    UserListResponse,
    UserTierUpdate,
    UserUpdate,
    UserResponse,
)
from app.core.database import get_db
from app.core.pricing_plans import PRICING_PLANS, normalize_tier
from app.core.security import get_current_user, require_admin_user
from app.services.user_service import UserService

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/me", response_model=UserResponse)
def get_me(current_user=Depends(get_current_user)):
    """Get the currently authenticated user's profile."""
    return current_user


@router.post("/", response_model=UserResponse)
def create_user(payload: UserCreate, db: Session = Depends(get_db)):
    """Create a new user (admin/public)."""
    service = UserService(db)
    logger.info("create_user called for email=%s", payload.email)
    existing = service.get_user_by_email(payload.email)
    if existing:
        logger.warning("create_user: email already registered: %s", payload.email)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered")

    try:
        user = service.create_user(
            email=payload.email,
            full_name=payload.full_name,
            password=payload.password,
        )
        logger.info("User created: id=%s email=%s", getattr(user, "id", "<unknown>"), payload.email)
        return user
    except IntegrityError as e:
        logger.exception("IntegrityError creating user: %s", payload.email)
        try:
            db.rollback()
        except Exception:
            logger.exception("Failed to rollback DB session after IntegrityError")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered")
    except Exception:
        logger.exception("Unexpected error creating user: %s", payload.email)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@router.get("/", response_model=UserListResponse)
def list_users(
    q: Optional[str] = Query(None, description="Search by email or full name"),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    _admin_user=Depends(require_admin_user),
    db: Session = Depends(get_db),
):
    """List/search users. Admin-only."""
    service = UserService(db)

    logger.info("list_users called by admin; search=%s limit=%s offset=%s", q, limit, offset)
    users, total = service.list_users(
        search=q,
        limit=limit,
        offset=offset,
    )

    return {
        "items": users,
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get("/by-email", response_model=UserResponse)
def get_user_by_email(email: str = Query(..., description="User email"), db: Session = Depends(get_db)):
    """Get a user by email."""
    service = UserService(db)
    logger.info("get_user_by_email email=%s", email)
    user = service.get_user_by_email(email)
    if not user:
        logger.warning("get_user_by_email: user not found: %s", email)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return user


@router.patch("/{user_id}/tier", response_model=UserResponse)
def update_user_tier(
    user_id: UUID,
    payload: UserTierUpdate,
    _admin_user=Depends(require_admin_user),
    db: Session = Depends(get_db),
):
    """Update a user's pricing tier. Admin-only."""
    normalized_tier = normalize_tier(payload.tier)
    if payload.tier.strip().lower() not in PRICING_PLANS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid tier. Use one of: basic, intermediate, enterprise",
        )

    service = UserService(db)
    user = service.get_user(user_id)
    if not user:
        logger.warning("update_user_tier: user not found: %s", user_id)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    logger.info("update_user_tier: user_id=%s tier=%s", user_id, normalized_tier)
    try:
        return service.update_user_tier(user, normalized_tier)
    except Exception:
        logger.exception("Failed to update user tier for user_id=%s", user_id)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@router.get("/{user_id}", response_model=UserResponse)
def get_user(user_id: UUID, db: Session = Depends(get_db)):
    """Get a user by ID."""
    service = UserService(db)
    user = service.get_user(user_id)
    if not user:
        logger.warning("get_user: user not found: %s", user_id)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return user


@router.put("/{user_id}", response_model=UserResponse)
def update_user(user_id: UUID, payload: UserUpdate, db: Session = Depends(get_db)):
    """Update a user's profile."""
    service = UserService(db)
    user = service.get_user(user_id)
    if not user:
        logger.warning("update_user: user not found: %s", user_id)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    if payload.full_name is not None:
        user.full_name = payload.full_name
    if payload.email is not None:
        # Optional: check for duplicates
        existing = service.get_user_by_email(payload.email)
        if existing and existing.id != user_id:
            logger.warning("update_user: email already in use: %s by user_id=%s", payload.email, user_id)
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already in use")
        user.email = payload.email
    logger.info("update_user: updating user_id=%s", user_id)
    try:
        return service.update_user(
            user,
            full_name=payload.full_name,
            email=payload.email,
            password=payload.password,
        )
    except IntegrityError:
        logger.exception("IntegrityError updating user: %s", user_id)
        try:
            db.rollback()
        except Exception:
            logger.exception("Failed to rollback DB session after IntegrityError on update")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already in use")
    except Exception:
        logger.exception("Unexpected error updating user: %s", user_id)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")
