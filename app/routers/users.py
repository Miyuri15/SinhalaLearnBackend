# app/routers/users.py

from fastapi import APIRouter, Depends, HTTPException, Query, status
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


@router.get("/me", response_model=UserResponse)
def get_me(current_user=Depends(get_current_user)):
    """Get the currently authenticated user's profile."""
    return current_user


@router.post("/", response_model=UserResponse)
def create_user(payload: UserCreate, db: Session = Depends(get_db)):
    """Create a new user (admin/public)."""
    service = UserService(db)
    existing = service.get_user_by_email(payload.email)
    if existing:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered")

    return service.create_user(
        email=payload.email,
        full_name=payload.full_name,
        password=payload.password,
    )


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
    user = service.get_user_by_email(email)
    if not user:
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
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    return service.update_user_tier(user, normalized_tier)


@router.get("/{user_id}", response_model=UserResponse)
def get_user(user_id: UUID, db: Session = Depends(get_db)):
    """Get a user by ID."""
    service = UserService(db)
    user = service.get_user(user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return user


@router.put("/{user_id}", response_model=UserResponse)
def update_user(user_id: UUID, payload: UserUpdate, db: Session = Depends(get_db)):
    """Update a user's profile."""
    service = UserService(db)
    user = service.get_user(user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    if payload.full_name is not None:
        user.full_name = payload.full_name
    if payload.email is not None:
        # Optional: check for duplicates
        existing = service.get_user_by_email(payload.email)
        if existing and existing.id != user_id:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already in use")
        user.email = payload.email
    return service.update_user(
        user,
        full_name=payload.full_name,
        email=payload.email,
        password=payload.password,
    )
