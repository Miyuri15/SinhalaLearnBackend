# app/services/user_service.py

from typing import Optional
from uuid import UUID
from sqlalchemy.orm import Session

from app.repositories.user_repository import UserRepository
from app.core.security import get_password_hash


class UserService:
    """Business logic for users."""

    def __init__(self, db: Session):
        self.repository = UserRepository(db)

    def get_user(self, user_id: UUID):
        return self.repository.get_user(user_id)

    def get_user_by_email(self, email: str):
        return self.repository.get_user_by_email(email)

    def list_users(
        self,
        search: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ):
        return self.repository.list_users_with_count(search=search, limit=limit, offset=offset)

    def create_user(
        self,
        email: str,
        full_name: Optional[str],
        password: str,
        role: Optional[str] = None,
    ):
        password_hash = get_password_hash(password)
        return self.repository.create_user(email, full_name, password_hash, role=role)

    def admin_exists(self) -> bool:
        return self.repository.admin_exists()

    def update_user(self, user, full_name: Optional[str] = None, email: Optional[str] = None, password: Optional[str] = None):
        if full_name is not None:
            user.full_name = full_name
        if email is not None:
            user.email = email
        if password is not None:
            user.password_hash = get_password_hash(password)
        self.repository.db.add(user)
        self.repository.db.commit()
        self.repository.db.refresh(user)
        return user

    def update_user_tier(self, user, tier: str):
        user.tier = tier
        self.repository.db.add(user)
        self.repository.db.commit()
        self.repository.db.refresh(user)
        return user

    def bootstrap_admin(self, email: str, full_name: Optional[str], password: str):
        from app.shared.models.user import ADMIN_ROLE

        existing = self.get_user_by_email(email)
        password_hash = get_password_hash(password)
        if existing:
            existing.full_name = full_name if full_name is not None else existing.full_name
            existing.password_hash = password_hash
            existing.role = ADMIN_ROLE
            self.repository.db.add(existing)
            self.repository.db.commit()
            self.repository.db.refresh(existing)
            return existing

        return self.repository.create_user(
            email=email,
            full_name=full_name,
            password_hash=password_hash,
            role=ADMIN_ROLE,
        )
