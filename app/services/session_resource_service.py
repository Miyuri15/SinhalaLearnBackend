# app/services/session_resource_service.py

from typing import List
from uuid import UUID
from sqlalchemy.orm import Session

from app.repositories.session_resource_repository import SessionResourceRepository


class SessionResourceService:
    """Business logic for session-resource linking."""

    def __init__(self, db: Session):
        self.repository = SessionResourceRepository(db)

    def attach_resource_to_session(self, session_id: UUID, resource_id: UUID, label: str = None):
        return self.repository.attach_resource_to_session(session_id, resource_id, label=label)

    def get_session_resources(self, session_id: UUID) -> List:
        return self.repository.get_session_resources(session_id)

    def get_resources_by_session_id(self, session_id: UUID) -> List:
        return self.repository.get_resources_by_session_id(session_id)

    def get_session_resource_by_label(self, session_id: UUID, label: str):
        return self.repository.get_session_resource_by_label(session_id, label)

    def upsert_session_resource(self, session_id: UUID, resource_id: UUID, label: str):
        return self.repository.upsert_session_resource(session_id, resource_id, label)

    def detach_all_resources(self, session_id: UUID) -> int:
        return self.repository.delete_resources_for_session(session_id)

    def detach_resource(self, session_id: UUID, resource_id: UUID, label: str = None) -> bool:
        """Remove a single resource link from a session. Returns True if found and removed."""
        return self.repository.detach_resource_from_session(session_id, resource_id, label)

    def detach_resources_by_label(self, session_id: UUID, label: str) -> bool:
        """Remove all resource links with a specific label from a session. Returns True if found and removed."""
        return self.repository.detach_resources_by_label(session_id, label)
    
    def get_resource_ids_by_session(self, session_id: UUID) -> List[UUID]:
        """Helper to get just the list of resource IDs for a session."""
        return self.repository.get_resource_ids_by_session(session_id)

