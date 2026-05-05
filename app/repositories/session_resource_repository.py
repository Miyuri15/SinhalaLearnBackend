# app/repositories/session_resource_repository.py

from typing import List, Optional
from uuid import UUID
from sqlalchemy.orm import Session

from app.shared.models.session_resources import SessionResource
from app.shared.models.resource_file import ResourceFile


class SessionResourceRepository:
    """Data access for SessionResource link table."""

    def __init__(self, db: Session):
        self.db = db

    def attach_resource_to_session(self, session_id: UUID, resource_id: UUID, label: Optional[str] = None) -> SessionResource:
        link = SessionResource(session_id=session_id, resource_id=resource_id, label=label)
        self.db.add(link)
        self.db.commit()
        self.db.refresh(link)
        return link

    def get_session_resource_by_label(self, session_id: UUID, label: str) -> Optional[SessionResource]:
        return (
            self.db.query(SessionResource)
            .filter(
                SessionResource.session_id == session_id,
                SessionResource.label == label,
            )
            .first()
        )

    def upsert_session_resource(self, session_id: UUID, resource_id: UUID, label: str) -> SessionResource:
        existing = self.get_session_resource_by_label(session_id, label)
        if existing:
            existing.resource_id = resource_id
            self.db.commit()
            self.db.refresh(existing)
            return existing

        return self.attach_resource_to_session(session_id, resource_id, label=label)

    def get_session_resources(self, session_id: UUID) -> List[SessionResource]:
        return (
            self.db.query(SessionResource)
            .filter(SessionResource.session_id == session_id)
            .all()
        )

    def get_resources_by_session_id(self, session_id: UUID) -> List[dict]:
        results = (
            self.db.query(ResourceFile, SessionResource.label)
            .join(SessionResource, SessionResource.resource_id == ResourceFile.id)
            .filter(SessionResource.session_id == session_id)
            .all()
        )
        
        resources = []
        for resource_file, label in results:
            res_dict = {c.name: getattr(resource_file, c.name) for c in resource_file.__table__.columns}
            res_dict["resource_type"] = label
            resources.append(res_dict)
            
        return resources

    def delete_resources_for_session(self, session_id: UUID) -> int:
        rows = (
            self.db.query(SessionResource)
            .filter(SessionResource.session_id == session_id)
            .delete()
        )
        self.db.commit()
        return rows

    def detach_resource_from_session(self, session_id: UUID, resource_id: UUID, label: str = None) -> bool:
        """Remove a single resource link from a session. Returns True if a row was deleted."""
        query = self.db.query(SessionResource).filter(
            SessionResource.session_id == session_id,
            SessionResource.resource_id == resource_id,
        )
        if label:
            query = query.filter(SessionResource.label == label)
        rows_deleted = query.delete(synchronize_session=False)
        self.db.commit()
        return rows_deleted > 0

    def detach_resources_by_label(self, session_id: UUID, label: str) -> bool:
        """Remove all resource links with a specific label from a session. Returns True if rows were deleted."""
        rows_deleted = (
            self.db.query(SessionResource)
            .filter(
                SessionResource.session_id == session_id,
                SessionResource.label == label
            )
            .delete(synchronize_session=False)
        )
        self.db.commit()
        return rows_deleted > 0



    def get_resource_ids_by_session(self, session_id: UUID) -> List[str]:
        """Retrieve all resource IDs associated with a given session as strings."""

        results = (
            self.db.query(SessionResource.resource_id)
            .filter(SessionResource.session_id == session_id)
            .all()
        )

        return [str(resource_id) for (resource_id,) in results]