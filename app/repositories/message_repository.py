# app/repositories/message_repository.py

from typing import Optional, List, Dict
from uuid import UUID
from sqlalchemy.orm import Session

from app.shared.models.message import Message
import logging

logger = logging.getLogger(__name__)


class MessageRepository:
    """Data access layer for Message."""

    def __init__(self, db: Session):
        self.db = db

    def create_user_message(
        self,
        session_id: UUID,
        content: Optional[str],
        modality: str = "text",
        grade_level: Optional[str] = None,
        audio_url: Optional[str] = None,
        transcript: Optional[str] = None,
        audio_duration_sec: Optional[float] = None,
    ) -> Message:
        msg = Message(
            session_id=session_id,
            role="user",
            modality=modality,
            content=content,
            grade_level=grade_level,
            audio_url=audio_url,
            transcript=transcript,
            audio_duration_sec=audio_duration_sec,
        )
        logger.debug("Persisting new user message to DB (session=%s)", session_id)
        self.db.add(msg)
        self.db.flush()
        logger.debug("Message persisted with id=%s", getattr(msg, "id", None))
        return msg

    def create_system_message(self, session_id: UUID, content: Optional[str]) -> Message:
        msg = Message(session_id=session_id, role="system", modality="text", content=content)
        self.db.add(msg)
        self.db.commit()
        self.db.refresh(msg)
        return msg

    def create_assistant_message(
        self,
        session_id: UUID,
        content: Optional[str],
        model_info: Optional[Dict] = None,
        grade_level: Optional[str] = None,
        parent_msg_id: Optional[UUID] = None,
    ) -> Message:
        msg = Message(
            session_id=session_id,
            role="assistant",
            modality="text",
            grade_level=grade_level,
            content=content,
            model_name=(model_info or {}).get("model_name"),
            prompt_tokens=(model_info or {}).get("prompt_tokens"),
            completion_tokens=(model_info or {}).get("completion_tokens"),
            total_tokens=(model_info or {}).get("total_tokens"),
            parent_msg_id=parent_msg_id,
        )
        self.db.add(msg)
        self.db.commit()
        self.db.refresh(msg)
        return msg

    def list_session_messages(self, session_id: UUID) -> List[Message]:
        return (
            self.db.query(Message)
            .filter(Message.session_id == session_id)
            .order_by(Message.created_at.asc())
            .all()
        )

    def list_session_messages_with_attachments(self, session_id: UUID) -> List[Message]:
        """List messages with eager-loaded attachments and resource details."""
        from sqlalchemy.orm import joinedload
        from app.shared.models.message_relations import MessageAttachment
        from app.shared.models.resource_file import ResourceFile
        
        return (
            self.db.query(Message)
            .filter(Message.session_id == session_id)
            .options(
                joinedload(Message.attachments).joinedload(MessageAttachment.resource)
            )
            .order_by(Message.created_at.asc())
            .all()
        )
