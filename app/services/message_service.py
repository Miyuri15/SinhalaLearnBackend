# app/services/message_service.py
from datetime import datetime, timezone
import logging
from typing import Optional, List, Dict
from uuid import UUID
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)

from app.repositories.message_repository import MessageRepository
from app.shared.models.message import Message


class MessageService:
    """Business logic for chat messages."""

    def __init__(self, db: Session):
        self.db = db
        self.repository = MessageRepository(db)
        self.logger = logging.getLogger(__name__)

    def validate_message_payload(self, modality: str, content: Optional[str], audio_url: Optional[str]):
        """Validate message creation payload."""
        # Accept enum values gracefully by coercing to string
        modality_val = getattr(modality, "value", None) or str(modality)
        if modality_val not in ["text", "voice", "image", "file"]:
            raise ValueError(f"Invalid modality: {modality_val}")
        
        if modality == "text" and not content:
            raise ValueError("Text messages must have content")
        
        # if modality == "voice" and not audio_url:
        #     raise ValueError("Voice messages must have audio_url")

    def create_user_message_with_validation(
        self,
        session_id: UUID,
        user_id: UUID,
        content: Optional[str],
        modality: str = "text",
        grade_level: Optional[str] = None,
        audio_url: Optional[str] = None,
        transcript: Optional[str] = None,
        audio_duration_sec: Optional[float] = None,
        session_service = None,
    ):
        """Create user message with session ownership validation."""
        # Import here to avoid circular dependency
        if session_service is None:
            from app.services.chat_session_service import ChatSessionService
            session_service = ChatSessionService(self.db)
        
        try:
            self.logger.debug(" start: session_id=%s user_id=%s modality=%s", session_id, user_id, modality)
            # If session doesn't exist, create it on-demand using sensible defaults.
            # Otherwise verify ownership.
            session = session_service.get_session(session_id)
            if not session:
                # create session owned by this user. Defaults mirror ChatSessionCreate schema
                session = session_service.create_session(
                    user_id=user_id,
                    mode="learning",
                    channel="text",
                    title=None,
                    description=None,
                    grade=None,
                    subject=None,
                )
                session_id = session.id
            else:
                # ownership check
                if not session_service.validate_ownership(session_id, user_id):
                    raise PermissionError("You don't have permission to access this session")
            
            # Validate message payload (coercion handled in validator)
            self.validate_message_payload(modality, content, audio_url)
            
            # Create message
            self.logger.debug("Creating user message for session_id=%s user_id=%s", session_id, user_id)
            msg = self.repository.create_user_message(
                session_id=session_id,
                content=content,
                modality=modality,
                grade_level=grade_level,
                audio_url=audio_url,
                transcript=transcript,
                audio_duration_sec=audio_duration_sec,
            )
            session.updated_at = datetime.now(timezone.utc)
            self.db.commit()
            self.db.refresh(msg)
            self.logger.info("User message created: %s (session=%s)", getattr(msg, "id", None), session_id)
            return msg
        except PermissionError as e:
            self.db.rollback()
            self.logger.warning("Permission denied for user %s on session %s", user_id, session_id)
            raise

        except ValueError as e:
            self.db.rollback()
            self.logger.warning("Validation error: %s", str(e))
            raise

        except SQLAlchemyError as e:
            self.db.rollback()
            self.logger.exception("Database error while creating message")
            raise

        except Exception as e:
            self.db.rollback()
            self.logger.exception("Unexpected error while creating message")
            raise

    def create_user_message(
        self,
        session_id: UUID,
        content: Optional[str],
        modality: str = "text",
        grade_level: Optional[str] = None,
        audio_url: Optional[str] = None,
        transcript: Optional[str] = None,
        audio_duration_sec: Optional[float] = None,
    ):
        return self.repository.create_user_message(
            session_id=session_id,
            content=content,
            modality=modality,
            grade_level=grade_level,
            audio_url=audio_url,
            transcript=transcript,
            audio_duration_sec=audio_duration_sec,
        )

    def create_system_message(self, session_id: UUID, content: Optional[str]):
        return self.repository.create_system_message(session_id, content)

    def create_assistant_message(
        self,
        session_id: UUID,
        content: Optional[str],
        model_info: Optional[Dict] = None,
        grade_level: Optional[str] = None,
        parent_msg_id: Optional[UUID] = None,
    ):
        return self.repository.create_assistant_message(
            session_id=session_id,
            content=content,
            model_info=model_info,
            grade_level=grade_level,
            parent_msg_id=parent_msg_id,
        )

    def list_session_messages(self, session_id: UUID) -> List:
        return self.repository.list_session_messages(session_id)
    
    def list_session_messages_with_attachments(self, session_id: UUID) -> List:
        """List messages with attachments and resource details."""
        return self.repository.list_session_messages_with_attachments(session_id)
    
    def get_message(self, message_id: UUID) -> Optional[Message]:
        """Get a single message by ID."""
        return self.db.query(Message).filter(Message.id == message_id).first()
    
    def get_message_with_ownership_check(self, message_id: UUID, user_id: UUID, session_service=None):
        """Get message and validate session ownership."""
        message = self.get_message(message_id)
        if not message:
            raise ValueError("Message not found")
        
        # Import here to avoid circular dependency
        if session_service is None:
            from app.services.chat_session_service import ChatSessionService
            session_service = ChatSessionService(self.db)
        
        # Validate session ownership
        if not session_service.validate_ownership(message.session_id, user_id):
            raise PermissionError("You don't have permission to access this message")
        
        return message
    
    def get_message_details(self, message_id: UUID, user_id: UUID):
        """Get message with all related details."""
        message = self.get_message_with_ownership_check(message_id, user_id)
        
        # Get additional details
        # TODO: consolidate these service calls into a single repo join to reduce round trips
        from app.services.message_attachment_service import MessageAttachmentService
        from app.services.message_context_service import MessageContextService
        from app.services.message_safety_service import MessageSafetyService
        
        attachment_service = MessageAttachmentService(self.db)
        context_service = MessageContextService(self.db)
        safety_service = MessageSafetyService(self.db)
        
        attachments = attachment_service.get_message_resources(message_id)
        sources = context_service.get_message_sources(message_id)
        safety_report = safety_service.get_safety_report(message_id)
        
        return {
            **message.__dict__,
            "attachments": attachments,
            "sources": sources,
            "safety_report": safety_report,
        }
    
    def delete_message(self, message_id: UUID, user_id: UUID):
        """Delete message after ownership validation."""
        message = self.get_message_with_ownership_check(message_id, user_id)
        
        self.db.delete(message)
        self.db.commit()
    
    def attach_resources_to_message(
        self,
        message_id: UUID,
        user_id: UUID,
        resource_ids: List[UUID],
        display_name: Optional[str] = None,
        attachment_type: Optional[str] = None,
    ):
        """Attach resources to message after validation."""
        if not resource_ids:
            raise ValueError("At least one resource ID is required")
        
        # Validate message and ownership
        self.get_message_with_ownership_check(message_id, user_id)

        # Validate resources (exist + ownership)
        from app.services.resource_service import ResourceService
        resource_service = ResourceService(self.db)
        resource_service.ensure_resources_owned(resource_ids, user_id)

        # Attach resources
        from app.services.message_attachment_service import MessageAttachmentService
        attachment_service = MessageAttachmentService(self.db)
        
        attachments = []
        for resource_id in resource_ids:
            attachment = attachment_service.attach_resource(
                message_id=message_id,
                resource_id=resource_id,
                display_name=display_name,
                attachment_type=attachment_type,
            )
            attachments.append(attachment)
        
        return attachments
    
    def detach_resources_from_message(
        self,
        message_id: UUID,
        user_id: UUID,
        resource_ids: List[UUID],
    ):
        """Detach resources from message after validation."""
        if not resource_ids:
            raise ValueError("At least one resource ID is required")
        
        # Validate message and ownership
        self.get_message_with_ownership_check(message_id, user_id)

        # Detach resources
        from app.services.message_attachment_service import MessageAttachmentService
        attachment_service = MessageAttachmentService(self.db)
        
        for resource_id in resource_ids:
            attachment_service.detach_resource(
                message_id=message_id,
                resource_id=resource_id,
            )
            
    def generate_ai_response(
        self,
        message_id: UUID,
        user_id: UUID,
        resource_ids: Optional[List[UUID]] = None,
    ):
        """Generate AI response for a user message using RAG."""
        message = self.get_message_with_ownership_check(message_id, user_id)
        
        # RAG parameters
        query_embedding = None
        bm25_k: int = 8 # number of documents for BM25 fallback
        final_k: int = 3 # number of chunks after dense re-rank

        if message.role != "user":
            raise ValueError("Can only generate responses for user messages")
        
        # Get user query from message
        user_query = message.content or message.transcript or ""
        if not user_query:
            raise ValueError("Cannot generate response without user query content")
        
        if not resource_ids:
            raise ValueError("No resources provided for RAG. Attach resources to generate a response.")
        
        # generate query embedding
        # query_embedding: Optional[List[float]] = None,
        from app.components.document_processing.services.embedding_service import generate_text_embedding
        query_embedding: list[float] = generate_text_embedding(user_query)
        logging.info("Generated query embedding for message %s: %s", message_id, query_embedding[:5])
        # Generate response using RAG
        from app.services.rag_service import RAGService
        rag_service = RAGService(self.db)
        
        result = rag_service.generate_response(
            session_id=message.session_id,
            user_message_id=message_id,
            user_query=user_query,
            resource_ids=resource_ids,
            query_embedding=query_embedding,
            bm25_k=bm25_k,
            final_k=final_k,
            grade_level=message.grade_level,
            user_id=user_id,
        )

        logger.info("RAG response generated for message %s", message_id)
        
        # Get the created assistant message
        assistant_message = self.db.query(Message).filter(
            Message.id == result["assistant_message_id"]
        ).first()
        
        return assistant_message
