# app/routers/messages.py

import logging
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from uuid import UUID
import uuid
from app.schemas.message import (
    MessageCreate,
    MessageDetachRequest,
    MessageDetail,
    MessageResponse,
    MessageCreateResponse,
    MessageWithAttachmentsResponse,
    MessageAttachRequest,
    MessageAttachmentResponse,
    MessageAttachmentWithResource,
    MessageContextChunkResponse,
    MessageSafetyReportResponse,
    XAIExplanationResponse,
    GenerateResponseRequest,
    ProcessingLogResponse,
)
from app.schemas.resource import ResourceProcessResponse
from app.services.message_service import MessageService
from app.services.message_attachment_service import MessageAttachmentService
from app.services.message_context_service import MessageContextService
from app.services.message_safety_service import MessageSafetyService
from app.services.session_resource_service import SessionResourceService
from app.services.chat_session_service import ChatSessionService
from app.services.resource_service import ResourceService
from app.services.rag_service import RAGService
from app.services.usage_service import UsageService
from app.core.database import get_db
from app.core.security import get_current_user
from app.shared.models.user import User
from app.shared.models.message import Message
from app.routers.websockets import manager
import asyncio
from typing import List, Optional, Dict, Any
from app.services.safety_summary_service import SafetySummaryService
from fastapi import status
from app.schemas.safety_summary import SafetySummaryResponse
from app.repositories.processing_log_repository import ProcessingLogRepository

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/sessions/{session_id}", response_model=MessageCreateResponse)
def create_user_message(
    session_id: str,
    payload: MessageCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    try:
        chat_session_service = ChatSessionService(db)

        # 🔑 If frontend sends "undefined", CREATE SESSION PROPERLY
        if session_id in ("undefined", "null", "", None):
            session = chat_session_service.create_session(
                user_id=current_user.id,
                mode="learning",      # or infer later
                channel="text",
                title="New Chat",
            )
            parsed_session_id = session.id
        else:
            parsed_session_id = UUID(session_id)

        session_for_limit = chat_session_service.get_session(parsed_session_id)
        if session_for_limit is None or session_for_limit.mode == "learning":
            UsageService(db).check_learning_request_limit(current_user.id)

        message_service = MessageService(db)

        message = message_service.create_user_message_with_validation(
            session_id=parsed_session_id,   # ✅ guaranteed to exist
            user_id=current_user.id,
            content=payload.content,
            modality=payload.modality,
            grade_level=payload.grade_level,
            audio_url=payload.audio_url,
            transcript=payload.transcript,
            audio_duration_sec=payload.audio_duration_sec,
        )
        
        # 🎯 Generate intelligent title if this is a new session with default title
        if session_id in ("undefined", "null", "", None) and payload.content:
            try:
                from app.components.document_processing.services.classifier_service import generate_session_title
                intelligent_title = generate_session_title(payload.content)
                chat_session_service.update_session_title(parsed_session_id, current_user.id, intelligent_title)
                logger.info(f"Generated title '{intelligent_title}' for new session {parsed_session_id}")
            except Exception as e:
                logger.warning(f"Failed to generate session title: {e}")
                # Continue with default "New Chat" title

        attachments = payload.attachments or []
        if attachments:
            # Validate that all resources exist and are owned by the user
            resource_ids = [att.resource_id for att in attachments]
            resource_service = ResourceService(db)
            try:
                resource_service.ensure_resources_owned(resource_ids, current_user.id)
            except ValueError as e:
                logger.warning(f"Resource validation error for user {current_user.id}: {e}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=str(e)
                )
            except PermissionError as e:
                logger.warning(f"User {current_user.id} attempted to attach unowned resources: {e}")
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=str(e)
                )
            
            attachment_service = MessageAttachmentService(db)
            session_resource_service = SessionResourceService(db)
            
            for att in attachments:
                # Attach to message
                attachment_service.attach_resource(
                    message_id=message.id,
                    resource_id=att.resource_id,
                    display_name=att.display_name,
                    attachment_type=att.attachment_type,
                )
                
                # Also attach to session
                session_resource_service.attach_resource_to_session(
                    session_id=parsed_session_id,
                    resource_id=att.resource_id,
                )

            logger.info(f"Attached {len(attachments)} resources to message {message.id} and session {parsed_session_id}")

        logger.info(f"User message created: {message.id} in session {parsed_session_id} by user {current_user.id}")

        return message

    except ValueError as e:
        logger.warning(f"Validation error creating message in session {session_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except PermissionError as e:
        logger.warning(f"User {current_user.id} attempted unauthorized message creation in session {session_id}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to create message", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to create message",
        )

@router.get("/{message_id}", response_model=MessageResponse)
def get_message(
    message_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Get a single message.
    
    Args:
        message_id: ID of the message to retrieve
        current_user: Authenticated user
        db: Database session
        
    Returns:
        MessageResponse with message details
        
    Raises:
        HTTPException 403: User doesn't own the session
        HTTPException 404: Message not found
        HTTPException 500: Database error
    """
    try:
        message_service = MessageService(db)
        message = message_service.get_message_with_ownership_check(message_id, current_user.id)
        logger.debug(f"Retrieved message {message_id} for user {current_user.id}")
        return message
        
    except ValueError as e:
        logger.warning(f"Message {message_id} not found for user {current_user.id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except PermissionError as e:
        logger.warning(f"User {current_user.id} attempted unauthorized access to message {message_id}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error retrieving message {message_id} for user {current_user.id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve message"
        )


@router.get("/{message_id}/details", response_model=MessageDetail)
def get_message_details(
    message_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Get a single message with all details (attachments, context, safety).
    
    Args:
        message_id: ID of the message
        current_user: Authenticated user
        db: Database session
        
    Returns:
        MessageDetail with full message information
        
    Raises:
        HTTPException 403: User doesn't own the session
        HTTPException 404: Message not found
        HTTPException 500: Database error
    """
    try:
        message_service = MessageService(db)
        details = message_service.get_message_details(message_id, current_user.id)
        logger.debug(f"Retrieved message details for {message_id} by user {current_user.id}")
        return details
        
    except ValueError as e:
        logger.warning(f"Message {message_id} not found for details request by user {current_user.id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except PermissionError as e:
        logger.warning(f"User {current_user.id} attempted unauthorized access to message details {message_id}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error retrieving message details {message_id} for user {current_user.id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve message details"
        )


@router.delete("/{message_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_message(
    message_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Delete a message.
    
    Args:
        message_id: ID of the message to delete
        current_user: Authenticated user
        db: Database session
        
    Raises:
        HTTPException 403: User doesn't own the session
        HTTPException 404: Message not found
        HTTPException 500: Database error
    """
    try:
        message_service = MessageService(db)
        message_service.delete_message(message_id, current_user.id)
        logger.info(f"Message {message_id} deleted by user {current_user.id}")
        
    except ValueError as e:
        logger.warning(f"Message {message_id} not found for deletion by user {current_user.id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except PermissionError as e:
        logger.warning(f"User {current_user.id} attempted unauthorized deletion of message {message_id}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting message {message_id} for user {current_user.id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete message"
        )


@router.post("/{message_id}/attachments", response_model=List[MessageAttachmentResponse])
def attach_files_to_message(
    message_id: UUID,
    payload: MessageAttachRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Attach files/resources that are relevant only to this message.
    
    Args:
        message_id: ID of the message
        payload: Resource attachment data
        current_user: Authenticated user
        db: Database session
        
    Returns:
        List of MessageAttachmentResponse objects
        
    Raises:
        HTTPException 400: Invalid resource IDs
        HTTPException 403: User doesn't own the session
        HTTPException 404: Message not found
        HTTPException 500: Database error
    """
    try:
        # Validate that all resources exist and are owned by the user
        resource_service = ResourceService(db)
        try:
            resource_service.ensure_resources_owned(payload.resource_ids, current_user.id)
        except ValueError as e:
            logger.warning(f"Resource validation error for user {current_user.id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )
        except PermissionError as e:
            logger.warning(f"User {current_user.id} attempted to attach unowned resources: {e}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=str(e)
            )
        
        message_service = MessageService(db)
        attachments = message_service.attach_resources_to_message(
            message_id=message_id,
            user_id=current_user.id,
            resource_ids=payload.resource_ids,
            display_name=payload.display_name,
            attachment_type=payload.attachment_type,
        )
        logger.info(f"Attached {len(attachments)} resources to message {message_id} by user {current_user.id}")
        return attachments
        
    except ValueError as e:
        logger.warning(f"Validation error attaching resources to message {message_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except PermissionError as e:
        logger.warning(f"User {current_user.id} attempted unauthorized attachment to message {message_id}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error attaching resources to message {message_id} for user {current_user.id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to attach resources"
        )

# remove attachments
@router.delete("/{message_id}/attachments", status_code=status.HTTP_204_NO_CONTENT)
def detach_files_from_message(
    message_id: UUID,
    payload: MessageDetachRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Detach files/resources from this message.
    
    Args:
        message_id: ID of the message
        payload: Resource detachment data
        current_user: Authenticated user
        db: Database session
    Raises:
        HTTPException 400: Invalid resource IDs
        HTTPException 403: User doesn't own the session
        HTTPException 404: Message not found
        HTTPException 500: Database error
    """
    try:
        message_service = MessageService(db)
        message_service.detach_resources_from_message(
            message_id=message_id,
            user_id=current_user.id,
            resource_ids=payload.resource_ids,
        )
        logger.info(f"Detached resources from message {message_id} by user {current_user.id}")
        
    except ValueError as e:
        logger.warning(f"Validation error detaching resources from message {message_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except PermissionError as e:
        logger.warning(f"User {current_user.id} attempted unauthorized detachment from message {message_id}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error detaching resources from message {message_id} for user {current_user.id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to detach resources"
        )


@router.post("/{message_id}/attachments/process", response_model=List[ResourceProcessResponse])
async def process_message_attachments(
    message_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    OCR, chunk, and embed all resources attached to a message.

    Args:
        message_id: ID of the message whose attachments should be processed
        current_user: Authenticated user
        db: Database session

    Returns:
        List of processing results for each attached resource

    Raises:
        HTTPException 400: No attachments or validation error
        HTTPException 403: User doesn't own the session
        HTTPException 404: Message not found
        HTTPException 500: Processing error
    """
    try:
        message_service = MessageService(db)
        message = message_service.get_message_with_ownership_check(message_id, current_user.id)

        attachment_service = MessageAttachmentService(db)
        attachments = attachment_service.get_message_resources(message_id)
        resource_ids = list({att.resource_id for att in attachments})

        # If no message attachments, fall back to session resources
        if not resource_ids:
            session_resource_service = SessionResourceService(db)
            session_resources = session_resource_service.get_session_resources(message.session_id)
            resource_ids = list({res.resource_id for res in session_resources})

        if not resource_ids:
            raise ValueError("No attachments found for this message or session")

        resource_service = ResourceService(db)
        processing_log_repo = ProcessingLogRepository(db)
        results = []
        loop = asyncio.get_running_loop()

        total_resources = len(resource_ids)

        progress_log = []

        for idx, resource_id in enumerate(resource_ids):
            # Capture loop variables for the callback closure
            current_res_id = resource_id
            current_index = idx + 1
            progress_log.clear()

            def progress_callback(
                stage: str,
                progress: float,
                details: Optional[Dict[str, Any]] = None,
                _res_id=current_res_id,
                _index=current_index,
            ):
                event = {
                    "stage": stage,
                    "progress": progress,
                    "details": details,
                }
                progress_log.append(event)

                # Persist to database
                try:
                    processing_log_repo.create(
                        resource_id=_res_id,
                        stage=stage,
                        progress=progress,
                        details=details,
                        user_id=current_user.id,
                        session_id=message.session_id,
                        message_id=message_id,
                    )
                    db.commit()
                except Exception as log_err:
                    logger.warning(
                        "Failed to persist processing log | resource=%s stage=%s: %s",
                        _res_id, stage, log_err,
                    )
                    db.rollback()

                logger.info(
                    "Progress update | user=%s resource=%s stage=%s progress=%s",
                    current_user.id,
                    _res_id,
                    stage,
                    progress,
                )

                asyncio.run_coroutine_threadsafe(
                    manager.send_personal_message(
                        {
                            "type": "processing_progress",
                            "resource_id": str(_res_id),
                            "message_id": str(message_id),
                            "stage": stage,
                            "progress": round(progress, 1),
                            "document_index": _index,
                            "total_documents": total_resources,
                            "details": details,
                        },
                        str(current_user.id),
                    ),
                    loop,
                )

            # Run synchronous processing in a thread pool to allow async WebSocket messages
            result = await asyncio.to_thread(
                resource_service.process_resource,
                resource_id=resource_id,
                user_id=current_user.id,
                progress_callback=progress_callback,
            )

            result["processing_steps"] = progress_log.copy()

            results.append(result)

        logger.info(
            "Processed %s attachments for message %s by user %s",
            len(results),
            message_id,
            current_user.id,
        )
        return results

    except ValueError as e:
        logger.warning(f"Validation error processing attachments for message {message_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except PermissionError as e:
        logger.warning(f"User {current_user.id} attempted unauthorized processing for message {message_id}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )
    except Exception as e:
        logger.error(
            f"Error processing attachments for message {message_id} for user {current_user.id}: {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process message attachments"
        )

@router.post("/{message_id}/generate", response_model=MessageResponse)
def generate_ai_response(
    message_id: UUID,
    payload: GenerateResponseRequest = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Generate assistant response using RAG.
    
    Args:
        message_id: ID of the user message to respond to
        payload: Optional generation parameters
        current_user: Authenticated user
        db: Database session
        
    Returns:
        MessageResponse with generated assistant message
        
    Raises:
        HTTPException 400: Invalid parameters or message type
        HTTPException 403: User doesn't own the session
        HTTPException 404: Message not found
        HTTPException 500: Generation or database error
    """
    try:
        message_service = MessageService(db)
        message = message_service.get_message_with_ownership_check(message_id, current_user.id)

        attachment_service = MessageAttachmentService(db)
        attachments = attachment_service.get_message_resources(message_id)
        resource_ids = [att.resource_id for att in attachments]

        if not resource_ids:
            session_resource_service = SessionResourceService(db)
            session_resources = session_resource_service.get_session_resources(message.session_id)
            resource_ids = [res.resource_id for res in session_resources]

        assistant_message = message_service.generate_ai_response(
            message_id=message_id,
            user_id=current_user.id,
            resource_ids=resource_ids,
        )

        # Attach safety summary to generated assistant message
        try:
            summary_service = SafetySummaryService(db)
            summary = summary_service.build_summary(assistant_message.id)
            if summary:  # Only create dict if summary exists
                assistant_message.safety_summary = {
                    "overall_severity": summary.get("overall_severity"),
                    "confidence_score": summary.get("confidence_score"),
                    "reliability": summary.get("reliability"),
                }
        except Exception as e:
            logger.debug(f"No safety summary for generated message {assistant_message.id}: {e}")

        logger.info(f"AI response generated for message {message_id} by user {current_user.id}")
        return assistant_message
        
    except ValueError as e:
        logger.warning(f"Validation error generating response for message {message_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except PermissionError as e:
        logger.warning(f"User {current_user.id} attempted unauthorized response generation for message {message_id}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error generating response for message {message_id} for user {current_user.id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate AI response"
        )


@router.get("/sessions/{session_id}", response_model=List[MessageResponse])
def get_message_history(
    session_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Get all messages for a session.
    
    Args:
        session_id: ID of the chat session
        current_user: Authenticated user
        db: Database session
        
    Returns:
        List of MessageResponse objects in chronological order
        
    Raises:
        HTTPException 403: User doesn't own the session
        HTTPException 404: Session not found
        HTTPException 500: Database error
    """
    try:
        session_service = ChatSessionService(db)
        session_service.get_session_with_ownership_check(session_id, current_user.id)
        
        message_service = MessageService(db)
        messages = message_service.list_session_messages_with_attachments(session_id)

        # Build response including resource_ids and safety summary per message
        summary_service = SafetySummaryService(db)
        log_repo = ProcessingLogRepository(db)
        message_ids_with_logs = log_repo.get_message_ids_with_logs([m.id for m in messages])
        response_messages = []
        for message in messages:
            resource_ids = [att.resource_id for att in getattr(message, "attachments", [])]

            safety_summary = None
            if getattr(message, "role", None) == "assistant":
                try:
                    summary = summary_service.build_summary(message.id)
                    if summary:  # Only create dict if summary exists
                        safety_summary = {
                            "overall_severity": summary.get("overall_severity"),
                            "confidence_score": summary.get("confidence_score"),
                            "reliability": summary.get("reliability"),
                        }
                except Exception as e:
                    logger.debug(f"No safety summary for message {message.id}: {e}")

            response_messages.append({
                "id": message.id,
                "session_id": message.session_id,
                "role": message.role,
                "modality": message.modality,
                "content": message.content,
                "grade_level": message.grade_level,
                "audio_url": message.audio_url,
                "transcript": message.transcript,
                "audio_duration_sec": message.audio_duration_sec,
                "created_at": message.created_at,
                "resource_ids": resource_ids,
                "parent_msg_id": message.parent_msg_id,
                "safety_summary": safety_summary,
                "has_processing_log": message.id in message_ids_with_logs,
            })
        
        logger.debug(f"Retrieved {len(messages)} messages with attachments for session {session_id} by user {current_user.id}")
        return response_messages
        
    except ValueError as e:
        logger.warning(f"Session {session_id} not found for message history by user {current_user.id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except PermissionError as e:
        logger.warning(f"User {current_user.id} attempted unauthorized access to message history for session {session_id}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error retrieving message history for session {session_id} for user {current_user.id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve message history"
        )


@router.get("/sessions/{session_id}/with-attachments", response_model=List[MessageWithAttachmentsResponse])
def get_message_history_with_attachments(
    session_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Get all messages for a session with attachments and resource details.
    Includes resource file information (filename, mime_type, size, storage_path) for each attachment.
    
    Args:
        session_id: ID of the chat session
        current_user: Authenticated user
        db: Database session
        
    Returns:
        List of MessageWithAttachmentsResponse objects with full attachment details
        
    Raises:
        HTTPException 403: User doesn't own the session
        HTTPException 404: Session not found
        HTTPException 500: Database error
    """
    try:
        session_service = ChatSessionService(db)
        session_service.get_session_with_ownership_check(session_id, current_user.id)
        
        message_service = MessageService(db)
        messages = message_service.list_session_messages_with_attachments(session_id)
        
        # Transform to response format with full resource details
        response_messages = _transform_messages_with_full_attachments(messages)
        
        logger.debug(f"Retrieved {len(messages)} messages with attachments for session {session_id} by user {current_user.id}")
        return response_messages
        
    except ValueError as e:
        logger.warning(f"Session {session_id} not found for message history by user {current_user.id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except PermissionError as e:
        logger.warning(f"User {current_user.id} attempted unauthorized access to message history for session {session_id}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error retrieving message history with attachments for session {session_id} for user {current_user.id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve message history with attachments"
        )

def _transform_messages_with_full_attachments(messages):
    """Transform message models to MessageWithAttachmentsResponse format with full resource details."""
    response_messages = []
    for message in messages:
        message_dict = {
            "id": message.id,
            "session_id": message.session_id,
            "role": message.role,
            "modality": message.modality,
            "content": message.content,
            "grade_level": message.grade_level,
            "audio_url": message.audio_url,
            "transcript": message.transcript,
            "audio_duration_sec": message.audio_duration_sec,
            "created_at": message.created_at,
            "attachments": _transform_attachments_with_resource_details(message.attachments)
        }
        response_messages.append(message_dict)
    return response_messages


def _transform_attachments_with_resource_details(attachments):
    """Transform attachment models to include full resource details for MessageAttachmentWithResource."""
    return [
        {
            "id": attachment.id,
            "message_id": attachment.message_id,
            "resource_id": attachment.resource_id,
            "display_name": attachment.display_name,
            "attachment_type": attachment.attachment_type,
            "created_at": attachment.created_at,
            "resource_filename": getattr(attachment.resource, "original_filename", None) if hasattr(attachment, "resource") else None,
            "resource_mime_type": getattr(attachment.resource, "mime_type", None) if hasattr(attachment, "resource") else None,
            "resource_size_bytes": getattr(attachment.resource, "size_bytes", None) if hasattr(attachment, "resource") else None,
            "resource_storage_path": getattr(attachment.resource, "storage_path", None) if hasattr(attachment, "resource") else None,
        }
        for attachment in attachments
    ]


@router.get("/{message_id}/sources", response_model=List[MessageContextChunkResponse])
def get_message_sources(
    message_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Get resource chunks used for this response.
    
    Args:
        message_id: ID of the assistant message
        current_user: Authenticated user
        db: Database session
        
    Returns:
        List of MessageContextChunkResponse objects
        
    Raises:
        HTTPException 403: User doesn't own the session
        HTTPException 404: Message not found
        HTTPException 500: Database error
    """
    try:
        message_service = MessageService(db)
        message_service.get_message_with_ownership_check(message_id, current_user.id)
        
        context_service = MessageContextService(db)
        sources = context_service.get_message_sources(message_id)
        
        logger.debug(f"Retrieved {len(sources)} sources for message {message_id} by user {current_user.id}")
        return sources
        
    except ValueError as e:
        logger.warning(f"Message {message_id} not found for sources request by user {current_user.id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except PermissionError as e:
        logger.warning(f"User {current_user.id} attempted unauthorized access to sources for message {message_id}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error retrieving sources for message {message_id} for user {current_user.id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve message sources"
        )


@router.get("/{message_id}/safety", response_model=MessageSafetyReportResponse)
def get_message_safety_report(
    message_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Get hallucination/safety report for an assistant message.
    
    Args:
        message_id: ID of the assistant message
        current_user: Authenticated user
        db: Database session
        
    Returns:
        MessageSafetyReportResponse with safety analysis
        
    Raises:
        HTTPException 403: User doesn't own the session
        HTTPException 404: Message or report not found
        HTTPException 500: Database error
    """
    try:
        message_service = MessageService(db)
        message_service.get_message_with_ownership_check(message_id, current_user.id)
        
        safety_service = MessageSafetyService(db)
        safety_report = safety_service.get_safety_report_with_xai(message_id)
        
        if not safety_report:
            logger.warning(f"Safety report not found for message {message_id} by user {current_user.id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Safety report not found for this message"
            )
        
        logger.debug(f"Retrieved safety report for message {message_id} by user {current_user.id}")
        return safety_report
        
    except ValueError as e:
        logger.warning(f"Message {message_id} not found for safety report by user {current_user.id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except PermissionError as e:
        logger.warning(f"User {current_user.id} attempted unauthorized access to safety report for message {message_id}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving safety report for message {message_id} for user {current_user.id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve safety report"
        )

@router.get("/{message_id}/xai", response_model= XAIExplanationResponse)
def get_message_xai_explanation(
    message_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Fetch only the XAI explanation attached to a safety report."""
    try:
        message_service = MessageService(db)
        message_service.get_message_with_ownership_check(message_id, current_user.id)

        safety_service = MessageSafetyService(db)
        report = safety_service.get_safety_report_with_xai(message_id)
        if not report or report.xai_explanation is None:
            logger.warning(f"XAI explanation not found for message {message_id} by user {current_user.id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="XAI explanation not available for this message"
            )
        return {"xai_explanation": report.xai_explanation}
    except ValueError as e:
        logger.warning(f"Message {message_id} not found for XAI request by user {current_user.id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except PermissionError as e:
        logger.warning(f"User {current_user.id} attempted unauthorized access to XAI for message {message_id}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving XAI for message {message_id} for user {current_user.id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve XAI explanation"
        )


@router.get(
    "/{message_id}/safety-summary",
    response_model=SafetySummaryResponse
)
def get_message_safety_summary(
    message_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Get a user-facing safety summary for an assistant message.

    This endpoint provides a high-level risk assessment without
    exposing raw hallucination details.
    """
    try:
        # Ownership check (reuse existing logic)
        message_service = MessageService(db)
        message_service.get_message_with_ownership_check(
            message_id, current_user.id
        )

        summary_service = SafetySummaryService(db)
        summary = summary_service.build_summary(message_id)

        return summary

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )

    except PermissionError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e),
        )

    except Exception as e:
        logger.error(
            f"Error retrieving safety summary for message {message_id}: {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve safety summary",
        )


@router.get(
    "/{message_id}/processing-logs",
    response_model=List[ProcessingLogResponse],
)
def get_message_processing_logs(
    message_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Get all processing logs for a specific message.
    """
    try:
        message_service = MessageService(db)
        message_service.get_message_with_ownership_check(message_id, current_user.id)

        repo = ProcessingLogRepository(db)
        logs = repo.get_logs_for_message(message_id)
        return logs

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )

    except PermissionError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e),
        )

    except Exception as e:
        logger.error(
            f"Error retrieving processing logs for message {message_id}: {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve processing logs",
        )
