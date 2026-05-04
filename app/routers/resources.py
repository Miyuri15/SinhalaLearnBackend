# app/routers/resources.py

import logging
from typing import List, Optional
from urllib.parse import quote
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, status, Query, Request
from fastapi.responses import FileResponse
import os
from sqlalchemy.orm import Session
from uuid import UUID
from app.schemas.resource import (
    ResourceFileResponse,
    ResourceFileUpdate,
    ResourceUploadResponse,
    ResourceBulkUploadResponse,
    ResourceProcessResponse,
    ResourceBatchProcessRequest,
    ResourceExtractedTextResponse
)
from app.schemas.resource_chunk import ResourceChunkResponse
from app.services.resource_service import ResourceService
from app.services.resource_chunk_service import ResourceChunkService
from app.services.evaluation.user_context_service import UserContextService
from app.core.database import get_db
from app.core.security import get_current_user
from app.shared.models.user import User
from app.utils.resource_text import split_extracted_text_pages

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/upload", response_model=ResourceBulkUploadResponse)
async def upload_resource(
    files: List[UploadFile] = File(...),
    resource_type: Optional[str] = Query(None, enum=["syllabus", "question_paper", "rubric"]),
    chat_session_id: Optional[UUID] = Query(None, description="Optional chat session ID to attach the resource to"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Upload one or more resource files (PDF, image, audio).
    
    Args:
        files: List of uploaded files
        resource_type: Optional type of resource to set as active context (syllabus, question_paper)
        chat_session_id: Optional chat session ID to attach the resource to
        current_user: Authenticated user
        db: Database session
        
    Returns:
        ResourceBulkUploadResponse with list of upload details
        
    Raises:
        HTTPException 400: Invalid file type or empty file
        HTTPException 500: File storage or database error
    """
    user_id = current_user.id  # Load user_id early to avoid lazy loading issues
    upload_results = []
    
    for file in files:
        try:
            # Read file content
            content = await file.read()
            
            # Upload via service
            resource_service = ResourceService(db)
            resource = resource_service.upload_resource_from_file(
                user_id=user_id,
                filename=file.filename,
                content_type=file.content_type,
                content=content,
                resource_type=resource_type,
            )
            
            # Update user context if resource_type is provided (Legacy/Global Context)
            if resource_type:
                context_service = UserContextService(db)
                if resource_type == "syllabus":
                    context_service.update_syllabus(user_id, resource.id)
                elif resource_type == "question_paper":
                    context_service.update_question_paper(user_id, resource.id)
                elif resource_type == "rubric":
                    context_service.update_rubric(user_id, resource.id)
            
            # Attach to Chat Session if provided
            if chat_session_id and resource_type:
                from app.services.chat_session_service import ChatSessionService
                chat_service = ChatSessionService(db)
                chat_service.attach_resource(
                    session_id=chat_session_id,
                    user_id=user_id,
                    resource_id=resource.id,
                    role=resource_type
                )
            
            upload_results.append(ResourceUploadResponse(
                resource_id=resource.id,
                filename=file.filename,
                size_bytes=len(content),
                mime_type=file.content_type,
            ))
            
        except ValueError as e:
            logger.warning(f"Validation error uploading resource {file.filename}: {e}")
            # For multiple files, we might want to continue with others or fail all
            # For now, fail all if any fails
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Validation error for {file.filename}: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Error uploading resource {file.filename} for user {user_id}: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to upload {file.filename}"
            )
    
    logger.info(f"Uploaded {len(upload_results)} resources for user {user_id}")
    return ResourceBulkUploadResponse(uploads=upload_results)

from app.utils.file_validation import validate_files, MAX_FILES
@router.post("/upload/batch", response_model=List[ResourceUploadResponse])
async def upload_resources(
    files: List[UploadFile] = Depends(validate_files),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Upload multiple resource files (PDF, image, audio).

    Args:
        files: Uploaded files
        current_user: Authenticated user
        db: Database session
        
    Returns:
        List of ResourceUploadResponse with upload details
        
    Raises:
        HTTPException 400: Invalid file(s) or empty payload
        HTTPException 500: File storage or database error
    """
    created_paths: List[str] = []
    try:
        if not files:
            raise ValueError("No files uploaded")

        resource_service = ResourceService(db)
        responses: List[ResourceUploadResponse] = []

        # Insert all records, then commit once at the end
        for file in files:
            content = await file.read()
            resource = resource_service.upload_resource_from_file(
                user_id=current_user.id,
                filename=file.filename,
                content_type=file.content_type,
                content=content,
                commit=False,  # defer commit until all succeed
            )

            # Track file path for cleanup if batch fails later
            if getattr(resource, "storage_path", None):
                created_paths.append(resource.storage_path)

            responses.append(
                ResourceUploadResponse(
                    resource_id=resource.id,
                    filename=file.filename,
                    size_bytes=len(content),
                    mime_type=file.content_type,
                )
            )

        # Commit everything atomically
        db.commit()

        logger.info(f"{len(responses)} resources uploaded by user {current_user.id}")
        return responses
        
    except ValueError as e:
        # Roll back any pending DB work, then cleanup files
        try:
            db.rollback()
        except Exception:
            pass
        for p in created_paths:
            try:
                if p and os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass
        logger.warning(f"Validation error uploading resources: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        # Roll back any pending DB work, then cleanup files
        try:
            db.rollback()
        except Exception:
            pass
        for p in created_paths:
            try:
                if p and os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass
        logger.error(f"Error uploading resources for user {current_user.id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to upload resources"
        )

@router.post("/upload-only/batch", response_model=List[ResourceUploadResponse])
async def upload_resources(
    files: List[UploadFile] = Depends(validate_files),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Upload multiple resource files (PDF, image, audio).

    Args:
        files: Uploaded files
        current_user: Authenticated user
        db: Database session
        
    Returns:
        List of ResourceUploadResponse with upload details
        
    Raises:
        HTTPException 400: Invalid file(s) or empty payload
        HTTPException 500: File storage or database error
    """
    created_paths: List[str] = []
    try:
        if not files:
            raise ValueError("No files uploaded")

        resource_service = ResourceService(db)
        responses: List[ResourceUploadResponse] = []

        # Insert all records, then commit once at the end
        for file in files:
            content = await file.read()
            resource = resource_service.only_upload_resource_from_file(
                user_id=current_user.id,
                filename=file.filename,
                content_type=file.content_type,
                content=content,
                commit=False,  # defer commit until all succeed
            )

            # Track file path for cleanup if batch fails later
            if getattr(resource, "storage_path", None):
                created_paths.append(resource.storage_path)

            responses.append(
                ResourceUploadResponse(
                    resource_id=resource.id,
                    filename=file.filename,
                    size_bytes=len(content),
                    mime_type=file.content_type,
                )
            )

        # Commit everything atomically
        db.commit()

        logger.info(f"{len(responses)} resources uploaded by user {current_user.id}")
        return responses
        
    except ValueError as e:
        # Roll back any pending DB work, then cleanup files
        try:
            db.rollback()
        except Exception:
            pass
        for p in created_paths:
            try:
                if p and os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass
        logger.warning(f"Validation error uploading resources: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        # Roll back any pending DB work, then cleanup files
        try:
            db.rollback()
        except Exception:
            pass
        for p in created_paths:
            try:
                if p and os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass
        logger.error(f"Error uploading resources for user {current_user.id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to upload resources"
        )


@router.get("/", response_model=List[ResourceFileResponse])
def list_resources(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    List all resources for the current user.
    
    Args:
        current_user: Authenticated user
        db: Database session
        
    Returns:
        List of ResourceFileResponse objects
        
    Raises:
        HTTPException 500: Database error
    """
    try:
        resource_service = ResourceService(db)
        resources = resource_service.list_user_resources(current_user.id)
        
        logger.debug(f"Retrieved {len(resources)} resources for user {current_user.id}")
        return resources
        
    except Exception as e:
        logger.error(f"Error listing resources for user {current_user.id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve resources"
        )


@router.get("/{resource_id}", response_model=ResourceFileResponse)
def get_resource(
    resource_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Get resource metadata.
    
    Args:
        resource_id: ID of the resource
        current_user: Authenticated user
        db: Database session
        
    Returns:
        ResourceFileResponse with resource details
        
    Raises:
        HTTPException 403: User doesn't own the resource
        HTTPException 404: Resource not found
        HTTPException 500: Database error
    """
    try:
        resource_service = ResourceService(db)
        resource = resource_service.get_resource_with_ownership_check(resource_id, current_user.id)
        logger.debug(f"Retrieved resource {resource_id} for user {current_user.id}")
        return resource
        
    except ValueError as e:
        logger.warning(f"Resource {resource_id} not found for user {current_user.id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except PermissionError as e:
        logger.warning(f"User {current_user.id} attempted unauthorized access to resource {resource_id}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error retrieving resource {resource_id} for user {current_user.id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve resource"
        )


@router.get("/{resource_id}/chunks", response_model=List[ResourceChunkResponse])
def get_resource_chunks(
    resource_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    List all chunks for a resource (admin/debug use).
    
    Args:
        resource_id: ID of the resource
        current_user: Authenticated user
        db: Database session
        
    Returns:
        List of ResourceChunkResponse objects
        
    Raises:
        HTTPException 403: User doesn't own the resource
        HTTPException 404: Resource not found
        HTTPException 500: Database error
    """
    try:
        # Verify ownership
        resource_service = ResourceService(db)
        resource_service.get_resource_with_ownership_check(resource_id, current_user.id)
        
        # Get chunks
        chunk_service = ResourceChunkService(db)
        chunks = chunk_service.get_chunks_for_resource(resource_id)
        logger.debug(f"Retrieved {len(chunks)} chunks for resource {resource_id}")
        return chunks
        
    except ValueError as e:
        logger.warning(f"Resource {resource_id} not found for user {current_user.id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except PermissionError as e:
        logger.warning(f"User {current_user.id} attempted unauthorized access to resource {resource_id}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error retrieving chunks for resource {resource_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve chunks"
        )


@router.get("/{resource_id}/extracted-text", response_model=ResourceExtractedTextResponse)
def get_resource_extracted_text(
    resource_id: UUID,
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=10),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Get full extracted OCR text for a resource.
    Protected; owner-only.
    """
    try:
        resource_service = ResourceService(db)
        resource = resource_service.get_resource_with_ownership_check(
            resource_id,
            current_user.id,
        )

        chunk_service = ResourceChunkService(db)
        chunks = chunk_service.get_chunks_for_resource(resource_id)

        extracted_text = (getattr(resource, "extracted_text", None) or "").strip()
        if not extracted_text:
            extracted_text = "\n\n".join(
                chunk.content
                for chunk in chunks
                if getattr(chunk, "content", None)
            ).strip()

        pages = split_extracted_text_pages(extracted_text)
        start_index = (page - 1) * page_size
        end_index = start_index + page_size
        page_slice = pages[start_index:end_index]
        paginated_text = "\n\n".join(page_slice).strip()

        return ResourceExtractedTextResponse(
            resource_id=resource.id,
            status="completed" if paginated_text else "empty",
            extracted_text=paginated_text,
            chunks_count=len(chunks),
            page=page,
            page_size=page_size,
            total_pages=len(pages),
            returned_pages=len(page_slice),
            has_next=end_index < len(pages),
            has_previous=page > 1 and bool(pages),
            language=getattr(resource, "language", None),
        )

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except Exception:
        logger.error(
            f"Error retrieving extracted text for resource {resource_id}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve extracted text",
        )


@router.put("/{resource_id}", response_model=ResourceFileResponse)
def update_resource(
    resource_id: UUID,
    payload: ResourceFileUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Update resource metadata.
    
    Args:
        resource_id: ID of the resource to update
        payload: Resource update data
        current_user: Authenticated user
        db: Database session
        
    Returns:
        Updated ResourceFileResponse
        
    Raises:
        HTTPException 403: User doesn't own the resource
        HTTPException 404: Resource not found
        HTTPException 500: Database error
    """
    try:
        resource_service = ResourceService(db)
        resource = resource_service.update_resource(
            resource_id=resource_id,
            user_id=current_user.id,
            original_filename=payload.original_filename,
            language=payload.language,
        )
        logger.info(f"Resource {resource_id} updated by user {current_user.id}")
        return resource
        
    except ValueError as e:
        logger.warning(f"Resource {resource_id} not found for update by user {current_user.id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except PermissionError as e:
        logger.warning(f"User {current_user.id} attempted unauthorized update to resource {resource_id}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )
    except Exception as e:
        db.rollback()
        logger.error(f"Error updating resource {resource_id} for user {current_user.id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update resource"
        )


@router.delete("/{resource_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_resource(
    resource_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Delete a resource file.
    
    Args:
        resource_id: ID of the resource to delete
        current_user: Authenticated user
        db: Database session
        
    Raises:
        HTTPException 403: User doesn't own the resource
        HTTPException 404: Resource not found
        HTTPException 500: Database or file system error
    """
    try:
        resource_service = ResourceService(db)
        resource_service.delete_resource(resource_id, current_user.id)
        logger.info(f"Resource {resource_id} deleted by user {current_user.id}")
        
    except ValueError as e:
        logger.warning(f"Resource {resource_id} not found for deletion by user {current_user.id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except PermissionError as e:
        logger.warning(f"User {current_user.id} attempted unauthorized deletion of resource {resource_id}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting resource {resource_id} for user {current_user.id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete resource"
        )


@router.post("/{resource_id}/process", response_model=ResourceProcessResponse)
def process_resource(
    resource_id: UUID,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    OCR, chunk, and embed a resource.
    
    Args:
        resource_id: ID of the resource to process
        current_user: Authenticated user
        db: Database session
        
    Returns:
        ResourceProcessResponse with processing results
        
    Raises:
        HTTPException 400: Resource already processed or invalid type
        HTTPException 403: User doesn't own the resource
        HTTPException 404: Resource not found
        HTTPException 500: Processing error
    """
    try:
        resource_service = ResourceService(db)
        result = resource_service.process_resource(
            resource_id,
            current_user.id,
            text_type_classifier=getattr(request.app.state, "text_type_classifier", None),
            handwritten_ocr_predictor=getattr(request.app.state, "handwritten_ocr_predictor", None),
            text_type_conf_threshold=getattr(request.app.state, "text_type_conf_threshold", 0.7),
        )
        logger.info(f"Resource {resource_id} processing initiated by user {current_user.id}")
        return result
        
    except ValueError as e:
        logger.warning(f"Validation error processing resource {resource_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except PermissionError as e:
        logger.warning(f"User {current_user.id} attempted unauthorized processing of resource {resource_id}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error processing resource {resource_id} for user {current_user.id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process resource"
        )


@router.post("/process/batch", response_model=List[ResourceProcessResponse])
def process_resources_batch(
    payload: ResourceBatchProcessRequest,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    OCR, chunk, and embed multiple resources.

    Args:
        payload: List of resource IDs to process
        current_user: Authenticated user
        db: Database session

    Returns:
        List of ResourceProcessResponse with processing results for each resource

    Raises:
        HTTPException 400: Empty list or validation error
        HTTPException 403: User doesn't own one or more resources
        HTTPException 404: One or more resources not found
        HTTPException 500: Processing error
    """
    try:
        if not payload.resource_ids:
            raise ValueError("No resource IDs provided")

        if len(payload.resource_ids) > MAX_FILES:
            raise ValueError(f"Too many resources to process. Maximum allowed is {MAX_FILES}.")

        resource_service = ResourceService(db)
        
        # Validate ownership of all resources upfront
        resource_service.ensure_resources_owned(payload.resource_ids, current_user.id)

        results = []
        for resource_id in payload.resource_ids:
            result = resource_service.process_resource(
                resource_id,
                current_user.id,
                text_type_classifier=getattr(request.app.state, "text_type_classifier", None),
                handwritten_ocr_predictor=getattr(request.app.state, "handwritten_ocr_predictor", None),
                text_type_conf_threshold=getattr(request.app.state, "text_type_conf_threshold", 0.7),
            )
            results.append(result)

        logger.info(
            f"{len(results)} resources processed by user {current_user.id}"
        )
        return results

    except ValueError as e:
        logger.warning(f"Validation error processing resources: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except PermissionError as e:
        logger.warning(f"User {current_user.id} attempted unauthorized batch processing: {e}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error processing resources batch for user {current_user.id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process resources"
        )


@router.get("/{resource_id}/download")
def download_resource(
    resource_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Download the resource file (protected; owner-only).

    Sets Content-Disposition to attachment with the original filename.
    """
    try:
        resource_service = ResourceService(db)
        resource = resource_service.get_resource_with_ownership_check(resource_id, current_user.id)

        if not resource.storage_path or not os.path.exists(resource.storage_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="The requested file no longer exists"
            )

        filename = resource.original_filename or os.path.basename(resource.storage_path)
        headers = {"Content-Disposition": f'attachment; filename="{filename}"'}

        return FileResponse(
            path=resource.storage_path,
            media_type=resource.mime_type or "application/octet-stream",
            headers=headers,
        )

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to download resource")


@router.get("/{resource_id}/view")
def view_resource(
    resource_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    View the resource file inline (protected; owner-only).

    Sets Content-Disposition to inline with the original filename to
    allow browser rendering for supported MIME types (e.g., PDFs, images, audio).
    """
    try:
        resource_service = ResourceService(db)
        resource = resource_service.get_resource_with_ownership_check(resource_id, current_user.id)

        if not resource.storage_path or not os.path.exists(resource.storage_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="The requested file no longer exists"
            )

        filename = resource.original_filename or os.path.basename(resource.storage_path)
        encoded_filename = quote(filename)
        headers = {
            "Content-Disposition": f"inline; filename*=UTF-8''{encoded_filename}"
        }

        return FileResponse(
            path=resource.storage_path,
            media_type=resource.mime_type or "application/octet-stream",
            headers=headers,
        )

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to view resource")
