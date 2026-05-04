# app/services/resource_service.py

import os
from pathlib import Path
from typing import Optional, List, Iterable, Set, Callable, Dict, Any
from uuid import UUID
from sqlalchemy.orm import Session

from app.repositories.resource_repository import ResourceRepository
from app.shared.models.resource_file import ResourceFile
from app.shared.models.resource_chunks import ResourceChunk
from app.shared.models.message_relations import MessageContextChunk

# Configure upload directory
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


class ResourceService:
    """Business logic for resources (files)."""

    def __init__(self, db: Session):
        self.db = db
        self.repository = ResourceRepository(db)

    def validate_file_upload(self, filename: Optional[str], content_type: Optional[str], content: bytes):
        """Validate file upload requirements."""
        if not filename:
            raise ValueError("No filename provided")
        
        if not content_type:
            raise ValueError("No content type provided")
        
        if not content:
            raise ValueError("Empty file uploaded")

    def save_file_to_disk(self, user_id: UUID, filename: str, content: bytes) -> Path:
        """Save uploaded file to disk with safe naming."""
        safe_filename = f"{user_id}_{filename}"
        file_path = UPLOAD_DIR / safe_filename
        
        with open(file_path, "wb") as f:
            f.write(content)
        
        return file_path

    def upload_resource_from_file(
        self,
        user_id: UUID,
        filename: str,
        content_type: str,
        content: bytes,
        *,
        commit: bool = True,
        resource_type: Optional[str] = None,
    ):
        """Handle complete file upload process with validation."""
        # Validate
        self.validate_file_upload(filename, content_type, content)
        
        # Save to disk
        try:
            file_path = self.save_file_to_disk(user_id, filename, content)
        except Exception as e:
            raise ValueError(f"Failed to save file: {e}")
        
        # Create database record
        try:
            resource = self.repository.upload_resource(
                user_id=user_id,
                original_filename=filename,
                storage_path=str(file_path),
                mime_type=content_type,
                size_bytes=len(content),
                source_type="user_upload",
                commit=commit,
            )
            self.process_resource(resource.id, user_id, resource_type=resource_type)
            return resource
        except Exception as e:
            # Cleanup file if database save failed
            if file_path.exists():
                try:
                    file_path.unlink()
                except Exception:
                    pass
            raise

    def only_upload_resource_from_file(
        self,
        user_id: UUID,
        filename: str,
        content_type: str,
        content: bytes,
        *,
        commit: bool = True,
    ):
        """Handle complete file upload process with validation."""
        # Validate
        self.validate_file_upload(filename, content_type, content)
        
        # Save to disk
        try:
            file_path = self.save_file_to_disk(user_id, filename, content)
        except Exception as e:
            raise ValueError(f"Failed to save file: {e}")
        
        # Create database record
        try:
            resource = self.repository.upload_resource(
                user_id=user_id,
                original_filename=filename,
                storage_path=str(file_path),
                mime_type=content_type,
                size_bytes=len(content),
                source_type="user_upload",
                commit=commit,
            )
            return resource
        except Exception as e:
            # Cleanup file if database save failed
            if file_path.exists():
                try:
                    file_path.unlink()
                except Exception:
                    pass
            raise

    def upload_resource(
        self,
        user_id: UUID,
        original_filename: Optional[str],
        storage_path: Optional[str],
        mime_type: Optional[str],
        size_bytes: Optional[int],
        source_type: Optional[str] = None,
        language: Optional[str] = None,
    ):
        return self.repository.upload_resource(
            user_id=user_id,
            original_filename=original_filename,
            storage_path=storage_path,
            mime_type=mime_type,
            size_bytes=size_bytes,
            source_type=source_type,
            language=language,
        )

    def get_resource(self, resource_id: UUID):
        return self.repository.get_resource(resource_id)

    def list_user_resources(self, user_id: UUID) -> List:
        return self.repository.list_user_resources(user_id)

    def ensure_resources_owned(self, resource_ids: Iterable[UUID], user_id: UUID):
        """Validate that all provided resources exist and belong to the user.

        Raises:
            ValueError: if any resource id does not exist
            PermissionError: if any resource is not owned by the user
        """
        unique_ids: Set[UUID] = set(resource_ids)
        if not unique_ids:
            return

        rows = (
            self.db.query(ResourceFile.id, ResourceFile.user_id)
            .filter(ResourceFile.id.in_(unique_ids))
            .all()
        )

        found_ids = {row.id for row in rows}
        missing = unique_ids - found_ids
        if missing:
            raise ValueError("One or more resources were not found")

        not_owned = {row.id for row in rows if row.user_id != user_id}
        if not_owned:
            raise PermissionError("One or more resources are not owned by the user")
    
    def get_resource_with_ownership_check(self, resource_id: UUID, user_id: UUID):
        """Get resource and verify ownership."""
        resource = self.repository.get_resource(resource_id)
        if not resource:
            raise ValueError("Resource not found")
        
        if resource.user_id != user_id:
            raise PermissionError("You don't have permission to access this resource")
        
        return resource
    
    def update_resource(
        self,
        resource_id: UUID,
        user_id: UUID,
        original_filename: Optional[str] = None,
        language: Optional[str] = None,
    ):
        """Update resource after ownership validation."""
        resource = self.get_resource_with_ownership_check(resource_id, user_id)
        
        # Update fields
        if original_filename is not None:
            resource.original_filename = original_filename
        if language is not None:
            resource.language = language
        
        self.db.commit()
        self.db.refresh(resource)
        return resource
    
    def delete_resource(self, resource_id: UUID, user_id: UUID, *, commit: bool = True):
        """Delete resource and associated file after ownership validation."""
        resource = self.get_resource_with_ownership_check(resource_id, user_id)

        chunk_ids = (
            self.db.query(ResourceChunk.id)
            .filter(ResourceChunk.resource_id == resource_id)
            .all()
        )
        chunk_ids = [row.id for row in chunk_ids]

        if chunk_ids:
            (
                self.db.query(MessageContextChunk)
                .filter(MessageContextChunk.chunk_id.in_(chunk_ids))
                .delete(synchronize_session=False)
            )

        (
            self.db.query(ResourceChunk)
            .filter(ResourceChunk.resource_id == resource_id)
            .delete(synchronize_session=False)
        )
        
        # Delete physical file
        if resource.storage_path and os.path.exists(resource.storage_path):
            try:
                os.remove(resource.storage_path)
            except Exception:
                # Log but don't fail the operation
                pass
        
        # Delete database record
        self.db.delete(resource)
        if commit:
            self.db.commit()
        else:
            self.db.flush()
    
    def process_resource(
        self, 
        resource_id: UUID, 
        user_id: UUID, 
        resource_type: Optional[str] = None,
        progress_callback: Optional[Callable[[str, float, Optional[Dict[str, Any]]], None]] = None,
        text_type_classifier: Optional[Any] = None,
        handwritten_ocr_predictor: Optional[Callable[[Any], str]] = None,
        text_type_conf_threshold: float = 0.7,
    ):
        """Process resource (OCR, chunk, embed) after validation."""
        resource = self.get_resource_with_ownership_check(resource_id, user_id)
        
        # Delegate to document processor service
        from app.components.document_processing.services.resource_processor_service import ResourceProcessorService
        
        processor = ResourceProcessorService(
            self.db,
            text_type_classifier=text_type_classifier,
            handwritten_ocr_predictor=handwritten_ocr_predictor,
            text_type_conf_threshold=text_type_conf_threshold,
        )
        result = processor.process_resource(resource, resource_type=resource_type, progress_callback=progress_callback)
        
        return result

    def search_documents(
        self, 
        resource_ids: List[UUID], 
        query_embedding: List[float], 
        top_k: int = 5
    ):
        """
        Search documents by semantic similarity using document embeddings.
        This is the first stage in two-stage retrieval.
        
        Args:
            resource_ids: List of resource IDs to search
            query_embedding: Query embedding vector
            top_k: Number of top documents to return
            
        Returns:
            List of most relevant documents with similarity scores
        """
        return self.repository.vector_search_documents(
            resource_ids=resource_ids,
            query_embedding=query_embedding,
            top_k=top_k
        )

    def list_resources_by_ids(self, resource_ids: List[UUID]) -> List:
        """List resources by their IDs."""
        return self.repository.list_resources_by_ids(resource_ids)

    def update_resource_extracted_text(
        self,
        resource_id: UUID,
        extracted_text: Optional[str],
        *,
        commit: bool = True,
    ):
        return self.repository.update_resource_extracted_text(
            resource_id=resource_id,
            extracted_text=extracted_text,
            commit=commit,
        )
