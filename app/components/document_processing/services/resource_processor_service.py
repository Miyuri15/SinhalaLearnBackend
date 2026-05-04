"""Service to process stored resources independently - OCR, chunking, and embedding."""

import os
import logging
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
from sqlalchemy.orm import Session
import cv2

from app.shared.models.resource_file import ResourceFile
from app.shared.models.resource_chunks import ResourceChunk

logger = logging.getLogger(__name__)


class ResourceProcessorService:
    """Process stored resources through OCR, chunking, and embedding pipeline."""

    def __init__(
        self,
        db: Session,
        text_type_classifier: Optional[Any] = None,
        handwritten_ocr_predictor: Optional[Callable[[Any], str]] = None,
        text_type_conf_threshold: float = 0.7,
    ):
        self.db = db
        self.text_type_classifier = text_type_classifier
        self.handwritten_ocr_predictor = handwritten_ocr_predictor
        self.text_type_conf_threshold = text_type_conf_threshold

    def _resolve_ml_model_predict(self) -> Optional[Callable[[Any], tuple[str, float]]]:
        """Normalize classifier instance into a callable(image) -> (label, confidence)."""
        if self.text_type_classifier is None:
            return None

        if callable(self.text_type_classifier):
            return self.text_type_classifier

        predict = getattr(self.text_type_classifier, "predict", None)
        if callable(predict):
            return predict

        logger.warning("Unsupported text type classifier; expected callable or object with .predict")
        return None

    def _validate_resource_file(self, resource: ResourceFile):
        """Ensure resource file exists and is accessible."""
        if not resource.storage_path:
            raise ValueError("Resource has no storage path")
        
        if not os.path.exists(resource.storage_path):
            raise ValueError(f"Resource file not found: {resource.storage_path}")
        
        file_ext = Path(resource.storage_path).suffix.lower()
        if file_ext not in ['.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.webp']:
            raise ValueError(f"Unsupported file type: {file_ext}")

    def _extract_text(
        self,
        file_path: str,
        resource_type: Optional[str] = None,
        progress_callback: Optional[Callable[[str, float, Optional[Dict[str, Any]]], None]] = None
    ) -> tuple[str, int, str]:
        print("🔥 ENTERED _extract_text")
        """
        Extract text from PDF or image file.

        Returns:
            (extracted_text, page_count, detected_language)
        """

        from pathlib import Path

        from app.components.document_processing.utils.file_operations import convert_file_to_images
        from app.components.document_processing.services.text_extraction import (
            extract_text_from_pdf,
            process_ocr_for_images_with_tables,
            classify_text_type,
            detect_language_from_text,
        )
        from app.components.document_processing.utils.text_cleaner import basic_clean
        
        print("🔥 Imports done")

        file_ext = Path(file_path).suffix.lower()

        try:

            # Stage 1
            if progress_callback:
                progress_callback("Preparing Document", 4.0, None)

            if file_ext == ".pdf":
                
                print(f"convert_file_to_images: {convert_file_to_images}")
                print(f"extract_text_from_pdf: {extract_text_from_pdf}")
                print(f"process_ocr_for_images_with_tables: {process_ocr_for_images_with_tables}")
                print(f"classify_text_type: {classify_text_type}")
                print(f"detect_language_from_text: {detect_language_from_text}")
                print(f"basic_clean: {basic_clean}")

                result = self._process_pdf(
                    file_path,
                    convert_file_to_images,
                    extract_text_from_pdf,
                    process_ocr_for_images_with_tables,
                    classify_text_type,
                    detect_language_from_text,
                    basic_clean,
                    resource_type=resource_type,
                    progress_callback=progress_callback
                )

                return result

            else:

                # Stage 2
                if progress_callback:
                    progress_callback("Processing Image", 30.0, None)

                result = self._process_image(
                    file_path,
                    convert_file_to_images,
                    process_ocr_for_images_with_tables,
                    classify_text_type,
                    detect_language_from_text,
                    basic_clean,
                    resource_type=resource_type,
                    progress_callback=progress_callback
                )

                # Stage 3
                if progress_callback:
                    progress_callback("Image Text Extraction Completed", 45.0, None)

                return result

        except Exception as e:
            logger.error(f"Failed to extract text from {file_path}: {e}")
            raise ValueError(f"Text extraction failed: {e}")   
     
    def _process_pdf(
        self,
        file_path,
        convert_file_to_images,
        extract_text_from_pdf,
        process_ocr_for_images_with_tables,
        classify_text_type,
        detect_language_from_text,
        basic_clean,
        resource_type: Optional[str] = None,
        progress_callback: Optional[Callable[[str, float, Optional[Dict[str, Any]]], None]] = None
    ):
        print("🔥 ENTERED _process_pdf")

        # Stage: Detect language from PDF metadata
        if progress_callback:
            progress_callback("Analyzing Document Language", 6.0, None)

        lang_hint = self._sniff_pdf_language(file_path, detect_language_from_text)
            
        # Try direct extraction for English PDFs
        if lang_hint == "english":

            if progress_callback:
                progress_callback("Language Detected", 25.0, {"language": lang_hint})
                progress_callback("Extracting Text", 30.0, None)

            text, pages = self._try_direct_pdf_extraction(
                file_path,
                extract_text_from_pdf,
                basic_clean
            )

            if text:
                if progress_callback:
                    progress_callback("Cleaning Extracted Text", 45.0, None)

                return text, pages, "english"

        # OCR fallback
        if progress_callback:
            progress_callback("Converting PDF to Images", 8.0, None)
            
        print("➡️ Calling convert_file_to_images")
        images = convert_file_to_images(file_path, "pdf")
        ml_model_predict = self._resolve_ml_model_predict()

        # Text classification
        if images:
            if progress_callback:
                progress_callback("Classifying Text Type", 10.0, None)

            detected_text_type, detected_confidence = self._classify_first_image(
                images[0],
                classify_text_type,
                ml_model_predict=ml_model_predict,
            )

            if detected_text_type == "unknown" or detected_confidence < self.text_type_conf_threshold:
                logger.info(
                    "First-page text type low-confidence/unknown: %s (confidence=%.3f)",
                    detected_text_type,
                    detected_confidence,
                )

            if progress_callback:
                progress_callback(
                    "Text Type Detected",
                    11.0,
                    {
                        "text_type": detected_text_type,
                        "text_type_confidence": detected_confidence,
                    },
                )

        # Layout analysis decision
        if resource_type:
            force_layout = self.is_need_to_analyze_layout(resource_type)
        else:
            force_layout = True

        if progress_callback:
            progress_callback("Running OCR Extraction", 12.0, None)

        extracted_text, page_count = process_ocr_for_images_with_tables(
            images,
            force_layout_analysis=force_layout,
            progress_callback=progress_callback,
            ml_model_predict=ml_model_predict,
            ml_conf_threshold=self.text_type_conf_threshold,
            trocr_predict=self.handwritten_ocr_predictor,
            region_conf_threshold=self.text_type_conf_threshold,
        )

        if progress_callback:
            preview = extracted_text[:300].strip() if extracted_text else ""
            progress_callback("Completed OCR Extraction", 40.0, {"preview": preview})

        if progress_callback:
            progress_callback("Cleaning Extracted Text", 41.0, None)

        cleaned_text = basic_clean(extracted_text)

        if progress_callback:
            progress_callback("Detecting Language from Text", 43.0, None)

        inferred_lang = detect_language_from_text(cleaned_text)

        final_lang = lang_hint if lang_hint != "unknown" else inferred_lang

        if progress_callback:
            progress_callback("Language Detection Completed", 45.0, {"detected_language": final_lang})

        logger.info(
            f"Final detected language for PDF: {final_lang} "
            f"(inferred from text: {inferred_lang})"
        )

        return cleaned_text, page_count, final_lang

    def _process_image(
        self,
        file_path,
        convert_file_to_images,
        process_ocr_for_images_with_tables,
        classify_text_type,
        detect_language_from_text,
        basic_clean,
        resource_type: Optional[str] = None,
        progress_callback: Optional[Callable[[str, float, Optional[Dict[str, Any]]], None]] = None
    ):
        ml_model_predict = self._resolve_ml_model_predict()

        text_type, text_type_confidence = classify_text_type(
            file_path,
            ml_model_predict=ml_model_predict,
            ml_conf_threshold=self.text_type_conf_threshold,
            return_confidence=True,
        )
        logger.info(
            "Detected text type for image: %s (confidence=%.3f)",
            text_type,
            text_type_confidence,
        )

        if text_type == "unknown" or text_type_confidence < self.text_type_conf_threshold:
            logger.info(
                "Image text type is unknown/low confidence; OCR will use fallback behavior (label=%s, confidence=%.3f)",
                text_type,
                text_type_confidence,
            )

        if progress_callback:
                progress_callback(
                    "Text Type Detected",
                    37.0,
                    {
                        "text_type": text_type,
                        "text_type_confidence": text_type_confidence,
                    },
                )

        images = convert_file_to_images(file_path, file_path.split('.')[-1])

        if resource_type:
            force_layout = self.is_need_to_analyze_layout(resource_type)
        else:
            force_layout = True

        if progress_callback:
            progress_callback("Layout Analysis", 35.0, None)
            progress_callback("Running OCR Extraction", 40.0, None)

        extracted_text, page_count = process_ocr_for_images_with_tables(
            images,
            force_layout_analysis=force_layout,
            progress_callback=progress_callback,
            ml_model_predict=ml_model_predict,
            ml_conf_threshold=self.text_type_conf_threshold,
            trocr_predict=self.handwritten_ocr_predictor,
            region_conf_threshold=self.text_type_conf_threshold,
        )

        cleaned_text = basic_clean(extracted_text)
        
        if progress_callback:
             progress_callback("Text Cleaning", 100.0, None)

        lang = detect_language_from_text(cleaned_text)

        if progress_callback:
             progress_callback("Language Detection", 100.0, None)

        return cleaned_text, page_count, lang

    def _try_direct_pdf_extraction(self, file_path, extract_text_from_pdf, basic_clean):
        try:
            text = extract_text_from_pdf(file_path)
            extracted_text, page_count = extract_text_from_pdf(file_path)

            if extracted_text.strip():
                cleaned_text = basic_clean(extracted_text)
                logger.info("Direct PDF text extraction succeeded (english hint)")
                return cleaned_text, page_count

        except Exception as e:
            logger.warning(f"Direct PDF text extraction failed, falling back to OCR: {e}")

        return None, 0

    def _classify_first_image(
        self,
        pil_image,
        classify_text_type,
        ml_model_predict: Optional[Callable[[Any], tuple[str, float]]] = None,
    ):
        """
        Classify first page image as handwritten / printed without saving to disk.
        """
        import numpy as np

        img_np = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2GRAY)

        text_type, text_type_confidence = classify_text_type(
            img_np,
            ml_model_predict=ml_model_predict,
            ml_conf_threshold=self.text_type_conf_threshold,
            return_confidence=True,
        )

        logger.info(
            "Detected text type for PDF: %s (confidence=%.3f)",
            text_type,
            text_type_confidence,
        )
        return text_type, text_type_confidence
    
    def _sniff_pdf_language(self, file_path, detect_language_from_text):
        """
        Try to detect language from embedded PDF text.
        Detect and reject legacy-encoded Sinhala (FMAbhaya-style).
        """

        def looks_like_legacy_sinhala(text: str) -> bool:
            """
            Detect common legacy Sinhala font patterns.
            These appear as ASCII junk like 'YS%', ',xld', 'ud;d', etc.
            """
            if not text:
                return False

            legacy_markers = [
                "YS%", ",xld", "ud;d", "kfuda", "wm ",
                ";", "%", "`", "/`", "Tn fõ"
            ]

            # If too many ASCII letters but no Sinhala Unicode range
            has_unicode_sinhala = any('\u0D80' <= ch <= '\u0DFF' for ch in text)
            ascii_ratio = sum(ch.isascii() for ch in text) / max(len(text), 1)

            if not has_unicode_sinhala and ascii_ratio > 0.8:
                # Check known patterns
                if any(marker in text for marker in legacy_markers):
                    return True

            return False

        try:
            import pdfplumber

            with pdfplumber.open(file_path) as pdf:
                if pdf.pages:
                    sample_text = pdf.pages[0].extract_text() or ""

                    # 🔥 If legacy Sinhala detected → force OCR
                    if looks_like_legacy_sinhala(sample_text):
                        logger.info("Detected legacy-encoded Sinhala in PDF. Forcing OCR.")
                        return "unknown"

                    lang = detect_language_from_text(sample_text)
                    logger.info(f"Detected script from PDF text sample: {lang}")
                    return lang
        except Exception as e:
            logger.debug(f"PDF language sniff failed: {e}")

        return "unknown"

    def _create_chunks(
        self,
        text: str,
        resource_id: str,
        progress_callback: Optional[Callable[[str, float, Optional[Dict[str, Any]]], None]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Split text into chunks and generate embeddings.
        
        Returns:
            List of chunk dictionaries with embeddings
        """
        from app.components.document_processing.services.embedding_service import embed_chunks
        
        try:
            embedded_chunks = embed_chunks(text, doc_id=resource_id, progress_callback=progress_callback)
            return embedded_chunks
        except Exception as e:
            logger.error(f"Failed to create chunks for resource {resource_id}: {e}")
            raise ValueError(f"Chunking failed: {e}")

    def _generate_pseudo_questions_for_chunk(self, text: str) -> str:
        sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 20]
        questions = []

        for s in sentences[:3]:
            questions.append(f"{s} යනු කුමක්ද?")
            questions.append(f"{s} ගැන පැහැදිලි කරන්න.")

        return "\n".join(questions[:5])

    def _save_chunks_to_db(self, chunks: List[Dict[str, Any]], resource_id: str):
        """Persist chunks with embeddings to database."""
        EXPECTED_DIM = 768  # gemini-embedding-001 dimension
        valid_chunks = 0

        for chunk_data in chunks:
            content = chunk_data.get("text")
            embedding = chunk_data.get("embedding")

            if not embedding or len(embedding) != EXPECTED_DIM:
                logger.error(
                    "Invalid embedding for chunk %s of resource %s. "
                    "Expected %d dimensions, got %s",
                    chunk_data.get("chunk_id"),
                    resource_id,
                    EXPECTED_DIM,
                    0 if not embedding else len(embedding)
                )
                raise ValueError(
                    f"Embedding generation failed for chunk {chunk_data.get('chunk_id')}"
                )
            chunk = ResourceChunk(
                resource_id=resource_id,
                chunk_index=chunk_data.get("chunk_id"),
                content=content,
                content_length=len(content) if content else None,
                token_count=None,
                embedding=chunk_data.get("embedding"),
                embedding_model=chunk_data.get("embedding_model"),
                start_char=chunk_data.get("start_char"),
                end_char=chunk_data.get("end_char"),
                pseudo_questions=(
                    self._generate_pseudo_questions_for_chunk(content)
                    if content else None
                ),
            )

            self.db.add(chunk)
            valid_chunks += 1
        
        logger.info("Prepared %d valid chunks for resource %s", valid_chunks, resource_id)

    def _create_document_embedding(
        self,
        text: str,
        resource_id: str,
        progress_callback: Optional[Callable[[str, float, Optional[Dict[str, Any]]], None]] = None,
    ) -> Optional[List[float]]:
        """Generate a document-level embedding for fast filtering."""
        from app.components.document_processing.services.embedding_service import embed_document_text
        
        try:
            embedding = embed_document_text(text, progress_callback=progress_callback)
            if embedding:
                logger.info("Generated document embedding for resource %s", resource_id)
            return embedding
        except Exception as e:
            logger.error(f"Failed to create document embedding for resource {resource_id}: {e}")
            return None
    
    def process_resource(
        self,
        resource: ResourceFile,
        resource_type: Optional[str] = None,
        progress_callback: Optional[Callable[[str, float, Optional[Dict[str, Any]]], None]] = None
    ) -> Dict[str, Any]:
        """
        Process a stored resource: extract text, chunk, embed, and save.
        
        Args:
            resource: ResourceFile instance with storage_path
            doc_type: Optional document type (for metadata only)
            progress_callback: Optional callback for progress updates (stage, percentage, details)
            
        Returns:
            Processing results including extracted text and chunk count
            
        Raises:
            ValueError: If resource file is missing or processing fails
        """
        # Stage 1
        if progress_callback:
            progress_callback("Starting Processing", 0.0, None)

        # Check if already processed
        if resource.extracted_text:

            if progress_callback:
                progress_callback("Checking Existing Data", 10.0, None)

            chunk_count = self.db.query(ResourceChunk).filter(
                ResourceChunk.resource_id == resource.id
            ).count()

            if chunk_count > 0:
                logger.info(
                    "Resource %s already processed with %d chunks",
                    resource.id,
                    chunk_count
                )

                if progress_callback:
                    progress_callback(
                        "Already Processed",
                        100.0,
                        {"chunks": chunk_count}
                    )

                return {
                    "resource_id": str(resource.id),
                    "status": "already_processed",
                    "extracted_text_length": len(resource.extracted_text),
                    "chunks_created": chunk_count,
                    "message": "Resource already processed, skipping"
                }
            else:
                logger.info(
                    "Resource %s has extracted text but no chunks, reprocessing",
                    resource.id
                )

        # Stage 2
        if progress_callback:
            progress_callback("Validating Document", 2.0, None)

        self._validate_resource_file(resource)

        logger.info(
            "Starting processing for resource %s: %s",
            resource.id,
            resource.original_filename
        )

        # Extract text (OCR stage handled inside)
        extracted_text, page_count, detected_language = self._extract_text(
            resource.storage_path,
            resource_type=resource_type,
            progress_callback=progress_callback
        )

        if not extracted_text.strip():
            raise ValueError("No text could be extracted from the document")

        logger.info(
            "Extracted %d characters from %d pages",
            len(extracted_text),
            page_count
        )

        # Stage after OCR
        if progress_callback:
            progress_callback(
                "OCR Completed",
                47.0,
                {"pages": page_count}
            )

        try:

            # Save extracted text & language
            resource.extracted_text = extracted_text
            resource.language = detected_language

            # Stage: embedding
            if progress_callback:
                progress_callback(
                    "Generating Document Embedding",
                    55.0,
                    None
                )

            if resource.document_embedding is None:

                logger.info(
                    "Generating document embedding for resource %s",
                    resource.id
                )

                from app.shared.ai.embeddings import EMBED_MODEL

                document_embedding = self._create_document_embedding(
                    extracted_text,
                    str(resource.id),
                    progress_callback=progress_callback,
                )

                if document_embedding:
                    resource.document_embedding = document_embedding
                    resource.embedding_model = EMBED_MODEL

            # Stage: chunking
            if progress_callback:
                progress_callback(
                    "Creating Text Chunks",
                    75.0,
                    None
                )

            chunks = self._create_chunks(extracted_text, str(resource.id), progress_callback=progress_callback)

            # Stage: saving
            if progress_callback:
                progress_callback(
                    "Saving Chunks",
                    95.0,
                    {"chunks": len(chunks)}
                )

            self._save_chunks_to_db(chunks, resource.id)

            self.db.commit()

            # Stage: completed
            if progress_callback:
                progress_callback(
                    "Processing Completed",
                    100.0,
                    None
                )

        except Exception:
            self.db.rollback()
            raise

        return {
            "resource_id": str(resource.id),
            "status": "completed",
            "pages": page_count,
            "extracted_text_length": len(extracted_text),
            "chunks_created": len(chunks),
            "message": "Resource processed successfully"
        }

    def is_need_to_analyze_layout(self, resource_type: Optional[str]) -> bool:

        # New OCR routing: answer sheets should avoid layout-block OCR because it
        # tends to keep printed snippets and miss longer handwritten answers.
        if resource_type in ["question_paper", "answer_sheet"]:
            logger.info(f"Resource type {resource_type} detected, forcing not to layout analysis for OCR.")
            return False
        
        logger.info(f"Resource type {resource_type} detected, enabling layout analysis for OCR.")
        return True
