# app/components/document_processing/services/embedding_service.py

from typing import List, Dict, Optional, Callable, Any

from app.shared.ai.embeddings import generate_embedding, EMBED_MODEL
from app.components.document_processing.utils.text_cleaner import basic_clean
from app.components.document_processing.utils.chunker import chunk_text


def generate_text_embedding(text: str) -> List[float]:
    cleaned = basic_clean(text)
    if not cleaned:
        return []
    return generate_embedding(
        cleaned,
        service_name="text_embedding",
        metadata_json={
            "source": "generate_text_embedding",
        },
    )


def embed_document_text(
    text: str,
    progress_callback: Optional[Callable[[str, float, Optional[Dict[str, Any]]], None]] = None,
) -> List[float]:
    if not text or not text.strip():
        return []
    cleaned = basic_clean(text)
    if not cleaned:
        return []
    if progress_callback:
        progress_callback("Generating Document Embedding", 55.0, {"status": "in_progress"})
    embedding = generate_embedding(
        cleaned,
        service_name="document_embedding",
        metadata_json={
            "source": "embed_document_text",
        },
    )
    if progress_callback:
        progress_callback("Document Embedding Complete", 60.0, {"dimensions": len(embedding)})
    return embedding


def embed_chunks(
    text: str,
    doc_id: Optional[str] = None,
    max_tokens: int = 220,
    overlap_tokens: int = 60,
    progress_callback: Optional[Callable[[str, float, Optional[Dict[str, Any]]], None]] = None,
) -> List[Dict]:
    if not text or not text.strip():
        return []

    cleaned = basic_clean(text)
    chunk_list = chunk_text(
        cleaned,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens
    )

    results = []
    total_chunks = len(chunk_list)

    for ch in chunk_list:
        c_text = ch["text"]
        c_id = ch["chunk_id"]
        c_numbering = ch.get("numbering")
        c_start = ch.get("start_char", 0)
        c_end = ch.get("end_char", len(c_text))

        if progress_callback and total_chunks > 0:
            # Map chunk progress across the 75→95 range
            chunk_progress = 75.0 + ((c_id / total_chunks) * 20.0)
            progress_callback(
                "Embedding Chunk",
                round(chunk_progress, 1),
                {
                    "current_chunk": c_id + 1,
                    "total_chunks": total_chunks,
                    "current_action": f"Embedding chunk {c_id + 1} of {total_chunks}",
                },
            )

        vec = generate_embedding(
            c_text,
            resource_id=doc_id,
            service_name="chunk_embedding",
            metadata_json={
                "source": "embed_chunks",
                "doc_id": str(doc_id) if doc_id else None,
                "chunk_id": c_id,
                "global_id": f"{doc_id}_{c_id}" if doc_id else str(c_id),
            },
        )

        global_id = f"{doc_id}_{c_id}" if doc_id else str(c_id)

        results.append({
            "chunk_id": c_id,
            "global_id": global_id,
            "text": c_text,
            "numbering": c_numbering,
            "embedding": vec,
            "embedding_model": EMBED_MODEL,
            "start_char": c_start,
            "end_char": c_end,
        })

    return results
