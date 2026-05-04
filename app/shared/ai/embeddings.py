# app/shared/ai/embeddings.py

import logging
import threading
from app.core.gemini_client import GeminiClient
from app.core.config import settings
from google.genai import types
import huggingface_hub
logger = logging.getLogger(__name__)

if settings.HF_TOKEN:
    huggingface_hub.login(token=settings.HF_TOKEN, add_to_git_credential=False)

# Global semaphore to prevent CPU thrashing during heavy XLM-R math
# (One encoding task at a time per backend process)
ml_semaphore = threading.Semaphore(1)

# Global thread-safe cache for sentence embeddings
_embedding_cache = {}
_cache_lock = threading.Lock()


_xlmr = None
_xlmr_lock = threading.Lock()


def get_xlmr_model():
    global _xlmr
    if _xlmr is None:
        with _xlmr_lock:
            if _xlmr is None:
                logger.info("Loading XLM-R sentence transformer model")
                from sentence_transformers import SentenceTransformer

                _xlmr = SentenceTransformer(
                    "sentence-transformers/paraphrase-xlm-r-multilingual-v1"
                )
                logger.info("Loaded XLM-R sentence transformer model")
    return _xlmr

EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 768


def generate_embedding(
    text: str,
    user_id=None,
    session_id=None,
    message_id=None,
    resource_id=None,
    service_name: str = "embedding_generation",
    metadata_json: dict | None = None,
) -> list[float]:
    """
    Generate a 768-dim embedding using Gemini and log API usage.
    """
    if not text or not text.strip():
        logger.debug("Skipping embedding generation for empty text")
        return []

    import time
    import random
    from app.services.api_usage_log_service import ApiUsageLogService

    request_start_time = time.time()
    request_id = f"embedding-{int(request_start_time * 1000)}-{random.randint(1000, 9999)}"

    try:
        logger.debug("Generating embedding via Gemini for %d characters", len(text))
        client = GeminiClient.get_client()
        if not client:
            logger.warning("Gemini client unavailable; returning empty embedding")
            return []

        result = client.models.embed_content(
            model=EMBED_MODEL,
            contents=text,
            config=types.EmbedContentConfig(
                output_dimensionality=EMBED_DIM
            )
        )

        embedding_values = []

        if result.embeddings:
            embedding_values = result.embeddings[0].values

        duration_ms = round((time.time() - request_start_time) * 1000, 2)

        logger.info(
            "Generated embedding via Gemini: status=%s dimensions=%d duration_ms=%.2f",
            "success" if embedding_values else "empty_response",
            len(embedding_values),
            duration_ms,
        )

        ApiUsageLogService.create_log(
            request_id=request_id,
            provider="gemini",
            service_name=service_name,
            model_name=EMBED_MODEL,
            status="success" if embedding_values else "empty_response",
            user_id=user_id,
            session_id=session_id,
            message_id=message_id,
            prompt_chars=len(text or ""),
            response_chars=0,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            attempt_number=1,
            max_retries=0,
            is_retry=False,
            duration_ms=duration_ms,
            metadata_json={
                **(metadata_json or {}),
                "resource_id": str(resource_id) if resource_id else None,
                "embedding_dimensions": len(embedding_values),
                "output_dimensionality": EMBED_DIM,
            },
        )

        return embedding_values

    except Exception as e:
        duration_ms = round((time.time() - request_start_time) * 1000, 2)

        logger.exception(
            "Gemini embedding generation failed after %.2f ms", duration_ms
        )

        ApiUsageLogService.create_log(
            request_id=request_id,
            provider="gemini",
            service_name=service_name,
            model_name=EMBED_MODEL,
            status="failed",
            user_id=user_id,
            session_id=session_id,
            message_id=message_id,
            prompt_chars=len(text or ""),
            response_chars=0,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            attempt_number=1,
            max_retries=0,
            is_retry=False,
            error_type=type(e).__name__,
            error_message=str(e)[:1000],
            duration_ms=duration_ms,
            metadata_json={
                **(metadata_json or {}),
                "resource_id": str(resource_id) if resource_id else None,
                "output_dimensionality": EMBED_DIM,
            },
        )

        return []

def semantic_similarity(a: str, b: str) -> float:
    """
    Compute cosine similarity using local XLM-R
    (fast, offline, safe fallback).
    """
    if not a.strip() or not b.strip():
        logger.debug("Semantic similarity skipped for empty input")
        return 0.0

    try:
        from sentence_transformers import util

        # Check cache first
        with _cache_lock:
            a_vec = _embedding_cache.get(a)
            b_vec = _embedding_cache.get(b)

        # Encode if missing (batching is preferred, but this is a fallback)
        if a_vec is None or b_vec is None:
            with ml_semaphore:
                if a_vec is None:
                    a_vec = get_xlmr_model().encode(a, convert_to_tensor=True)
                    with _cache_lock: _embedding_cache[a] = a_vec
                if b_vec is None:
                    b_vec = get_xlmr_model().encode(b, convert_to_tensor=True)
                    with _cache_lock: _embedding_cache[b] = b_vec

        sim = float(util.cos_sim(a_vec, b_vec))
        logger.debug("Computed semantic similarity: %.4f", sim)
        return min(sim * 1.2, 1.0)
    except Exception as e:
        logger.exception("Semantic similarity computation failed")
        return 0.5


def ensure_sentences_cached(sentences: list[str]):
    """
    Batch-encode multiple sentences into the global cache in one pass.
    Drastically faster than encoding one by one.
    """
    if not sentences:
        logger.debug("No sentences provided for embedding cache warmup")
        return

    # Deduplicate and filter already cached
    to_encode = []
    with _cache_lock:
        for s in set(sentences):
            if s and s.strip() and s not in _embedding_cache:
                to_encode.append(s)

    if not to_encode:
        logger.debug("All %d sentences already present in embedding cache", len(set(sentences)))
        return

    # Batch encode with semaphore
    logger.info("Batch encoding %d new sentences into embedding cache", len(to_encode))
    with ml_semaphore:
        new_embs = get_xlmr_model().encode(
            to_encode, 
            batch_size=32, 
            convert_to_tensor=True,
            show_progress_bar=True
        )
        
        with _cache_lock:
            for i, s in enumerate(to_encode):
                _embedding_cache[s] = new_embs[i]

    logger.info("Cached %d new sentence embeddings", len(to_encode))
