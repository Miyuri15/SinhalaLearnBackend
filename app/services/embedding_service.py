# app/services/embedding_service.py
import numpy as np
from functools import lru_cache

class EmbeddingService:
    _model = None

    @classmethod
    def get_model(cls):
        if cls._model is None:
            # Add timeout and retry configuration
            import torch
            from sentence_transformers import SentenceTransformer

            cls._model = SentenceTransformer(
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                device='cpu'  # Explicitly set device
            )
        return cls._model

    @classmethod
    def embed(cls, text: str) -> np.ndarray:
        model = cls.get_model()
        return model.encode(text, normalize_embeddings=True)

    @classmethod
    def embed_batch(cls, texts: list) -> np.ndarray:
        """Batch embed multiple texts for efficiency"""
        if not texts:
            return np.array([])
        model = cls.get_model()
        # Use larger batch size for better performance
        return model.encode(
            texts, 
            normalize_embeddings=True,
            batch_size=32,  # Adjust based on your memory
            show_progress_bar=False
        )
