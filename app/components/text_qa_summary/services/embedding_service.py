import torch
from typing import List
import numpy as np
from app.core.config import settings


class EmbeddingService:
    _model = None
    
    @classmethod
    def get_model(cls):
        if cls._model is None:
            from sentence_transformers import SentenceTransformer

            model_name = settings.MODEL_EMBEDDING_NAME
            print(f"[DEBUG] Loading embedding model: {model_name}")
            try:
                cls._model = SentenceTransformer(model_name)
                print(f"[DEBUG] Embedding model loaded successfully: {model_name}")
            except Exception as e:
                print(f"[ERROR] Failed to load embedding model {model_name}: {e}")
                raise
        return cls._model
    
    @staticmethod
    def get_embeddings(texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for a list of texts
        """
        model = EmbeddingService.get_model()
        
        # Handle empty texts
        if not texts:
            return []
        
        # Get embeddings
        try:
            embeddings = model.encode(
                texts,
                convert_to_tensor=False,
                show_progress_bar=False,
                normalize_embeddings=True
            )
            
            print(f"[DEBUG] Generated embeddings for {len(texts)} texts")
            return embeddings.tolist()
            
        except Exception as e:
            print(f"[ERROR] Error generating embeddings: {e}")
            raise
    
    @staticmethod
    def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors
        """
        if not vec1 or not vec2:
            return 0.0
            
        try:
            v1 = np.array(vec1)
            v2 = np.array(vec2)
            
            dot_product = np.dot(v1, v2)
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return float(dot_product / (norm1 * norm2))
        except Exception as e:
            print(f"[ERROR] Error calculating cosine similarity: {e}")
            return 0.0
