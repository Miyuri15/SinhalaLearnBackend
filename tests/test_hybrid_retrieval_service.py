import pytest
from unittest.mock import MagicMock, patch
from uuid import uuid4
from app.services.hybrid_retrieval_service import HybridRetrievalService

class TestHybridRetrievalService:

    @pytest.fixture
    def mock_db(self):
        return MagicMock()

    @pytest.fixture
    def hybrid_service(self, mock_db):
        return HybridRetrievalService(mock_db)

    def test_tokenize_sinhala(self, hybrid_service):
        text = "මෙම පාඩම බලන්න."
        tokens = hybrid_service._tokenize_sinhala(text)
        assert "මෙම" in tokens
        assert "පාඩම" in tokens
        assert "බලන්න" in tokens

    def test_retrieve_no_resources(self, hybrid_service):
        hybrid_service.resource_service.list_resources_by_ids = MagicMock(return_value=[])
        
        results = hybrid_service.retrieve(resource_ids=[uuid4()], query="test", query_embedding=[0.1]*1536)
        
        assert results == []
        hybrid_service.resource_service.list_resources_by_ids.assert_called_once()

    def test_retrieve_with_embeddings_success(self, hybrid_service):
        res_id = uuid4()
        resource = MagicMock()
        resource.id = res_id
        resource.document_embedding = [0.1]*1536
        
        hybrid_service.resource_service.list_resources_by_ids = MagicMock(return_value=[resource])
        hybrid_service.resource_service.search_documents = MagicMock(return_value=[{"resource_id": res_id, "similarity": 0.9}])
        
        # Chunks for dense search
        chunk_hit = {"id": uuid4(), "resource_id": res_id, "content": "hit content", "similarity": 0.8}
        hybrid_service.chunk_service.vector_search = MagicMock(return_value=[chunk_hit])
        hybrid_service.chunk_service.get_chunks_by_resource = MagicMock(return_value=[MagicMock()]) # for step 4 logging

        results = hybrid_service.retrieve(resource_ids=[res_id], query="test", query_embedding=[0.1]*1536)

        assert len(results) == 1
        assert results[0]["content"] == "hit content"
        assert results[0]["rank"] == 1
        hybrid_service.resource_service.search_documents.assert_called_once()
        hybrid_service.chunk_service.vector_search.assert_called_once()

    def test_retrieve_bm25_fallback(self, hybrid_service):
        res_id = uuid4()
        resource = MagicMock()
        resource.id = res_id
        resource.document_embedding = None # No document embedding
        
        hybrid_service.resource_service.list_resources_by_ids = MagicMock(return_value=[resource])
        
        # BM25 needs chunks
        chunk = MagicMock()
        chunk.id = uuid4()
        chunk.resource_id = res_id
        chunk.content = "සිංහල පාඩම"
        chunk.pseudo_questions = "මොකක්ද මේ?"
        hybrid_service.chunk_service.get_chunks_by_resource = MagicMock(return_value=[chunk])
        
        # Dense search hit
        chunk_hit = {"id": chunk.id, "resource_id": res_id, "content": "සිංහල පාඩම", "similarity": 0.8}
        hybrid_service.chunk_service.vector_search = MagicMock(return_value=[chunk_hit])

        results = hybrid_service.retrieve(resource_ids=[res_id], query="සිංහල", query_embedding=[0.1]*1536)

        assert len(results) == 1
        assert results[0]["content"] == "සිංහල පාඩම"
        # BM25 should have been triggered because top_resource_ids was empty after search_documents (which wasn't even called)
        hybrid_service.chunk_service.get_chunks_by_resource.assert_any_call([res_id])
        hybrid_service.chunk_service.vector_search.assert_called_once()

    def test_retrieve_dense_fallback_to_manual_cosine(self, hybrid_service):
        res_id = uuid4()
        resource = MagicMock()
        resource.id = res_id
        resource.document_embedding = [0.1]*1536
        
        hybrid_service.resource_service.list_resources_by_ids = MagicMock(return_value=[resource])
        hybrid_service.resource_service.search_documents = MagicMock(return_value=[{"resource_id": res_id}])
        
        # Vector search returns nothing
        hybrid_service.chunk_service.vector_search = MagicMock(return_value=[])
        
        # Manual fallback chunks
        chunk = MagicMock()
        chunk.id = uuid4()
        chunk.resource_id = res_id
        chunk.content = "fallback content"
        chunk.embedding = [0.2]*1536
        chunk.chunk_index = 0
        chunk.embedding_model = "test-model"
        hybrid_service.chunk_service.get_chunks_by_resource = MagicMock(return_value=[chunk])
        hybrid_service.chunk_service.cosine_similarity = MagicMock(return_value=0.95)

        results = hybrid_service.retrieve(resource_ids=[res_id], query="test", query_embedding=[0.1]*1536)

        assert len(results) == 1
        assert results[0]["content"] == "fallback content"
        assert results[0]["similarity"] == 0.95
        hybrid_service.chunk_service.cosine_similarity.assert_called_once()

    def test_retrieve_malformed_input_empty_query(self, hybrid_service):
        # Boundary case: empty query
        res_id = uuid4()
        resource = MagicMock()
        resource.id = res_id
        resource.document_embedding = [0.1]*1536
        
        hybrid_service.resource_service.list_resources_by_ids = MagicMock(return_value=[resource])
        hybrid_service.resource_service.search_documents = MagicMock(return_value=[{"resource_id": res_id}])
        hybrid_service.chunk_service.vector_search = MagicMock(return_value=[])
        hybrid_service.chunk_service.get_chunks_by_resource = MagicMock(return_value=[])

        results = hybrid_service.retrieve(resource_ids=[res_id], query="", query_embedding=[0.1]*1536)
        
        # Should handle it gracefully
        assert results == []
