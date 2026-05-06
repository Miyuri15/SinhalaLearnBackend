import pytest
from app.services.answerability_service import AnswerabilityService

class TestAnswerabilityService:

    def test_extract_key_terms(self):
        # Using words that are safe for the basic [අ-෴] regex
        text = "මෙම පාඩම බලන්න. අම්මා කොහේද?"
        terms = AnswerabilityService.extract_key_terms(text)
        assert "පාඩම" in terms
        assert "බලන්න" in terms
        assert "අම්මා" in terms
        # Stopwords
        assert "මෙම" not in terms
        assert "කොහේද" not in terms

    def test_has_relevant_content_greeting(self):
        assert AnswerabilityService.has_relevant_content("ආයුබෝවන්", "Context", [])
        assert AnswerabilityService.has_relevant_content("hello", "Context", [])

    def test_has_relevant_content_summary_intent(self):
        # Summary intent is always answerable if chunks exist
        chunks = [{"id": 1, "content": "some content"}]
        assert AnswerabilityService.has_relevant_content("සාරාංශ කරන්න", "Context", chunks, intent="summary")

    def test_has_relevant_content_qa_answer_threshold(self):
        # Use simple words that won't be split by the basic regex
        question = "පාඩම කුමක්ද?"
        context = "මෙම පාඩම ඉතිහාසය ගැනයි."
        chunks = [{"similarity": 0.8}]
        
        # Should be answerable if threshold is met
        assert AnswerabilityService.has_relevant_content(question, context, chunks, intent="qa_answer", threshold=0.1)

    def test_extract_key_terms_english(self):
        text = "Tell me about photosynthesis process"
        terms = AnswerabilityService.extract_key_terms(text)
        assert "photosynthesis" in terms
        assert "process" in terms
        assert "about" not in terms
