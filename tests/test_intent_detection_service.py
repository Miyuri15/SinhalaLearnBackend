import pytest
from unittest.mock import patch
import numpy as np

from app.services.intent_detection_service import IntentDetectionService

class TestIntentDetectionService:
    
    def test_rule_based_summary(self):
        query = "මෙම පාඩම සාරාංශ කරන්න"
        intent = IntentDetectionService.detect_intent(query)
        assert intent == "summary"

    def test_rule_based_qa_generate(self):
        query = "මෙම පාඩමෙන් ප්‍රශ්න සාදන්න"
        intent = IntentDetectionService.detect_intent(query)
        assert intent == "qa_generate"

    def test_rule_based_explanation(self):
        query = "කරුණාකර මෙය විස්තර කරන්න"
        intent = IntentDetectionService.detect_intent(query)
        assert intent == "explanation"

    def test_rule_based_qa_answer_question_mark(self):
        query = "මෙය යනු කුමක්ද?"
        intent = IntentDetectionService.detect_intent(query)
        assert intent == "qa_answer"

    def test_rule_based_qa_answer_keyword(self):
        query = "මට කියන්න මේක ගැන"
        intent = IntentDetectionService.detect_intent(query)
        assert intent == "qa_answer"

    @patch('app.services.intent_detection_service.generate_text_embedding')
    def test_semantic_gate_fallback(self, mock_generate_embedding):
        # Mock embedding to return None to trigger fallback immediately
        mock_generate_embedding.return_value = None
        
        query = "unknown query without keywords"
        intent = IntentDetectionService.detect_intent(query)
        
        assert intent == "qa_answer"
        mock_generate_embedding.assert_called_once_with(query)

    @patch('app.services.intent_detection_service.generate_text_embedding')
    def test_semantic_gate_success(self, mock_generate_embedding):
        # We need to mock embedding to return specific vectors to trigger high cosine similarity
        
        # Helper to return dummy embeddings
        def dummy_embedding(text):
            if text == "semantic greeting match":
                # Matches greeting anchor
                return [1.0, 0.0, 0.0]
            elif text == "හායි ආයුබෝවන් කොහොමද":
                # Greeting anchor
                return [1.0, 0.0, 0.0]
            else:
                # Other anchors
                return [0.0, 1.0, 0.0]
                
        mock_generate_embedding.side_effect = dummy_embedding
        
        query = "semantic greeting match"
        intent = IntentDetectionService.detect_intent(query)
        
        # It should match 'greeting' because it has a cosine similarity of 1.0 with the greeting anchor
        assert intent == "greeting"

    @patch('app.services.intent_detection_service.generate_text_embedding')
    def test_semantic_gate_low_score_fallback(self, mock_generate_embedding):
        # Mock embeddings to always have a low similarity
        def dummy_embedding(text):
            if text == "low similarity query":
                return [1.0, 0.0, 0.0]
            else:
                return [0.0, 1.0, 0.0]
                
        mock_generate_embedding.side_effect = dummy_embedding
        
        query = "low similarity query"
        intent = IntentDetectionService.detect_intent(query)
        
        # Since score is 0.0 < 0.65, it should fallback to 'qa_answer'
        assert intent == "qa_answer"

    def test_empty_query(self):
        # Test boundary case with empty query
        query = ""
        # The intent detection service currently assumes string input, empty string will go to semantic gate
        # since it doesn't match rules. The empty string embedding behavior depends on `generate_text_embedding`.
        # We can mock it.
        with patch('app.services.intent_detection_service.generate_text_embedding', return_value=None):
            intent = IntentDetectionService.detect_intent(query)
            assert intent == "qa_answer"

    def test_whitespace_query(self):
        query = "   "
        with patch('app.services.intent_detection_service.generate_text_embedding', return_value=None):
            intent = IntentDetectionService.detect_intent(query)
            assert intent == "qa_answer"
