import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from app.utils.sinhala_safety_engine import (
    extract_concepts, 
    concept_map_check, 
    detect_misconceptions, 
    attach_evidence
)

class TestSafetyVerification:

    def test_extract_concepts(self):
        # Using simple words that are safe for the basic [අ-෴] regex
        text = "මෙම පාඩම ඉතා බලන්න. අම්මා ගෙදර ගියා."
        concepts = extract_concepts(text)
        assert "පාඩම" in concepts
        assert "බලන්න" in concepts
        assert "අම්මා" in concepts
        assert "ගෙදර" in concepts
        assert "මෙම" not in concepts
        assert "ඉතා" not in concepts

    def test_concept_map_check(self):
        source = "ශාක ආහාර නිපදවයි."
        generated = "ශාක නිපදවයි." # missing "ආහාර"
        
        result = concept_map_check(generated, source)
        assert "ආහාර" in result["missing_concepts"]
        assert len(result["extra_concepts"]) == 0

    def test_detect_misconceptions_flagging(self):
        source = "පෘථිවිය සූර්යයා වටා ගමන් කරයි. එය ග්‍රහලෝකයකි."
        
        # Sentences with 6+ safe concepts to trigger flagging
        generated_unsupported = "අඟහරු ලෝකය රතු පාට අභ්‍යවකාශ වස්තුවකි. එහි පිටසක්වල ජීවීන් බොහෝ පිරිසක් වාසය කරති."
        
        misconceptions = detect_misconceptions(generated_unsupported, source)
        assert len(misconceptions) > 0

    @patch('app.services.semantic_similarity_service.SemanticSimilarityService.compute_pairwise_similarities')
    def test_attach_evidence(self, mock_compute):
        flagged = [{
            "sentence": "පිටසක්වල ජීවීන් සිටී",
            "severity": "high",
            "unseen_ratio": 0.8
        }]
        context = "පෘථිවියේ ජීවීන් සිටී. අභ්‍යවකාශය විශාලය."
        
        # Mock similarity matrix
        mock_compute.return_value = np.array([[0.9, 0.2]])
        
        enriched = attach_evidence(flagged, context)
        
        assert len(enriched) == 1
        assert enriched[0]["evidence"] == "පෘථිවියේ ජීවීන් සිටී"
        assert enriched[0]["semantic_similarity_score"] == 0.9

    def test_extract_concepts_short_words(self):
        # Words with length < 3 should be ignored by [අ-෴]{3,}
        text = "ඒ අ ඔ ඕ"
        assert extract_concepts(text) == set()
