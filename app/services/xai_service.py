# app/services/xai_service.py
import logging
import re
import string
from typing import Dict, List, Optional
from uuid import UUID

try:
    from sinling import SinhalaTokenizer
except ImportError:
    SinhalaTokenizer = None

logger = logging.getLogger(__name__)


class XAIService:
    """Explainable AI service that provides transparency into RAG responses."""

    SINHALA_STOP_WORDS = {
        "අතර",
        "අප",
        "අපි",
        "ඇත",
        "ඇති",
        "ඇතුළු",
        "ඒ",
        "එය",
        "එම",
        "ඔබ",
        "කර",
        "කරයි",
        "කරන",
        "කිරීම",
        "කරන්න",
        "කොට",
        "තුළ",
        "ද",
        "දී",
        "නම්",
        "නිසා",
        "බව",
        "මෙම",
        "මෙය",
        "මේ",
        "ය",
        "ලෙස",
        "වල",
        "වශයෙන්",
        "වැනි",
        "විසින්",
        "සඳහා",
        "සහ",
        "හා",
        "වේ",
        "දෙන්න",
        "දුන්නා",
        "ලදී",
        "තවද",
        "තුමා",
        "වීම",
    }
    PUNCTUATION = set(string.punctuation) | {"।", "“", "”", "‘", "’"}
    _tokenizer = SinhalaTokenizer() if SinhalaTokenizer else None
    
    @staticmethod
    def generate_explanation(
        user_query: str,
        generated_answer: str,
        retrieved_chunks: List[Dict],
        safety_report: Optional[Dict] = None,
        retrieval_metadata: Optional[Dict] = None,
    ) -> Dict:
        """
        Generate a comprehensive explanation for how the answer was produced.
        """
        
        # 1. Chunk contribution analysis
        chunk_analysis = XAIService._analyze_chunk_contributions(
            generated_answer, retrieved_chunks
        )
        
        # 2. Safety explanation (if applicable)
        safety_explanation = XAIService._explain_safety(
            safety_report, generated_answer, retrieved_chunks
        ) if safety_report else None
        
        # 3. Confidence breakdown
        confidence_breakdown = XAIService._breakdown_confidence(
            safety_report, chunk_analysis
        )
        
        # 4. Concept tracing
        concept_tracing = XAIService._trace_concepts(
            generated_answer, retrieved_chunks
        )
        
        return {
            "chunk_contributions": chunk_analysis,
            "safety_explanation": safety_explanation,
            "confidence_breakdown": confidence_breakdown,
            "concept_tracing": concept_tracing,
            "retrieval_stats": retrieval_metadata or {},
            "explanation_summary": XAIService._generate_summary(
                confidence_breakdown, chunk_analysis, safety_explanation
            )
        }
    
    @staticmethod
    def _analyze_chunk_contributions(
        answer: str, chunks: List[Dict]
    ) -> List[Dict]:
        """Analyze which chunks contributed most to the answer."""
        contributions = []
        answer_key_terms = set(XAIService._extract_key_terms(answer))
        
        for i, chunk in enumerate(chunks[:5]):  # Limit to top 5 chunks
            chunk_text = chunk.get("content", "")
            
            # Simple overlap analysis
            answer_words = set(answer.lower().split())
            chunk_words = set(chunk_text.lower().split())
            overlap = len(answer_words & chunk_words)
            total_unique = len(answer_words | chunk_words)
            chunk_key_terms = XAIService._extract_key_terms(chunk_text)
            shared_key_terms = [
                term for term in chunk_key_terms if term in answer_key_terms
            ][:10]
            
            contribution_score = overlap / total_unique if total_unique > 0 else 0
            
            contributions.append({
                "rank": i + 1,
                "chunk_id": str(chunk.get("id", "")),
                "similarity_score": chunk.get("similarity", 0),
                "contribution_score": round(contribution_score, 3),
                "preview": chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text,
                "key_terms": shared_key_terms,
            })
        
        return contributions

    @staticmethod
    def _extract_key_terms(text: str) -> List[str]:
        """Extract unique non-stop-word terms while preserving text order."""
        terms = []
        seen = set()

        for term in XAIService._tokenize_terms(text):
            term = XAIService._normalize_token(term)

            if (
                not term
                or XAIService._is_number_token(term)
                or term in XAIService.SINHALA_STOP_WORDS
                or term in seen
            ):
                continue
            seen.add(term)
            terms.append(term)

        return terms

    @staticmethod
    def _tokenize_terms(text: str) -> List[str]:
        """Tokenize terms with sinling when available, otherwise regex fallback."""
        text = XAIService._normalize_unicode(text)

        if XAIService._tokenizer:
            tokens = XAIService._tokenizer.tokenize(text)
            return [
                token.strip().lower()
                for token in tokens
                if token.strip()
                and not all(char in XAIService.PUNCTUATION for char in token)
            ]

        return re.findall(r"[^\W_]+", text.lower(), flags=re.UNICODE)

    @staticmethod
    def _normalize_unicode(text: str) -> str:
        """Remove hidden Unicode characters that split visually identical terms."""
        return text.replace("\u200d", "")

    @staticmethod
    def _normalize_token(token: str) -> str:
        """Apply lightweight Sinhala token normalization."""
        token = token.replace("\u200d", "")

        for suffix in ("ටත්", "ට"):
            if token.endswith(suffix) and len(token) > len(suffix) + 2:
                return token[: -len(suffix)]

        return token

    @staticmethod
    def _is_number_token(token: str) -> bool:
        """Return true for standalone numeric tokens and formatted numbers."""
        stripped = token.strip(".,:;!?()[]{}+-/%")
        numeric = stripped.translate(str.maketrans("", "", ".,:/+-"))
        return bool(numeric) and all(char.isdigit() for char in numeric)
    
    @staticmethod
    def _explain_safety(
        safety_report: Optional[Dict], 
        answer: str, 
        chunks: List[Dict]
    ) -> Dict:
        """Explain why certain parts were flagged."""
        if not safety_report:
            return {"has_issues": False, "message": "No safety issues detected."}
        
        flagged = safety_report.get("flagged", [])
        missing = safety_report.get("missing_concepts", [])
        extra = safety_report.get("extra_concepts", [])
        
        explanation = {
            "has_issues": len(flagged) > 0 or len(missing) > 0 or len(extra) > 0,
            "flagged_count": len(flagged),
            "missing_concepts_count": len(missing),
            "extra_concepts_count": len(extra),
            "details": []
        }
        
        # Explain flagged sentences
        for item in flagged[:3]:  # Limit to top 3
            explanation["details"].append({
                "type": "flagged_sentence",
                "sentence": item.get("sentence", ""),
                "severity": item.get("severity", "low"),
                "unseen_ratio": item.get("unseen_ratio", 0),
                "explanation": f"This sentence contains {int(item.get('unseen_ratio', 0)*100)}% concepts not found in the source materials."
            })
        
        # Explain missing concepts
        if missing:
            explanation["details"].append({
                "type": "missing_concepts",
                "concepts": missing[:10],
                "explanation": f"These concepts from the source material weren't included in the answer."
            })
        
        # Explain extra concepts
        if extra:
            explanation["details"].append({
                "type": "extra_concepts",
                "concepts": extra[:10],
                "explanation": f"These concepts in the answer weren't found in the source material."
            })
        
        return explanation
    
    @staticmethod
    def _breakdown_confidence(safety_report: Optional[Dict], chunk_analysis: List[Dict]) -> Dict:
        """Break down the confidence score into components."""
        
        # Calculate retrieval confidence
        retrieval_scores = [c.get("similarity_score", 0) for c in chunk_analysis if c.get("similarity_score")]
        retrieval_confidence = sum(retrieval_scores) / len(retrieval_scores) if retrieval_scores else 0
        
        # Calculate coverage confidence
        coverage_scores = [c.get("contribution_score", 0) for c in chunk_analysis]
        coverage_confidence = sum(coverage_scores) / len(coverage_scores) if coverage_scores else 0
        
        # Get safety confidence from report
        safety_confidence = safety_report.get("confidence_score", 1.0) if safety_report else 1.0
        
        # Calculate overall
        overall = (retrieval_confidence * 0.3 + coverage_confidence * 0.3 + safety_confidence * 0.4)
        
        return {
            "overall": round(overall, 3),
            "components": [
                {"name": "Retrieval Relevance", "score": round(retrieval_confidence, 3), "weight": 0.3},
                {"name": "Content Coverage", "score": round(coverage_confidence, 3), "weight": 0.3},
                {"name": "Factual Consistency", "score": round(safety_confidence, 3), "weight": 0.4},
            ]
        }
    
    @staticmethod
    def _trace_concepts(answer: str, chunks: List[Dict]) -> Dict:
        """Trace key concepts back to their source chunks."""
        from app.utils.sinhala_safety_engine import extract_concepts
        
        answer_concepts = list(extract_concepts(answer))[:15]  # Top 15 concepts
        
        concept_sources = []
        for concept in answer_concepts:
            sources = []
            for chunk in chunks[:3]:
                chunk_text = chunk.get("content", "")
                if concept in chunk_text:
                    sources.append({
                        "chunk_rank": chunks.index(chunk) + 1,
                        "preview": chunk_text[:100] + "..." if len(chunk_text) > 100 else chunk_text
                    })
            
            if sources:
                concept_sources.append({
                    "concept": concept,
                    "found_in_sources": len(sources) > 0,
                    "source_count": len(sources),
                    "sources": sources[:2]  # Limit to top 2 sources
                })
        
        return {
            "total_concepts": len(answer_concepts),
            "concepts_with_sources": len(concept_sources),
            "concept_details": concept_sources[:10]  # Limit display
        }
    
    @staticmethod
    def _generate_summary(confidence: Dict, chunks: List[Dict], safety: Optional[Dict]) -> str:
        """Generate a human-readable summary without misleading numbers."""
        
        chunk_count = len(chunks)
        has_safety_issues = safety.get("has_issues", False) if safety else False
        
        # Determine quality level without using the numeric confidence
        if chunk_count >= 3 and not has_safety_issues:
            quality = "highly confident"
        elif chunk_count >= 1 and not has_safety_issues:
            quality = "confident"
        elif chunk_count >= 1 and has_safety_issues:
            quality = "cautious"
        else:
            quality = "limited confidence"
        
        summary = f"I'm {quality} about this answer. "
        summary += f"It was generated from {chunk_count} relevant content chunks. "
        
        if has_safety_issues:
            summary += "Some parts may need verification against the source material."
        else:
            summary += "All content appears consistent with the source material."
        
        return summary

