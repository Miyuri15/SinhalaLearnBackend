# app/services/rag_service.py
import logging
from typing import List, Dict, Optional
from uuid import UUID
from sqlalchemy.orm import Session
import re 

from app.services.hybrid_retrieval_service import HybridRetrievalService
from app.services.resource_chunk_service import ResourceChunkService
from app.services.message_context_service import MessageContextService
from app.services.message_service import MessageService
from app.utils.sinhala_prompt_builder import build_qa_prompt, build_direct_answer_prompt
from app.utils.sinhala_summary_prompt_builder import build_summary_prompt
from app.utils.sinhala_safety_engine import concept_map_check, detect_misconceptions, attach_evidence
from app.services.message_safety_service import MessageSafetyService
from app.core.gemini_client import GeminiClient
from app.services.intent_detection_service import IntentDetectionService
from app.services.answerability_service import AnswerabilityService
from app.services.xai_service import XAIService

logger = logging.getLogger(__name__)


class RAGService:
    """RAG orchestration with hybrid retrieval, grounded generation, and safety checks."""
    
    # Threshold for considering retrieved content relevant
    RELEVANCE_THRESHOLD = 0.20  # Lowered to reduce false negatives; LLM will handle the final gate.


    def __init__(self, db: Session):
        self.db = db
        self.chunk_service = ResourceChunkService(db)
        self.context_service = MessageContextService(db)
        self.message_service = MessageService(db)
        self.safety_service = MessageSafetyService(db)

    def extract_question_count(self, query: str) -> int:
        match = re.search(r"\d+", query)
        if match:
            return int(match.group())
        return 5

    def generate_response(
        self,
        session_id: UUID,
        user_message_id: UUID,
        user_query: str,
        resource_ids: List[UUID],
        query_embedding: Optional[List[float]] = None,
        bm25_k: int = 20,
        final_k: int = 8,
        grade_level: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Dict:
        """Hybrid retrieval → grounded generation → safety checks → logging"""

        # -----------------------------
        # 0. Detect intent first
        # -----------------------------
        intent = IntentDetectionService.detect_intent(user_query)
        logger.info(f"Detected intent: {intent}")

        # -----------------------------
        # 1. Check for greetings/chit-chat (no RAG needed)
        # -----------------------------
        if intent == "greeting":
            greeting_text = "ආයුබෝවන්! මට ඔබට උදව් කළ හැකිය. කරුණාකර ඔබගේ ප්‍රශ්නය අසන්න."

            assistant_msg = self.message_service.create_assistant_message(
                session_id=session_id,
                content=greeting_text,
                model_info={"model_name": "rule-based"},
                parent_msg_id=user_message_id
            )

            logger.info("Greeting detected early — skipping RAG pipeline")

            return {
                "assistant_message_id": assistant_msg.id,
                "content": greeting_text,
                "sources": [],
                "retrieval_metadata": {"intent": "greeting", "used_chunks": 0},
                "safety": {
                    "is_valid": True,
                    "missing_concepts": [],
                    "extra_concepts": [],
                    "flagged": [],
                },
                "xai_explanation": None  # No XAI for greetings
            }

        if not query_embedding:
            raise ValueError("Query embedding is required for hybrid retrieval")

        # -----------------------------
        # 2. Hybrid retrieval
        # -----------------------------
        hybrid_service = HybridRetrievalService(self.db)
        hits = hybrid_service.retrieve(
            resource_ids=resource_ids,
            query=user_query,
            query_embedding=query_embedding,
            bm25_k=bm25_k,
            final_k=final_k,
        )

        # -----------------------------
        # 3. Check if we have any retrieved content
        # -----------------------------
        if not hits:
            # Zero-hallucination refusal - no chunks found at all
            refusal_text = "මෙම ප්‍රශ්නයට අදාල තොරතුරු ලබා දී ඇති අන්තර්ගතයේ නොමැත."
            assistant_msg = self.message_service.create_assistant_message(
                session_id=session_id,
                content=refusal_text,
                model_info={"model_name": "gemini-3-flash-preview"},
                parent_msg_id=user_message_id
            )
            
            return {
                "assistant_message_id": assistant_msg.id,
                "content": refusal_text,
                "sources": [],
                "retrieval_metadata": {"bm25_k": bm25_k, "final_k": final_k, "used_chunks": 0},
                "safety": None,  # No safety for unanswerable
                "xai_explanation": None  # No XAI for unanswerable
            }

        logger.info("Hybrid retrieval returned %d hits", len(hits))

        # -----------------------------
        # 4. Build context from retrieved chunks
        # -----------------------------
        context = "\n\n".join(h["content"] for h in hits)
        logger.info("Built context of length %d", len(context))

        # -----------------------------
        # 5. Determine if question is answerable based on intent and content
        # -----------------------------
        # Special handling for summary requests - they should always be considered answerable
        if intent == "summary":
            logger.info("Summary intent detected - treating as answerable regardless of relevance score")
            is_unanswerable = False
        else:
            # For other intent types, check if content is relevant
            # Pass the intent to the answerability service
            has_relevant_content = AnswerabilityService.has_relevant_content(
                user_query, 
                context, 
                hits, 
                intent=intent,  # Pass the intent
                threshold=self.RELEVANCE_THRESHOLD
            )
            
            # Determine if question is truly answerable
            is_unanswerable = not has_relevant_content

            if is_unanswerable:
                logger.warning("Question deemed unanswerable despite retrieval hits: %s", user_query)
                logger.warning("Top hit similarity: %.3f", hits[0].get("similarity", 0) if hits else 0)

        # -----------------------------
        # 6. Handle unanswerable questions (including non-summary)
        # -----------------------------
        if is_unanswerable:
            refusal_text = "මෙම ප්‍රශ්නයට අදාල තොරතුරු ලබා දී ඇති අන්තර්ගතයේ නොමැත."
            
            assistant_msg = self.message_service.create_assistant_message(
                session_id=session_id,
                content=refusal_text,
                model_info={"model_name": "gemini-3-flash-preview"},
                parent_msg_id=user_message_id
            )
            
            # Log the chunks that were retrieved but deemed irrelevant
            self.context_service.log_used_chunks(
                user_message_id,
                [
                    {
                        "chunk_id": h["id"],
                        "similarity_score": h.get("similarity"),
                        "rank": i + 1,
                        "was_irrelevant": True
                    }
                    for i, h in enumerate(hits)
                ],
            )
            
            return {
                "assistant_message_id": assistant_msg.id,
                "content": refusal_text,
                "sources": hits,  # Still return sources for transparency
                "retrieval_metadata": {"bm25_k": bm25_k, "final_k": final_k, "used_chunks": len(hits)},
                "safety": None,  # No safety for unanswerable
                "xai_explanation": None  # No XAI for unanswerable
            }

        # -----------------------------
        # 7. Log used chunks (for answerable questions only)
        # -----------------------------
        self.context_service.log_used_chunks(
            user_message_id,
            [
                {
                    "chunk_id": h["id"],
                    "similarity_score": h.get("similarity"),
                    "rank": i + 1,
                }
                for i, h in enumerate(hits)
            ],
        )

        # -----------------------------
        # 8. Select prompt type based on intent
        # -----------------------------
        # 🟢 Summary
        if intent == "summary":
            prompt = build_summary_prompt(
                context=context,
                grade_level=grade_level,
                query=user_query
            )
            message_grade_level = grade_level

        # 🟢 Q&A GENERATION (lesson practice)
        elif intent == "qa_generate":
            count = self.extract_question_count(user_query)
            count = max(1, min(count, 10))  # limit range for safety

            prompt = build_qa_prompt(
                context=context,
                count=count,
                query=user_query,
                grade=grade_level
            )
            message_grade_level = grade_level

        # 🟢 DIRECT ANSWER
        elif intent == "qa_answer":
            prompt = build_direct_answer_prompt(
                context=context,
                query=user_query,
                grade=grade_level
            )
            message_grade_level = grade_level

        # 🟡 Explanation fallback
        elif intent == "explanation":
            prompt = build_direct_answer_prompt(
                context=context,
                query=user_query,
                grade=grade_level,
            )
            message_grade_level = grade_level

        # 🔴 Safe fallback
        else:
            prompt = build_direct_answer_prompt(
                context=context,
                query=user_query
            )
            message_grade_level = None

        # -----------------------------
        # 9. Generate response with Gemini
        # -----------------------------
        generated_result = GeminiClient.generate_content(
            prompt=prompt, 
            user_id=user_id,
            session_id=session_id,
            message_id=user_message_id,
            service_name="message_generation",
        )
        generated = generated_result["text"]
        prompt_tokens = generated_result["prompt_tokens"]
        completion_tokens = generated_result["completion_tokens"]
        total_tokens = generated_result["total_tokens"]

        logger.info("Generated response of length %d", len(generated))

        # -----------------------------
        # 9.5. Check for LLM-detected unanswerability
        # -----------------------------
        REFUSAL_MARKER = "[NOT_ANSWERABLE]"
        is_llm_refusal = generated.strip().startswith(REFUSAL_MARKER)

        if is_llm_refusal:
            logger.info("Gemini detected question is unanswerable (marker found)")
            refusal_text = "මෙම ප්‍රශ්නයට අදාළ තොරතුරු ලබා දී ඇති අන්තර්ගතයේ නොමැත."
            
            assistant_msg = self.message_service.create_assistant_message(
                session_id=session_id,
                content=refusal_text,
                model_info={"model_name": "gemini-3-flash-preview"},
                parent_msg_id=user_message_id
            )

            # Log used chunks for metrics even if LLM refused
            self.context_service.log_used_chunks(
                user_message_id,
                [
                    {
                        "chunk_id": h["id"],
                        "similarity_score": h.get("similarity"),
                        "rank": i + 1,
                        "was_irrelevant": True
                    }
                    for i, h in enumerate(hits)
                ],
            )
            
            return {
                "assistant_message_id": assistant_msg.id,
                "content": refusal_text,
                "sources": hits,
                "retrieval_metadata": {"bm25_k": bm25_k, "final_k": final_k, "used_chunks": len(hits)},
                "safety": None,
                "xai_explanation": None
            }

        # -----------------------------
        # 10. Safety & misconception checks (only for answerable questions)
        # -----------------------------
        result = concept_map_check(generated, context)
        missing = result["missing_concepts"]
        extra = result["extra_concepts"]
        is_valid = len(missing) == 0 and len(extra) == 0

        flagged = detect_misconceptions(generated, context)
        logger.info("Detected %d flagged misconceptions", len(flagged))
        flagged = attach_evidence(flagged, context)
        logger.info("Attached evidence to flagged misconceptions")

        # ---- High-level summary ----
        logger.info(
            "Safety check summary | is_valid=%s | missing=%d | extra=%d | flagged=%d",
            is_valid,
            len(missing),
            len(extra),
            len(flagged),
        )

        # ---- Detailed concept logs ----
        if missing:
            logger.info("Missing concepts: %s", list(missing)[:15])

        if extra:
            logger.info("Extra concepts: %s", list(extra)[:15])

        # ---- Detailed misconception logs ----
        for i, f in enumerate(flagged, start=1):
            logger.info(
                "FLAGGED #%d | severity=%s | ratio=%.2f\nSENTENCE: %s\nEVIDENCE: %s",
                i,
                f.get("severity"),
                f.get("unseen_ratio"),
                f.get("sentence"),
                f.get("evidence"),
            )

        logger.info("Saving assistant message...")

        # -----------------------------
        # 11. Save assistant message
        # -----------------------------
        assistant_msg = self.message_service.create_assistant_message(
            session_id=session_id,
            content=generated,
            model_info={
                "model_name": "gemini-3-flash-preview", 
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            },
            grade_level=message_grade_level,
            parent_msg_id=user_message_id
        )

        # -----------------------------
        # 12. Compute summary and save safety report
        # -----------------------------
        from app.services.safety_summary_service import SafetySummaryService
        
        computed_values = SafetySummaryService.compute_from_flagged(flagged, is_unanswerable=False)

        # Build retrieval metadata for XAI generation
        retrieval_metadata = {"bm25_k": bm25_k, "final_k": final_k, "used_chunks": len(hits)}

        # Generate XAI explanation (only for answerable questions)
        xai_explanation = XAIService.generate_explanation(
            user_query=user_query,
            generated_answer=generated,
            retrieved_chunks=hits,
            safety_report={
                "flagged": flagged,
                "missing_concepts": list(missing),
                "extra_concepts": list(extra),
                "confidence_score": computed_values.get("computed_confidence_score", 1.0),
            },
            retrieval_metadata=retrieval_metadata,
        )

        self.safety_service.create_safety_report(
            assistant_msg.id,
            {
                "missing_concepts": list(missing)[:50] if missing else None,
                "extra_concepts": list(extra)[:50] if extra else None,
                "flagged_sentences": flagged if flagged else None,
                "reasoning": "Hybrid RAG with Sinhala QA/Summary",
                **computed_values,
                "xai_explanation": xai_explanation,
            },
        )

        logger.info("Assistant message and safety report saved.")

        # -----------------------------
        # 13. Return full response with all metadata
        # -----------------------------
        return {
            "assistant_message_id": assistant_msg.id,
            "content": generated,
            "sources": hits,
            "retrieval_metadata": retrieval_metadata,
            "safety": {
                "is_valid": is_valid,
                "missing_concepts": list(missing)[:10],
                "extra_concepts": list(extra)[:10],
                "flagged": flagged,
            },
            "xai_explanation": xai_explanation
        }