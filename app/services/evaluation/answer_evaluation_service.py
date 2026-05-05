# app/services/evaluation/answer_evaluation_service.py

# Responsibility wise breakdown:
# create_answer_document	Register uploaded answer
# create_evaluation_result	Store overall result
# create_question_score	Store per-subquestion marks
# create_complete_evaluation	Orchestrates full evaluation
# get_complete_evaluation_result	Read model for UI

from typing import Optional, List, Dict, Tuple
from uuid import UUID
from decimal import Decimal
import re
import logging
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

from app.repositories.evaluation.answer_evaluation_repository import AnswerEvaluationRepository
from app.repositories.evaluation.marking_schema_repository import MarkingSchemaRepository
from app.shared.models.session_resources import SessionResource
from app.shared.models.resource_file import ResourceFile
from app.shared.models.question_papers import Question, QuestionPaper, SubQuestion
from app.shared.models.evaluation_session import EvaluationSession, PaperConfig


class AnswerEvaluationService:
    """Business logic for answer evaluation."""
    
    def __init__(self, db: Session):
        self.repository = AnswerEvaluationRepository(db)
        self.marking_schema_repository = MarkingSchemaRepository(db)
    
    def create_answer_document(
        self,
        evaluation_session_id: UUID,
        resource_id: UUID,
        student_identifier: Optional[str] = None
    ):
        """Register an answer document."""
        return self.repository.create_answer_document(
            evaluation_session_id=evaluation_session_id,
            resource_id=resource_id,
            student_identifier=student_identifier
        )
    
    def get_answer_document(self, answer_document_id: UUID):
        """Get answer document by ID."""
        return self.repository.get_answer_document(answer_document_id)
    
    def get_answer_documents_by_evaluation_session(self, evaluation_session_id: UUID) -> List:
        """Get all answer documents for an evaluation session."""
        return self.repository.get_answer_documents_by_evaluation_session(evaluation_session_id)

    def get_answer_document_by_session_and_resource(self, evaluation_session_id: UUID, resource_id: UUID):
        """Get answer document by session and resource ID."""
        return self.repository.get_answer_document_by_session_and_resource(evaluation_session_id, resource_id)

    def update_mapped_answers(self, answer_document_id: UUID, mapped_answers: Dict):
        """Update mapped answers for an answer document."""
        return self.repository.update_mapped_answers(answer_document_id, mapped_answers)
    
    def create_evaluation_result(
        self,
        answer_document_id: UUID,
        total_score: Optional[Decimal] = None,
        overall_feedback: Optional[str] = None
    ):
        """Create an evaluation result."""
        return self.repository.create_evaluation_result(
            answer_document_id=answer_document_id,
            total_score=total_score,
            overall_feedback=overall_feedback
        )
    
    def get_evaluation_result(self, evaluation_result_id: UUID):
        """Get evaluation result by ID."""
        return self.repository.get_evaluation_result(evaluation_result_id)
    
    def get_evaluation_result_by_answer_document(self, answer_document_id: UUID):
        """Get evaluation result by answer document ID."""
        return self.repository.get_evaluation_result_by_answer_document(answer_document_id)
    
    def create_question_score(
        self,
        evaluation_result_id: UUID,
        sub_question_id: Optional[UUID] = None,
        question_id: Optional[UUID] = None,
        awarded_marks: Decimal = Decimal(0),
        feedback: Optional[str] = None,
        student_answer: Optional[str] = None
    ):
        """Create a score for a question or sub-question."""
        return self.repository.create_question_score(
            evaluation_result_id=evaluation_result_id,
            sub_question_id=sub_question_id,
            question_id=question_id,
            awarded_marks=awarded_marks,
            feedback=feedback,
            student_answer=student_answer
        )

    def create_student_answer(
        self,
        answer_document_id: UUID,
        answer_text: str,
        question_id: Optional[UUID] = None,
        sub_question_id: Optional[UUID] = None
    ):
        """Register a student answer."""
        return self.repository.create_student_answer(
            answer_document_id=answer_document_id,
            answer_text=answer_text,
            question_id=question_id,
            sub_question_id=sub_question_id
        )
    
    def get_question_scores_by_result(self, evaluation_result_id: UUID) -> List:
        """Get all question scores for an evaluation result."""
        return self.repository.get_question_scores_by_result(evaluation_result_id)
    
    def update_evaluation_result(
        self,
        evaluation_result_id: UUID,
        total_score: Optional[Decimal] = None,
        overall_feedback: Optional[str] = None
    ):
        """Update evaluation result."""
        return self.repository.update_evaluation_result(
            evaluation_result_id=evaluation_result_id,
            total_score=total_score,
            overall_feedback=overall_feedback
        )
    
    def create_complete_evaluation(
        self,
        answer_document_id: UUID,
        mapped_answers: List[Dict],  # [{question_id, sub_question_id, student_answer, ...}]
        question_lookup: Dict,      # {sub_question_id: {question_type, correct_answer, max_marks, question_text, resource_ids, ...}}
        rubric_weights: Optional[Dict[str, float]] = None,  # e.g., {"semantic": 0.5, "coverage": 0.3, "bm25": 0.2}
        db=None,
        overall_feedback: Optional[str] = None
    ):
        """
        Evaluate all mapped answers using question type and rubric.
        mapped_answers: list of dicts with keys: sub_question_id, student_answer, etc.
        question_lookup: dict mapping sub_question_id to question info (type, correct_answer, max_marks, ...)
        db: SQLAlchemy session (required for hybrid retrieval)
        """
        from app.services.hybrid_retrieval_service import HybridRetrievalService
        from app.services.semantic_similarity_service import SemanticSimilarityService
        from app.services.embedding_service import EmbeddingService
        scores = []
        for ans in mapped_answers:
            sub_qid = ans.get("sub_question_id")
            student_answer = ans.get("student_answer", "")
            qinfo = question_lookup.get(sub_qid, {})
            qtype = (qinfo.get("question_type") or "essay").lower()
            correct = qinfo.get("correct_answer")
            max_marks = qinfo.get("max_marks", 1)
            question_text = qinfo.get("question_text", "")
            resource_ids = qinfo.get("resource_ids", [])
            feedback = ""
            awarded = 0
            if qtype in ["mcq", "short"]:
                # Case-insensitive, whitespace-trimmed match
                if correct and student_answer.strip().lower() == correct.strip().lower():
                    awarded = max_marks
                    feedback = f"Correct answer. ({awarded}/{max_marks})"
                else:
                    awarded = 0
                    feedback = f"Incorrect answer. ({awarded}/{max_marks})"
            else:
                # Essay/structured: real semantic/coverage/bm25 scoring
                sem_weight = rubric_weights.get("semantic", 0.5) if rubric_weights else 0.5
                cov_weight = rubric_weights.get("coverage", 0.3) if rubric_weights else 0.3
                bm_weight = rubric_weights.get("bm25", 0.2) if rubric_weights else 0.2
                sem_score = cov_score = bm_score = 0.0
                reference_text = ""
                if db and resource_ids and question_text:
                    hybrid_service = HybridRetrievalService(db)
                    embedding = EmbeddingService.embed(question_text)
                    hits = hybrid_service.retrieve(
                        resource_ids=resource_ids,
                        query=question_text,
                        query_embedding=embedding,
                        bm25_k=3,
                        final_k=3
                    )
                    reference_text = " ".join([h["content"] for h in hits if h.get("content")])
                    sem_score = SemanticSimilarityService.similarity(student_answer, reference_text)
                    student_words = set(student_answer.split())
                    ref_words = set(reference_text.split())
                    cov_score = len(student_words & ref_words) / len(ref_words) if ref_words else 0.0
                    bm_score = max([h.get("similarity", 0.0) for h in hits], default=0.0)
                avg_score = sem_score * sem_weight + cov_score * cov_weight + bm_score * bm_weight
                awarded = float(max_marks) * avg_score
                feedback = f"Essay/structured answer scored. ({awarded:.2f}/{max_marks}) (Sim: {sem_score:.2f}, Cov: {cov_score:.2f}, BM25: {bm_score:.2f})"
            scores.append({
                "sub_question_id": sub_qid,
                "awarded_marks": awarded,
                "feedback": feedback
            })
        total_score = sum(Decimal(str(score["awarded_marks"])) for score in scores)
        # Try to get total_marks from question_lookup (sum of all max_marks)
        total_marks = sum(Decimal(str(q.get("max_marks", 0))) for q in question_lookup.values()) if question_lookup else None
        percentage_score = float(total_score) / float(total_marks) * 100 if total_marks and total_marks > 0 else None
        # Compose feedback string
        score_str = f"{float(total_score):.0f}/{float(total_marks):.0f}" if total_marks else f"{float(total_score):.0f}"
        percent_str = f" ({percentage_score:.1f}/100)" if percentage_score is not None else ""
        full_feedback = f"Score: {score_str}{percent_str}. "
        if overall_feedback:
            full_feedback += overall_feedback
        eval_result = self.repository.create_evaluation_result(
            answer_document_id=answer_document_id,
            total_score=total_score,
            overall_feedback=full_feedback
        )
        for score_data in scores:
            self.repository.create_question_score(
                evaluation_result_id=eval_result.id,
                sub_question_id=score_data["sub_question_id"],
                awarded_marks=Decimal(str(score_data["awarded_marks"])),
                feedback=score_data.get("feedback")
            )
        return eval_result
    
    def get_complete_evaluation_result(self, evaluation_result_id: UUID) -> Optional[Dict]:
        """Get complete evaluation result with all question scores."""
        result = self.repository.get_evaluation_result(evaluation_result_id)
        if not result:
            return None

        scores = self.repository.get_question_scores_by_result(evaluation_result_id)
        # Remove duplicates safely. Keep the most meaningful persisted row.
        unique_scores = {}
        for s in scores:
            key = (s.question_id, s.sub_question_id)
            existing = unique_scores.get(key)
            if existing is None or self._score_priority(s) > self._score_priority(existing):
                unique_scores[key] = s

        scores = list(unique_scores.values())
        
        # Sort scores naturally by question number
        def sort_key(score):
            q_num = ""
            if score.question and score.question.question_number:
                q_num = score.question.question_number
            elif score.sub_question:
                parent_num = ""
                if score.sub_question.question and score.sub_question.question.question_number:
                    parent_num = score.sub_question.question.question_number
                q_num = f"{parent_num}{score.sub_question.label}"
            return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', str(q_num))]

        scores.sort(key=sort_key)

        # Calculate grade (A: 85-100, B: 70-84, C: 55-69, D: 40-54, F: <40)
        def calc_grade(score):
            if score is None:
                return None
            try:
                score = float(score)
            except Exception:
                return None
            if score >= 85:
                return "A"
            elif score >= 70:
                return "B"
            elif score >= 55:
                return "C"
            elif score >= 40:
                return "D"
            else:
                return "F"

        # Fetch selection rules for correct "Total Available" summary
        eval_session = self.repository.db.query(EvaluationSession).filter(EvaluationSession.id == result.answer_document.evaluation_session_id).first()
        chat_session_id = eval_session.session_id if eval_session else None
        
        paper_configs = self.repository.db.query(PaperConfig).filter(
            PaperConfig.is_confirmed.is_(True),
            (PaperConfig.evaluation_session_id == result.answer_document.evaluation_session_id) |
            (PaperConfig.chat_session_id == chat_session_id)
        ).all()
        # NORMALIZE keys to paper_i, paper_ii for robust matching
        selection_map = {}
        config_info_by_part = {}
        for c in paper_configs:
            logger.info(
                "[CONFIRMED_PAPER_CONFIG] session=%s source=%s part=%s total_marks=%s total_main_questions=%s selection_rules=%s",
                result.answer_document.evaluation_session_id,
                "evaluation_session" if c.evaluation_session_id == result.answer_document.evaluation_session_id else "chat_session",
                c.paper_part,
                c.total_marks,
                c.total_main_questions,
                c.selection_rules,
            )
            norm_key = c.paper_part.lower().replace(" ", "_").replace("paper_", "paper_") if c.paper_part else ""
            if "i" in norm_key and "ii" not in norm_key: norm_key = "paper_i"
            elif "ii" in norm_key: norm_key = "paper_ii"
            if norm_key:
                config_info_by_part[norm_key] = {
                    "total_marks": float(c.total_marks) if c.total_marks is not None else None,
                    "total_main_questions": c.total_main_questions,
                }
            if c.selection_rules:
                selection_map[norm_key] = c.selection_rules

        # Calculate summary with leaf-node focus to prevent double-display
        total_awarded = 0
        total_max_allowed = 0
        marks_summary = {}
        # Get all unique parts
        parts = sorted(list(set([
            (s.question.part_name or "Other") if s.question else (s.sub_question.question.part_name or "Other") 
            for s in scores if (s.question or (s.sub_question and s.sub_question.question))
        ])))
        
        for part in parts:
            clean_part = part.replace("_", " ")
            marks_summary[clean_part] = []
            
            # Filter scores for this part
            part_scores = [
                s for s in scores 
                if ((s.question.part_name if s.question else s.sub_question.question.part_name) == part)
            ]
            
            # Group by main question for selection logic
            mq_data = {} # {mq_id: {"awarded": x, "max": y}}
            
            for score in part_scores:
                # Leaf Logic: only show in summary if it's a leaf
                is_leaf = True
                main_q = score.question
                if score.question:
                    if score.question.sub_questions: is_leaf = False
                elif score.sub_question:
                    if score.sub_question.children: is_leaf = False
                    main_q = score.sub_question.question
                
                if not is_leaf: continue

                label = ""
                if score.question:
                    label = score.question.question_number
                elif score.sub_question:
                    parent_num = score.sub_question.question.question_number if score.sub_question.question else ""
                    label = f"{parent_num}({score.sub_question.label})"

                awarded = float(score.awarded_marks or 0)
                max_m = float(score.question.max_marks if score.question else (score.sub_question.max_marks or 0))

                # Compute mq_id BEFORE using it in the entry
                mq_id = str(main_q.id) if main_q else "orphan"

                marks_summary[clean_part].append({
                    "label": label,
                    "awarded": round(awarded, 2),
                    "max": max_m,
                    "_mq_id": mq_id  # internal field used for selection tagging below
                })

                # Track for Total Available logic (Selection Aware)
                if mq_id not in mq_data:
                    mq_data[mq_id] = {
                        "awarded": 0,
                        "max": 0,
                        "attempted": 0,
                        "label": main_q.question_number if main_q else label,
                    }
                mq_data[mq_id]["awarded"] += awarded
                mq_data[mq_id]["max"] += max_m
                if awarded > 0:
                    mq_data[mq_id]["attempted"] += 1



            # Apply selection rules to calculate REAL Part Total (awarded AND max)
            norm_part = part.lower().replace(" ", "_")
            if norm_part.endswith("_ii") or norm_part == "paper_ii":
                norm_part = "paper_ii"
            elif norm_part.endswith("_i") or norm_part == "paper_i":
                norm_part = "paper_i"

            rules = selection_map.get(norm_part) or {}
            mode = str(rules.get('mode', 'all')).lower()
            # Support both 'any' and 'choose_any' as synonyms
            is_selection_mode = mode in ['any', 'choose_any', 'choose']
            required_count = rules.get('count') or rules.get('total') or rules.get('choose_any')

            selected_mq_ids = set(mq_data.keys())  # default: all selected
            selected_entries = list(mq_data.items())
            part_full_max = sum(v["max"] for v in mq_data.values())
            configured_part_total = (config_info_by_part.get(norm_part) or {}).get("total_marks")
            known_config_total = sum(
                info["total_marks"]
                for info in config_info_by_part.values()
                if info.get("total_marks") is not None
            )
            session_total_marks = getattr(eval_session, "total_marks", None)
            overall_total = float(session_total_marks or 0) if session_total_marks else 0
            inferred_part_total = None
            if configured_part_total and configured_part_total > 0:
                inferred_part_total = configured_part_total
            elif overall_total > known_config_total and norm_part in config_info_by_part:
                missing_parts = [
                    part_key
                    for part_key, info in config_info_by_part.items()
                    if info.get("total_marks") is None
                ]
                if len(missing_parts) == 1 and missing_parts[0] == norm_part:
                    inferred_part_total = overall_total - known_config_total

            if (not is_selection_mode) and inferred_part_total and 0 < inferred_part_total < part_full_max:
                main_max_values = [v["max"] for v in mq_data.values() if v["max"] > 0]
                typical_main_max = max(set(main_max_values), key=main_max_values.count) if main_max_values else 0
                if typical_main_max and inferred_part_total % typical_main_max == 0:
                    inferred_count = int(inferred_part_total / typical_main_max)
                    if 0 < inferred_count < len(mq_data):
                        mode = "any"
                        required_count = inferred_count
                        is_selection_mode = True
                        logger.info(
                            "[INFERRED_SELECTION_RULES] part=%s inferred_total=%s typical_main_max=%s mode=any count=%s",
                            norm_part,
                            inferred_part_total,
                            typical_main_max,
                            inferred_count,
                        )
            elif (not is_selection_mode) and norm_part != "paper_i" and norm_part in config_info_by_part:
                attempted_count = sum(1 for v in mq_data.values() if v.get("attempted", 0) > 0)
                has_unattempted = any(v.get("attempted", 0) <= 0 for v in mq_data.values())
                if attempted_count > 0 and has_unattempted:
                    mode = "attempted_only"
                    required_count = attempted_count
                    is_selection_mode = True
                    logger.info(
                        "[INFERRED_SELECTION_RULES] part=%s mode=attempted_only count=%s reason=missing_config_selection",
                        norm_part,
                        attempted_count,
                    )

            if is_selection_mode and required_count and len(mq_data) > int(required_count):
                # SYNC: Sort by (awarded DESC, attempted DESC) to match GradingService
                sorted_entries = sorted(
                    mq_data.items(),
                    key=lambda x: (x[1]["awarded"], x[1].get("attempted", 0)),
                    reverse=True
                )
                selected_entries = sorted_entries[:int(required_count)]
                selected_mq_ids = {e[0] for e in selected_entries}
                part_max_allowed = sum(e[1]["max"] for e in selected_entries)
                part_awarded = sum(e[1]["awarded"] for e in selected_entries)
            else:
                part_max_allowed = sum(v["max"] for v in mq_data.values())
                part_awarded = sum(v["awarded"] for v in mq_data.values())

            total_max_allowed += part_max_allowed
            total_awarded += part_awarded

            logger.info(
                "[RESULT_SELECTION] part=%s mode=%s required_count=%s main_questions=%s selected=%s part_awarded=%s part_max=%s",
                norm_part,
                mode,
                required_count,
                len(mq_data),
                [
                    {
                        "label": data.get("label"),
                        "awarded": round(data.get("awarded", 0), 2),
                        "max": round(data.get("max", 0), 2),
                        "attempted": data.get("attempted", 0),
                    }
                    for _, data in selected_entries
                ],
                round(part_awarded, 2),
                round(part_max_allowed, 2),
            )

            # Tag each score entry with is_selected so the frontend can highlight them
            for entry in marks_summary[clean_part]:
                entry["is_selected"] = entry.get("_mq_id") in selected_mq_ids
                entry["counts_toward_total"] = entry["is_selected"]
                entry["awarded_for_total"] = entry["awarded"] if entry["counts_toward_total"] else 0
                entry["max_for_total"] = entry["max"] if entry["counts_toward_total"] else 0
                entry["part_total_awarded"] = round(part_awarded, 2)
                entry["part_total_max"] = round(part_max_allowed, 2)


        
        # Override result.total_score if needed? 
        # No, result.total_score is already calculated best-of-N in GradingService.
        # We just need the UI to show the correct /Total.
        
        summary_total_max_allowed = total_max_allowed
        logger.debug(f"Generated marks summary. Total Available: {total_max_allowed}")


        # Extract missed concepts and improvement points from question feedbacks
        improvement_points = []
        seen_points = set()
        
        for score in scores:
            fb = score.feedback or ""
            # Combine both missing concepts and improvement points
            keywords = ["missing", "should mention", "need to mention", "not mentioned", "should include", "not included", "අඩංගු විය යුතුය", "සඳහන් කළ යුතුය", "පැහැදිලි කළ යුතුය", "improve", "clarify", "writing style", "description", "elaborate", "expand", "unclear", "not clear", "not detailed", "කෙටි", "විස්තර", "සවිස්තර"]
            
            if any(kw in fb.lower() for kw in keywords):
                for sent in re.split(r'[.\n]', fb):
                    sent_stripped = sent.strip()
                    if not sent_stripped:
                        continue
                    
                    if any(kw in sent_stripped.lower() for kw in keywords):
                        # Simple deduplication based on exact text match (case-insensitive)
                        lower_sent = sent_stripped.lower()
                        if lower_sent not in seen_points:
                            seen_points.add(lower_sent)
                            improvement_points.append(sent_stripped)
        # Add generic suggestions if none found
        if not improvement_points:
            improvement_points.append("Improve writing style and concept explanation.")

        feedback_obj = {
            "overall_feedback": result.overall_feedback,
            "improvement_points": improvement_points
        }

        total_max_allowed = self._resolve_total_max(
            evaluation_session_id=result.answer_document.evaluation_session_id,
            chat_session_id=chat_session_id,
            selection_map=selection_map,
        )
        if summary_total_max_allowed and (
            not total_max_allowed or summary_total_max_allowed < total_max_allowed
        ):
            logger.info(
                "[TOTAL_MAX_SOURCE] source=selection_aware_summary total=%s resolved_total=%s",
                summary_total_max_allowed,
                total_max_allowed,
            )
            total_max_allowed = round(summary_total_max_allowed, 2)
        percentage_score = None
        if total_max_allowed and total_max_allowed > 0:
            percentage_score = round((float(result.total_score or 0) / float(total_max_allowed)) * 100, 2)

        return {
            "id": result.id,
            "answer_document_id": result.answer_document_id,
            "total_score": round(float(result.total_score or 0), 2),
            "total_max": total_max_allowed,
            "percentage_score": percentage_score,
            "grade": calc_grade(result.total_score),
            "overall_feedback": result.overall_feedback,
            "evaluated_at": result.evaluated_at.isoformat() if result.evaluated_at else None,
            "marks_summary": marks_summary,
            "improvement_points": list(set(improvement_points))[:5],
            "question_feedbacks": [
                self._serialize_question_feedback(score)
                for score in scores
                if self._is_leaf_score(score)
            ],
            "marking_schema": self._serialize_marking_schema_for_session(
                result.answer_document.evaluation_session_id
            ),
            "answer_mapping": self._serialize_answer_mapping_for_document(
                result.answer_document
            ),
        }

    def _serialize_marking_schema_for_session(self, evaluation_session_id: UUID) -> Optional[Dict]:
        schema = self.marking_schema_repository.get_marking_schema_by_session(evaluation_session_id)
        if not schema:
            return None

        items = self.marking_schema_repository.get_marking_schema_items(schema.id)
        serialized_items = [
            {
                "id": str(item.id),
                "question_id": str(item.question_id) if item.question_id else None,
                "question_number": item.question_number,
                "question_text": item.question_text,
                "reference_text": item.reference_text,
                "max_marks": float(item.max_marks) if item.max_marks is not None else None,
                "part_name": item.part_name,
                "part_display": self._format_paper_part(item.part_name),
                "sort_order": item.sort_order,
            }
            for item in items
        ]

        return {
            "id": str(schema.id),
            "evaluation_session_id": str(schema.evaluation_session_id),
            "resource_id": str(schema.resource_id) if schema.resource_id else None,
            "is_confirmed": schema.is_confirmed,
            "created_at": schema.created_at.isoformat() if schema.created_at else None,
            "updated_at": schema.updated_at.isoformat() if schema.updated_at else None,
            "items": serialized_items,
        }

    def _serialize_answer_mapping_for_document(self, answer_doc) -> Dict:
        mapped_answers = answer_doc.mapped_answers or {}
        eval_session = self.repository.db.query(EvaluationSession).filter(
            EvaluationSession.id == answer_doc.evaluation_session_id
        ).first()

        question_lookup = {}
        if eval_session:
            question_papers = (
                self.repository.db.query(QuestionPaper)
                .filter(
                    (QuestionPaper.evaluation_session_id == eval_session.id) |
                    (QuestionPaper.chat_session_id == eval_session.session_id)
                )
                .all()
            )
            paper_ids = [paper.id for paper in question_papers]
            questions = (
                self.repository.db.query(Question)
                .filter(Question.question_paper_id.in_(paper_ids))
                .all()
                if paper_ids else []
            )

            for question in questions:
                part_name = getattr(question, "part_name", None)
                question_number = str(question.question_number or "").strip()
                question_lookup[str(question.id)] = {
                    "type": "question",
                    "paper_part": part_name,
                    "question_number": question_number,
                    "display_label": self._format_mapping_display(part_name, question_number),
                    "question_text": question.question_text,
                    "max_marks": question.max_marks,
                    "_sort": self._mapping_sort_key(part_name, question_number),
                }
                self._add_sub_question_mapping_lookup(question_lookup, question, question.sub_questions or [])

        details = []
        for question_id, answer_text in mapped_answers.items():
            info = question_lookup.get(str(question_id), {})
            details.append({
                "question_id": str(question_id),
                "display_label": info.get("display_label", str(question_id)),
                "paper_part": info.get("paper_part"),
                "question_number": info.get("question_number"),
                "question_text": info.get("question_text"),
                "student_answer": answer_text,
                "max_marks": float(info.get("max_marks")) if info.get("max_marks") is not None else None,
                "is_mapped_to_current_question": bool(info),
                "_sort": info.get("_sort", ((99, ""), (10**9, str(question_id)))),
            })

        details.sort(key=lambda item: (item.get("_sort"), item.get("display_label") or ""))
        for item in details:
            item.pop("_sort", None)

        return {
            "answer_document_id": str(answer_doc.id),
            "details": details,
            "raw_mapping": {str(key): value for key, value in mapped_answers.items()},
        }

    def _add_sub_question_mapping_lookup(self, lookup: Dict, question: Question, sub_questions: List[SubQuestion], parent_number: Optional[str] = None):
        base_number = parent_number or str(question.question_number or "").strip()
        part_name = getattr(question, "part_name", None)
        for sub_question in sub_questions:
            label = str(sub_question.label or "").strip()
            number = f"{base_number}({label})" if base_number and label else (label or base_number)
            lookup[str(sub_question.id)] = {
                "type": "sub_question",
                "paper_part": part_name,
                "question_number": number,
                "display_label": self._format_mapping_display(part_name, number),
                "question_text": sub_question.sub_question_text,
                "max_marks": sub_question.max_marks,
                "_sort": self._mapping_sort_key(part_name, number),
            }
            self._add_sub_question_mapping_lookup(lookup, question, sub_question.children or [], number)

    def _format_mapping_display(self, part_name: Optional[str], question_number: Optional[str]) -> str:
        part_display = self._format_paper_part(part_name)
        if part_display and question_number:
            return f"{part_display} - {question_number}"
        return question_number or part_display or "Unknown"

    def _mapping_sort_key(self, part_name: Optional[str], question_number: Optional[str]):
        part = self._normalize_part_key(part_name)
        part_order = {"paper_i": 1, "paper_ii": 2, "paper_iii": 3, "paper_iv": 4}.get(part, 99)
        number = str(question_number or "")
        pieces = [
            (0, int(piece)) if piece.isdigit() else (1, piece.lower())
            for piece in re.split(r"(\d+)", number)
            if piece
        ]
        return (part_order, pieces)

    def _score_priority(self, score) -> Tuple[int, float]:
        has_student_answer = 1 if getattr(score, "student_answer", None) and str(score.student_answer).strip() else 0
        has_feedback = 1 if getattr(score, "feedback", None) and str(score.feedback).strip() else 0
        awarded = float(score.awarded_marks or 0)
        return (has_student_answer, has_feedback, awarded)

    def _serialize_question_feedback(self, score) -> Dict:
        question = score.question or (score.sub_question.question if score.sub_question else None)
        question_number = str(question.question_number).strip() if question and question.question_number else None
        sub_label = str(score.sub_question.label).strip() if score.sub_question and score.sub_question.label else None
        paper_part = question.part_name if question and question.part_name else None
        is_leaf = self._is_leaf_score(score)
        max_marks = question.max_marks if score.question else (score.sub_question.max_marks if score.sub_question else None)

        return {
            "id": score.id,
            "evaluation_result_id": score.evaluation_result_id,
            "question_id": score.question_id,
            "sub_question_id": score.sub_question_id,
            "question_label": self._build_question_label(question_number, score.sub_question),
            "main_question_key": question_number,
            "sub_label": sub_label.lower() if sub_label else None,
            "paper_part": paper_part,
            "paper_part_display": self._format_paper_part(paper_part),
            "is_leaf": is_leaf,
            "awarded_marks": score.awarded_marks,
            "max_marks": Decimal(str(max_marks)) if max_marks is not None else None,
            "feedback": score.feedback,
        }

    def _build_question_label(self, question_number: Optional[str], sub_question) -> str:
        if sub_question and question_number:
            return f"{question_number}({sub_question.label})"
        if sub_question and sub_question.label:
            return str(sub_question.label)
        return question_number or "Unknown"

    def _format_paper_part(self, part_name: Optional[str]) -> Optional[str]:
        if not part_name:
            return None
        return str(part_name).replace("_", " ").strip()

    def _is_leaf_score(self, score) -> bool:
        if score.question:
            return not bool(score.question.sub_questions)
        if score.sub_question:
            return not bool(score.sub_question.children)
        return True

    def _resolve_total_max(
        self,
        evaluation_session_id: UUID,
        chat_session_id: Optional[UUID],
        selection_map: Dict[str, Dict],
    ) -> float:
        config_totals_by_part = self._resolve_paper_config_totals_by_part(evaluation_session_id, chat_session_id)
        configured_total = round(sum(config_totals_by_part.values()), 2) if config_totals_by_part else None
        schema_total = self._resolve_total_max_from_marking_schema(evaluation_session_id, selection_map)
        schema_parts = self._resolve_marking_schema_part_keys(evaluation_session_id)

        if configured_total is not None and configured_total > 0:
            eval_session = self.repository.db.query(EvaluationSession).filter(EvaluationSession.id == evaluation_session_id).first()
            session_total_marks = getattr(eval_session, "total_marks", None)
            overall_total = float(session_total_marks or 0) if session_total_marks else 0
            if overall_total > configured_total and schema_parts:
                missing_parts = schema_parts - set(config_totals_by_part)
                if len(missing_parts) == 1:
                    missing_part = next(iter(missing_parts))
                    inferred_total = round(overall_total - configured_total, 2)
                    if inferred_total > 0:
                        config_totals_by_part[missing_part] = inferred_total
                        configured_total = round(sum(config_totals_by_part.values()), 2)
                        logger.info(
                            "[INFERRED_PAPER_CONFIG_TOTAL] part=%s inferred_total=%s overall_total=%s known_parts=%s",
                            missing_part,
                            inferred_total,
                            overall_total,
                            config_totals_by_part,
                        )
            if not schema_parts or schema_parts.issubset(set(config_totals_by_part)):
                logger.info(
                    "[TOTAL_MAX_SOURCE] source=confirmed_paper_config total=%s parts=%s schema_parts=%s",
                    configured_total,
                    config_totals_by_part,
                    sorted(schema_parts),
                )
                return configured_total

        candidates = [
            value for value in (schema_total, configured_total)
            if value is not None and value > 0
        ]
        if candidates:
            logger.info(
                "[TOTAL_MAX_SOURCE] source=%s total=%s configured_total=%s schema_total=%s selection_map=%s",
                "confirmed_marking_schema" if candidates[0] == schema_total else "confirmed_paper_config",
                candidates[0],
                configured_total,
                schema_total,
                selection_map,
            )
            return candidates[0]

        return 0.0

    def _resolve_total_max_from_paper_config(
        self,
        evaluation_session_id: UUID,
        chat_session_id: Optional[UUID],
    ) -> Optional[float]:
        by_part = self._resolve_paper_config_totals_by_part(evaluation_session_id, chat_session_id)
        if not by_part:
            return None
        return round(sum(by_part.values()), 2)

    def _resolve_paper_config_totals_by_part(
        self,
        evaluation_session_id: UUID,
        chat_session_id: Optional[UUID],
    ) -> Dict[str, float]:
        configs = self.repository.db.query(PaperConfig).filter(
            PaperConfig.is_confirmed.is_(True),
            (PaperConfig.evaluation_session_id == evaluation_session_id) |
            (PaperConfig.chat_session_id == chat_session_id)
        ).all()

        by_part: Dict[str, float] = {}
        for cfg in configs:
            if cfg.total_marks is None:
                continue
            part_key = self._normalize_part_key(cfg.paper_part)
            current = by_part.get(part_key)
            cfg_total = float(cfg.total_marks)

            # Prefer evaluation-session specific totals over inherited chat-session values.
            if current is None or cfg.evaluation_session_id == evaluation_session_id:
                by_part[part_key] = cfg_total

        if not by_part:
            return {}
        return by_part

    def _resolve_marking_schema_part_keys(self, evaluation_session_id: UUID) -> set[str]:
        schema = self.marking_schema_repository.get_marking_schema_by_session(evaluation_session_id)
        if not schema:
            return set()
        items = self.marking_schema_repository.get_marking_schema_items(schema.id)
        return {
            self._normalize_part_key(item.part_name)
            for item in items
            if item.part_name
        }

    def _resolve_total_max_from_marking_schema(
        self,
        evaluation_session_id: UUID,
        selection_map: Dict[str, Dict],
    ) -> Optional[float]:
        schema = self.marking_schema_repository.get_marking_schema_by_session(evaluation_session_id)
        if not schema:
            return None

        items = self.marking_schema_repository.get_marking_schema_items(schema.id)
        if not items:
            return None

        part_totals: Dict[str, Dict[str, float]] = {}
        for item in items:
            part_name = self._normalize_part_key(item.part_name)
            main_key = self._extract_main_question_key(item.question_number)
            max_marks = float(item.max_marks or 0)

            if part_name not in part_totals:
                part_totals[part_name] = {}
            if main_key not in part_totals[part_name]:
                part_totals[part_name][main_key] = 0.0

            part_totals[part_name][main_key] += max_marks

        total_max = 0.0
        for part_name, main_questions in part_totals.items():
            part_rules = selection_map.get(part_name) or {}
            mode = str(part_rules.get("mode", "all")).lower()
            is_selection_mode = mode in ["any", "choose_any", "choose"]
            count = part_rules.get("count") or part_rules.get("total") or part_rules.get("choose_any")

            main_values = sorted(main_questions.values(), reverse=True)
            if is_selection_mode and count and len(main_values) > int(count):
                total_max += sum(main_values[:int(count)])
            else:
                total_max += sum(main_values)

        return round(total_max, 2)

    def _normalize_part_key(self, part_name: Optional[str]) -> str:
        part = (part_name or "other").lower().replace(" ", "_")
        if re.search(r"(_iv|iv)$", part):
            return "paper_iv"
        if re.search(r"(_iii|iii)$", part):
            return "paper_iii"
        if re.search(r"(_ii|ii)$", part):
            return "paper_ii"
        if re.search(r"(_i|i)$", part):
            return "paper_i"
        return part

    def _extract_main_question_key(self, question_number: Optional[str]) -> str:
        text = str(question_number or "").strip()
        match = re.match(r"^(\d+)", text)
        if match:
            return match.group(1)
        return text or "unknown"

    def _load_rubric_text(self, eval_session):
        res = (
            self.repository.db.query(SessionResource)
            .filter(
                SessionResource.session_id == eval_session.session_id,
                SessionResource.label == "rubric",
            )
            .first()
        )
        if res:
            rf = self.repository.db.query(ResourceFile).filter(ResourceFile.id == res.resource_id).first()
            return rf.extracted_text if rf else ""
        return ""

    def _load_syllabus_text(self, eval_session):
        res = (
            self.repository.db.query(SessionResource)
            .filter(
                SessionResource.session_id == eval_session.session_id,
                SessionResource.label == "syllabus",
            )
            .first()
        )
        if res:
            rf = self.repository.db.query(ResourceFile).filter(ResourceFile.id == res.resource_id).first()
            return rf.extracted_text if rf else ""
        return ""
