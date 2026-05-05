#app/services/evaluation/grading_service.py

import logging
import re
import json
import time
from math import floor
from hashlib import sha256
from uuid import UUID
from typing import Dict, Any, List, Tuple, Optional, Callable
from decimal import Decimal, ROUND_HALF_UP
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
from rank_bm25 import BM25Okapi
from sqlalchemy.orm import Session
from sqlalchemy import func
from sqlalchemy.orm.exc import StaleDataError

from app.shared.models.answer_evaluation import (
    AnswerDocument,
    EvaluationResult,
    QuestionScore,
    MarkingReference,
)
from app.shared.models.question_papers import Question, SubQuestion
from app.shared.models.session_resources import SessionResource
from app.shared.models.evaluation_session import EvaluationSession, MarkingSchema, MarkingSchemaItem, PaperConfig
from app.shared.models.resource_file import ResourceFile
from app.shared.models.rubrics import Rubric, RubricCriterion
from app.shared.ai.embeddings import get_xlmr_model, ml_semaphore, ensure_sentences_cached, _embedding_cache
from app.shared.ai.gemini_client import gemini_generate_evaluation
from app.core.config import settings
from app.core.gemini_client import GeminiClient
from app.services.evaluation.gemini_cost_policy import EvaluationGeminiClient

logger = logging.getLogger(__name__)


def _cosine_util():
    from sentence_transformers import util

    return util


class GradingService:
    # Class-level cache: Gemini-extracted references, keyed by eval_session_id.
    # Survives across instances so multiple students in the same session
    # don't trigger redundant Gemini calls.
    SCHEMA_REFERENCE_FORMAT_VERSION = "v7-budgeted-reference-extraction"
    _gemini_ref_cache: Dict[str, Dict[str, str]] = {}
    _gemini_ref_cache_lock = __import__('threading').Lock()

    @classmethod
    def clear_reference_cache(cls, eval_session_id: UUID) -> None:
        """Drop cached generated references for a session after user edits/deletes schema."""
        session_prefix = f"{eval_session_id}:"
        with cls._gemini_ref_cache_lock:
            stale_keys = [
                cache_key
                for cache_key in cls._gemini_ref_cache
                if cache_key == str(eval_session_id) or cache_key.startswith(session_prefix)
            ]
            for cache_key in stale_keys:
                del cls._gemini_ref_cache[cache_key]

    def __init__(self, db: Session):
        self.db = db
        self.gemini = GeminiClient()
        from app.repositories.evaluation.marking_reference_repository import MarkingReferenceRepository
        self.marking_refs = MarkingReferenceRepository(db)


    # ----------------------------------------------------------
    # MARK RESOLUTION
    # ----------------------------------------------------------
    def _resolve_max_marks(self, question, confirmed_marks: Optional[Dict[str, int]] = None):
        confirmed_marks = confirmed_marks or {}
        question_id = str(getattr(question, "id", "") or "")
        if question_id in confirmed_marks:
            return confirmed_marks[question_id]

        parsed_marks = self._extract_marks_from_question_text(
            getattr(question, "question_text", "") or getattr(question, "sub_question_text", "")
        )
        if parsed_marks:
            return parsed_marks

        part_name = self._resolve_part_name(question)
        if self._is_paper_i_part(part_name):
            try:
                parsed_value = int(getattr(question, "max_marks", 0) or 0)
            except (TypeError, ValueError):
                parsed_value = 0
            if parsed_value in {1, 2}:
                return parsed_value
            return 2

        if hasattr(question, "max_marks") and question.max_marks:
            return int(question.max_marks)

        if isinstance(question, SubQuestion):
            parent = question.parent_question
            if parent and parent.sub_questions:
                explicit = [sq.max_marks for sq in parent.sub_questions if sq.max_marks]
                if explicit:
                    return int(question.max_marks or 1)
                return max(1, int(parent.max_marks) // len(parent.sub_questions))

        return 1

    def _normalize_part_key(self, part_name: str) -> str:
        normalized = str(part_name or "").strip().lower().replace(" ", "_")
        if re.search(r'(_iv|iv)$', normalized):
            return "paper_iv"
        if re.search(r'(_iii|iii)$', normalized):
            return "paper_iii"
        if re.search(r'(_ii|ii)$', normalized):
            return "paper_ii"
        if re.search(r'(_i|i)$', normalized):
            return "paper_i"
        return normalized

    def _iter_leaf_targets(self, questions: List[Question]):
        def iter_subs(sub_questions):
            for sub_question in sub_questions or []:
                children = getattr(sub_question, "children", []) or []
                if children:
                    yield from iter_subs(children)
                else:
                    yield sub_question

        for question in questions or []:
            sub_questions = getattr(question, "sub_questions", []) or []
            if sub_questions:
                yield from iter_subs(sub_questions)
            else:
                yield question

    def _question_roots_from_map(self, question_map: Dict) -> List[Question]:
        roots: Dict[str, Question] = {}
        for target in {id(value): value for value in (question_map or {}).values()}.values():
            if isinstance(target, Question):
                roots[str(target.id)] = target
            elif isinstance(target, SubQuestion) and getattr(target, "question", None):
                roots[str(target.question.id)] = target.question
        return list(roots.values())

    def _load_confirmed_mark_overrides(
        self,
        eval_session_id: UUID,
        chat_session_id: Optional[UUID],
        questions: List[Question],
    ) -> Dict[str, int]:
        """
        Build max-mark overrides from confirmed user-facing configuration.
        Priority:
        1. Confirmed marking-schema item max_marks, because those are per question.
        2. Confirmed PaperConfig total_marks distributed across leaf targets in that part.
        """
        overrides: Dict[str, int] = {}

        confirmed_schema = (
            self.db.query(MarkingSchema)
            .filter(
                MarkingSchema.evaluation_session_id == eval_session_id,
                MarkingSchema.is_confirmed.is_(True),
            )
            .first()
        )
        if confirmed_schema:
            schema_items = (
                self.db.query(MarkingSchemaItem)
                .filter(MarkingSchemaItem.marking_schema_id == confirmed_schema.id)
                .all()
            )
            for item in schema_items:
                try:
                    mark = int(item.max_marks) if item.max_marks is not None else None
                except (TypeError, ValueError):
                    mark = None
                if mark and mark > 0 and item.question_id:
                    overrides[str(item.question_id)] = mark

        config_query = self.db.query(PaperConfig).filter(PaperConfig.is_confirmed.is_(True))
        if chat_session_id:
            config_query = config_query.filter(
                (PaperConfig.evaluation_session_id == eval_session_id)
                | (PaperConfig.chat_session_id == chat_session_id)
            )
        else:
            config_query = config_query.filter(PaperConfig.evaluation_session_id == eval_session_id)

        confirmed_configs = config_query.all()
        if not confirmed_configs:
            return overrides

        configs_by_part: Dict[str, PaperConfig] = {}
        for config in confirmed_configs:
            part_key = self._normalize_part_key(config.paper_part)
            if not part_key:
                continue
            existing = configs_by_part.get(part_key)
            if existing is None or (
                existing.evaluation_session_id is None and config.evaluation_session_id == eval_session_id
            ):
                configs_by_part[part_key] = config

        leaf_targets = list(self._iter_leaf_targets(questions))
        leaf_targets_by_part: Dict[str, List[Any]] = {}
        for target in leaf_targets:
            part_key = self._normalize_part_key(self._resolve_part_name(target))
            leaf_targets_by_part.setdefault(part_key, []).append(target)

        for part_key, config in configs_by_part.items():
            try:
                total_marks = int(config.total_marks) if config.total_marks is not None else None
            except (TypeError, ValueError):
                total_marks = None
            targets = leaf_targets_by_part.get(part_key) or []
            if not total_marks or total_marks <= 0 or not targets:
                continue
            if total_marks % len(targets) != 0:
                continue
            per_target_marks = total_marks // len(targets)
            if per_target_marks <= 0:
                continue
            existing_part_marks = [
                overrides.get(str(getattr(target, "id", "") or ""))
                for target in targets
            ]
            existing_part_marks = [mark for mark in existing_part_marks if mark is not None]
            should_apply_config_marks = (
                len(existing_part_marks) != len(targets)
                or sum(existing_part_marks) != total_marks
            )
            if not should_apply_config_marks:
                continue
            for target in targets:
                target_id = str(getattr(target, "id", "") or "")
                if target_id:
                    overrides[target_id] = per_target_marks

        return overrides

    def _extract_marks_from_question_text(self, text: str) -> Optional[int]:
        patterns = [
            r"\(\s*0?(\d{1,2})\s*(?:ලකුණු|marks|à¶½à¶šà·”à¶«à·”)",
            r"(?:ලකුණු|marks|à¶½à¶šà·”à¶«à·”)\s*0?(\d{1,2})",
        ]
        for pattern in patterns:
            match = re.search(pattern, str(text or ""), flags=re.IGNORECASE)
            if not match:
                continue
            try:
                mark = int(match.group(1))
            except (TypeError, ValueError):
                continue
            if mark > 0:
                return mark
        return None

    def _is_paper_i_part(self, part_name: str) -> bool:
        normalized = str(part_name or "").strip().lower().replace(" ", "_")
        return normalized in {"paper_i", "part_i", "i"} or "paper_i" in normalized or "part_i" in normalized

    def _resolve_part_name(self, target) -> str:
        """
        Correctly resolve the part name for any question or sub-question.
        Sub-questions do not have part_name directly, so we traverse the parent chain.
        """
        part = getattr(target, 'part_name', None)
        if part:
            return part

        if hasattr(target, 'question') and target.question:
            part = getattr(target.question, 'part_name', None)
            if part:
                return part

        curr = target
        while hasattr(curr, 'parent') and curr.parent:
            curr = curr.parent
            if hasattr(curr, 'question') and curr.question:
                part = getattr(curr.question, 'part_name', None)
                if part:
                    return part

        return 'Unknown'

    def _iter_all_subquestions(self, sub_questions: List[SubQuestion]):
        for sub_question in sub_questions or []:
            yield sub_question
            children = getattr(sub_question, "children", []) or []
            if children:
                yield from self._iter_all_subquestions(children)

    def _has_any_mapped_descendant(self, question: Question, mapped_answers: Dict[str, Any]) -> bool:
        all_subquestions = list(self._iter_all_subquestions(getattr(question, "sub_questions", []) or []))
        if not all_subquestions:
            return False

        normalized_keys = {str(k).lower().replace(" ", "").replace(".", "").replace("(", "").replace(")", "") for k in mapped_answers.keys()}
        question_number = str(question.question_number or "")

        for sub_question in all_subquestions:
            sq_id = str(sub_question.id)
            sq_label = str(sub_question.label or "").strip()
            composite_key = f"{question_number}{sq_label}".lower().replace(" ", "").replace(".", "").replace("(", "").replace(")", "")

            if sq_id in mapped_answers or composite_key in normalized_keys:
                return True

        return False


    # ----------------------------------------------------------
    # PUBLIC ENTRY
    # ----------------------------------------------------------
    def grade_answer_document(
        self,
        answer_doc_id: UUID,
        user_id: UUID,
        eval_session_id: UUID,
        syllabus_text: str,
        rubric_text: str,
        question_map: Dict,
        reference_map: Optional[Dict[str, str]] = None,
        progress_callback: Optional[Callable[[str, str, Optional[int]], None]] = None,
        include_feedback: bool = True,
    ) -> EvaluationResult:
        answer_doc = (
            self.db.query(AnswerDocument)
            .filter(AnswerDocument.id == answer_doc_id)
            .first()
        )
        if not answer_doc or not answer_doc.mapped_answers:
            raise ValueError("නිරවද්‍ය ලෙස පිළිතුරු හඳුනා ගැනීමට නොහැකි විය. කරුණාකර නැවත උත්සාහ කරන්න. (AI Mapping/Extraction Failure)")

        if progress_callback:
            progress_callback("initializing_grading", "Preparing grading context...")

        eval_result = (
            self.db.query(EvaluationResult)
            .filter(EvaluationResult.answer_document_id == answer_doc.id)
            .order_by(EvaluationResult.evaluated_at.desc())
            .first()
        )

        if eval_result:
            logger.info(f"Re-using existing evaluation result {eval_result.id} for doc {answer_doc.id}")
            self.db.query(QuestionScore).filter(
                QuestionScore.evaluation_result_id == eval_result.id
            ).delete()
            eval_result.total_score = Decimal(0)
            eval_result.overall_feedback = "Grading in progress..."
            eval_result.evaluated_at = func.now()
        else:
            eval_result = EvaluationResult(
                answer_document_id=answer_doc.id,
                total_score=Decimal(0),
                overall_feedback="Grading in progress...",
            )
            self.db.add(eval_result)

        self.db.commit()
        self.db.refresh(eval_result)

        eval_session = self.db.query(EvaluationSession).filter(EvaluationSession.id == eval_session_id).first()
        rubric_weights = self._get_rubric_weights(eval_session)
        chat_session_id = eval_session.session_id if eval_session else None
        confirmed_mark_overrides = self._load_confirmed_mark_overrides(
            eval_session_id=eval_session_id,
            chat_session_id=chat_session_id,
            questions=self._question_roots_from_map(question_map),
        )
        if confirmed_mark_overrides:
            logger.info(
                "Loaded %s confirmed max-mark override(s) for evaluation session %s.",
                len(confirmed_mark_overrides),
                eval_session_id,
            )

        total_questions = len(answer_doc.mapped_answers)
        processed_count = 0

        if progress_callback:
            progress_callback("evaluating_answers", "Scoring persisted mapped answers...", percent=2)
            progress_callback("evaluating_answers", "Using confirmed schema references...", percent=3)

        reference_map = reference_map or {}
        if not reference_map:
            raise ValueError("Marking schema must be confirmed before grading")

        gemini_ref_map = {}
        missing_reference_keys: List[str] = []
        for key in answer_doc.mapped_answers.keys():
            target = self._find_matching_question(key, question_map)
            if not target:
                continue
            saved_reference = self._get_saved_reference_text(reference_map, key, target)
            if saved_reference:
                gemini_ref_map[key] = saved_reference
            else:
                missing_reference_keys.append(str(key))

        if missing_reference_keys:
            raise ValueError("Marking schema must be confirmed before grading")

        # -------------------------------------------------------
        # PHASE 0.5: Pre-calculate all embeddings (SPEEDUP)
        # -------------------------------------------------------
        all_context_sentences = []

        for key, student_text in answer_doc.mapped_answers.items():
            if isinstance(student_text, str) and student_text.strip().lower() not in ["null", "none", ""]:
                all_context_sentences.append(student_text)
            target = self._find_matching_question(key, question_map)
            if target:
                q_text = getattr(target, "question_text", "") or getattr(target, "sub_question_text", "")
                if q_text:
                    all_context_sentences.append(q_text)
                # Use Gemini-extracted ref if available, else fall back to BM25
                ref = gemini_ref_map.get(key) or self._get_reference_context(target, syllabus_text, rubric_text)
                if ref:
                    all_context_sentences.extend([s.strip() for s in re.split(r'[.!?\n]', ref) if len(s.strip()) > 10])

        if progress_callback:
            progress_callback("evaluating_answers", "Mass-calculating sentence features...", percent=10)

        ensure_sentences_cached(all_context_sentences)

        # -------------------------------------------------------
        # PHASE 1: System scoring (XLM-R — CPU-bound)
        # Uses Gemini-extracted references (primary) with BM25 fallback.
        # -------------------------------------------------------
        score_inputs: List[Tuple] = []
        for key, student_text in answer_doc.mapped_answers.items():
            target = self._find_matching_question(key, question_map)
            if not target:
                logger.error(f"UNRESOLVED mapping key: {key}")
                continue

            if not student_text or str(student_text).strip().lower() in ["null", "none", ""]:
                continue

            if isinstance(target, Question) and self._has_any_mapped_descendant(target, answer_doc.mapped_answers):
                continue

            max_marks = self._resolve_max_marks(target, confirmed_mark_overrides)
            # PRIMARY: Gemini-extracted reference points
            # FALLBACK: BM25 retrieval from syllabus / correct_answer gold standard
            reference_text = gemini_ref_map.get(key)
            if not reference_text:
                reference_text = self._get_reference_context(target, syllabus_text, rubric_text)
                if reference_text:
                    logger.debug(f"Using BM25/gold-standard fallback for key: {key}")
            else:
                logger.debug(f"Using Gemini reference for key: {key} ({len(reference_text)} chars)")
            score_inputs.append((key, student_text, target, max_marks, reference_text))

        all_student_texts = [item[1] for item in score_inputs if item[1]]
        all_ref_texts = list(set([item[4] for item in score_inputs if item[4]]))
        student_emb_map = {text: _embedding_cache.get(text) for text in all_student_texts}
        ref_emb_map = {text: _embedding_cache.get(text) for text in all_ref_texts}

        scored_items: List[Dict] = []
        seen_question_ids = set()
        seen_keys = set()

        logger.info(f"Starting grading loop for {len(answer_doc.mapped_answers)} mapped items.")

        for key, student_text in answer_doc.mapped_answers.items():
            target = self._find_matching_question(key, question_map)

            if not target:
                logger.warning(f"No question target found for key: '{key}'")
                continue

            if not student_text or str(student_text).strip().lower() in ["null", "none", ""]:
                logger.info(f"Skipping empty answer for key: '{key}'")
                continue

            target_id = getattr(target, 'id', None)
            display_number = self._resolve_display_number(target, key)
            part_name = self._resolve_part_name(target)

            logger.info(f"Processing question {key} -> ID: {target_id} (Part: {part_name}, Label: {display_number})")

            if target_id in seen_question_ids or key in seen_keys:
                logger.warning(f"Skipping duplicate mapping for question: {key} (ID: {target_id})")
                continue

            if isinstance(target, Question) and self._has_any_mapped_descendant(target, answer_doc.mapped_answers):
                logger.info(f"Skipping parent question {key} because descendant sub-questions are mapped.")
                continue

            seen_question_ids.add(target_id)
            seen_keys.add(key)

            max_marks = self._resolve_max_marks(target, confirmed_mark_overrides)
            # Use Gemini-extracted ref (primary) with BM25 fallback
            reference_text = gemini_ref_map.get(key) or self._get_reference_context(target, syllabus_text, rubric_text)

            processed_count += 1
            if progress_callback:
                pct = 5 + int((processed_count / total_questions) * 65)
                progress_callback("evaluating_answers", f"Scoring question {display_number}...", percent=pct)

            s_emb = student_emb_map.get(student_text)
            r_emb = ref_emb_map.get(reference_text)

            # -------------------------------------------------------
            # SYSTEM SCORE — XLM-R is the SOLE marks engine.
            # Gemini is never used for marks. Only for feedback text.
            # -------------------------------------------------------
            system_score_ratio = self._calculate_system_score(
                student_text=student_text,
                reference_text=reference_text,
                weights=rubric_weights,
                max_marks=max_marks,
                student_emb=s_emb,
                reference_emb=r_emb
            )
            awarded_marks = Decimal(str(system_score_ratio * max_marks)).quantize(Decimal("0.01"))

            # Log calibration data for every question so thresholds can be verified
            logger.info(
                f"[SYSTEM_SCORE] Q{display_number} | "
                f"system_ratio={system_score_ratio:.4f} | "
                f"awarded={awarded_marks}/{max_marks} | "
                f"answer='{str(student_text)[:80]}'"
            )

            scored_items.append({
                "key": key,
                "student_text": student_text,
                "reference_text": reference_text,
                "target": target,
                "max_marks": max_marks,
                "awarded_marks": awarded_marks,
                "display_number": display_number,
                "student_emb": s_emb
            })

        logger.info(f"Scoring loop finished. Scored {len(scored_items)} items total.")

        # Pre-compute the exact marks that will be persisted so feedback can be
        # written against the same score the user sees in the UI.
        for item in scored_items:
            system_ratio = float(item["awarded_marks"]) / max(1, item["max_marks"])
            final_snapped_ratio = self._apply_discrete_bands(system_ratio, item["max_marks"])
            final_marks = Decimal(str(final_snapped_ratio * item["max_marks"])).quantize(Decimal("0.5"))
            item["system_ratio"] = system_ratio
            item["final_snapped_ratio"] = final_snapped_ratio
            item["final_marks"] = final_marks

        # -------------------------------------------------------
        # PHASE 2: Gemini feedback ONLY — no marks from Gemini ever
        # Only runs for Paper II (max_marks > 2) to save time.
        # -------------------------------------------------------
        eval_data_map: Dict[str, Dict] = {}

        if include_feedback and scored_items:
            if progress_callback:
                progress_callback("generating_feedback", "Generating Sinhala feedback (Paper II only)...", percent=70)

            # SPEED FIX: Only send Paper II essay questions to Gemini.
            # Paper I short answers (max_marks <= 2) use a simple system-generated message.
            essay_items = [item for item in scored_items if item["max_marks"] > 2]
            short_answer_items = [item for item in scored_items if item["max_marks"] <= 2]

            # Generate feedback only for essays via Gemini
            if essay_items:
                batch_results = self._get_batch_feedback_from_gemini(essay_items)
                eval_data_map.update(batch_results)

            # Short answer feedback is generated locally — improved for descriptiveness
            for item in short_answer_items:
                ratio = float(item["awarded_marks"]) / max(1, item["max_marks"])
                if ratio >= 0.99:
                    fb = "නිවැරදි පිළිතුරයි. ඔබ බලාපොරොත්තු වූ ප්‍රධාන කරුණ නිවැරදිව දක්වා ඇත."
                elif ratio >= 0.49:
                    fb = "පිළිතුර අර්ධ වශයෙන් නිවැරදිය. ප්‍රධාන කරුණ සඳහන් කර ඇති නමුත්, එය තවදුරටත් පැහැදිලි කිරීම හෝ අදාළ නිවැරදි පාරිභාෂිතය භාවිතා කිරීම අවශ්‍ය වේ."
                else:
                    fb = "පිළිතුර අසම්පූර්ණ හෝ නිවැරදි නොවේ. විෂය නිර්දේශයට අනුව නිවැරදි කරුණු කෙරෙහි වැඩි අවධානයක් යොමු කරන්න."
                eval_data_map[item["key"]] = {
                    "feedback": f"**ප්‍රශ්නය {item['display_number']}** {fb}"
                }

            if progress_callback:
                progress_callback("generating_feedback", "Feedback completed.", percent=95)
        else:
            if progress_callback:
                progress_callback("generating_feedback", "Using system-only scoring.", percent=95)

        # -------------------------------------------------------
        # PHASE 3: Persist QuestionScore rows
        # CRITICAL: final_marks come ONLY from XLM-R system score.
        # Gemini data is only read for .feedback text — never for marks.
        # -------------------------------------------------------
        if progress_callback:
            progress_callback("finalizing_results", "Saving scores...", percent=90)

        from app.services.evaluation.paper_config_service import PaperConfigService
        eval_session = self.db.query(EvaluationSession).filter(EvaluationSession.id == eval_session_id).first()
        chat_session_id = eval_session.session_id if eval_session else None

        config_service = PaperConfigService(self.db)
        paper_configs = self.db.query(PaperConfig).filter(
            PaperConfig.is_confirmed.is_(True),
            (PaperConfig.evaluation_session_id == eval_session_id) |
            (PaperConfig.chat_session_id == chat_session_id)
        ).all()

        selection_map = {}
        config_info_by_part = {}
        for c in paper_configs:
            logger.info(
                "[CONFIRMED_PAPER_CONFIG] session=%s source=%s part=%s total_marks=%s total_main_questions=%s selection_rules=%s",
                eval_session_id,
                "evaluation_session" if c.evaluation_session_id == eval_session_id else "chat_session",
                c.paper_part,
                c.total_marks,
                c.total_main_questions,
                c.selection_rules,
            )
            part_key = self._normalize_part_key(c.paper_part)
            if part_key:
                config_info_by_part[part_key] = {
                    "total_marks": Decimal(str(c.total_marks)) if c.total_marks is not None else None,
                    "total_main_questions": c.total_main_questions,
                }
            if c.selection_rules:
                # Normalize key to lower_snake_case for mapping
                norm_key = c.paper_part.lower().replace(" ", "_")
                # Handle common Roman numeral suffixes generically
                if re.search(r'(_iv|iv)$', norm_key): norm_key = "paper_iv"
                elif re.search(r'(_iii|iii)$', norm_key): norm_key = "paper_iii"
                elif re.search(r'(_ii|ii)$', norm_key): norm_key = "paper_ii"
                elif re.search(r'(_i|i)$', norm_key): norm_key = "paper_i"
                
                selection_map[norm_key] = c.selection_rules

        all_objs = {id(v): v for v in question_map.values()}.values()
        questions = [q for q in all_objs if isinstance(q, Question)]

        all_target_ids = {str(q.id) for q in questions if not isinstance(q, SubQuestion)}
        all_target_ids.update({str(sq.id) for q in questions for sq in getattr(q, 'sub_questions', []) or []})

        scored_target_ids = {str(item["target"].id) for item in scored_items}

        full_q_map = {}
        # Ensure unique questions by ID to prevent duplication
        unique_questions = []
        seen_qids = set()
        for q in questions:
            if str(q.id) not in seen_qids:
                unique_questions.append(q)
                seen_qids.add(str(q.id))

        for q in unique_questions:
            full_q_map[str(q.id)] = q
            for sq in getattr(q, 'sub_questions', []) or []:
                full_q_map[str(sq.id)] = sq

        part_scores = {}

        for q_id, target in full_q_map.items():
            item = next((it for it in scored_items if str(it["target"].id) == q_id), None)

            raw_part = self._resolve_part_name(target)
            part_name = raw_part.lower().replace(" ", "_")
            if re.search(r'(_iv|iv)$', part_name): part_name = "paper_iv"
            elif re.search(r'(_iii|iii)$', part_name): part_name = "paper_iii"
            elif re.search(r'(_ii|ii)$', part_name): part_name = "paper_ii"
            elif re.search(r'(_i|i)$', part_name): part_name = "paper_i"

            if item:
                # -------------------------------------------------------
                # MARKS SOURCE: XLM-R system score only.
                # _apply_discrete_bands snaps to clean mark steps.
                # Gemini output is NOT read here for marks under any circumstance.
                # -------------------------------------------------------
                system_ratio = item.get("system_ratio") or float(item["awarded_marks"]) / max(1, item["max_marks"])
                final_snapped_ratio = item.get("final_snapped_ratio")
                if final_snapped_ratio is None:
                    final_snapped_ratio = self._apply_discrete_bands(system_ratio, item["max_marks"])
                final_marks = item.get("final_marks")
                if final_marks is None:
                    final_marks = Decimal(str(final_snapped_ratio * item["max_marks"])).quantize(Decimal("0.5"))
                # Feedback text from Gemini (text only, never marks)
                meaning_data = eval_data_map.get(item["key"], {})
                feedback = meaning_data.get("feedback", "පිළිතුර අගය කරන ලදී.")
                if not feedback or self._feedback_contradicts_score(feedback, item):
                    logger.warning(
                        "Replacing missing or score-contradictory feedback for Q%s. final_marks=%s/%s feedback=%s",
                        item["display_number"],
                        final_marks,
                        item["max_marks"],
                        str(feedback)[:120],
                    )
                    feedback = self._build_score_aligned_feedback(item)
                logger.info(
                    f"[FINAL_MARK] Q{item['display_number']} | "
                    f"system_ratio={system_ratio:.4f} -> snapped_ratio={final_snapped_ratio:.4f} -> "
                    f"final_marks={final_marks}/{item['max_marks']}"
                )
                item_for_persistence = item
            else:
                final_marks = Decimal(0)
                feedback = "පිළිතුර හමු නොවුණි (0 ලකුණු)."
                if isinstance(target, SubQuestion) and target.question:
                    parent_q = target.question
                    parent_sqs = getattr(parent_q, 'sub_questions', []) or []
                    any_sq_scored = any(
                        any(str(sq.id) == str(it["target"].id) for it in scored_items)
                        for sq in parent_sqs
                    )
                    if not any_sq_scored:
                        parent_item = next(
                            (it for it in scored_items if str(it["target"].id) == str(parent_q.id)),
                            None
                        )
                        if parent_item:
                            n_sqs = max(1, len(parent_sqs))
                            distributed = parent_item["awarded_marks"] / Decimal(str(n_sqs))
                            final_marks = distributed.quantize(Decimal("0.5"), rounding=ROUND_HALF_UP)
                            feedback = "ප්‍රශ්නයේ සම්පූර්ණ පිළිතුරෙන් ලකුණු ලබා ගන්නා ලදී."

                # Create a bare-bones item for scoring logic if not in scored_items
                missing_item = {
                    "target": target,
                    "max_marks": self._resolve_max_marks(target, confirmed_mark_overrides),
                    "display_number": self._resolve_display_number(target, q_id)
                }
                # Try to find the key from mapped_answers if possible to capture the text even for skip-scored items
                matched_key = next((k for k, v in answer_doc.mapped_answers.items() if str(getattr(self._find_matching_question(k, question_map), 'id', '')) == q_id), None)
                if matched_key:
                    missing_item["key"] = matched_key
                    missing_item["student_text"] = answer_doc.mapped_answers[matched_key]
                
                item_for_persistence = missing_item

            # Capture student text from multiple possible sources
            student_text_captured = item_for_persistence.get("student_text") or \
                                    item_for_persistence.get("student_answer") or \
                                    (answer_doc.mapped_answers.get(item_for_persistence.get("key")) if item_for_persistence.get("key") else None)
                                    
            logger.info(f"DEBUG: Saving QuestionScore for Q_ID={target.id}. Captured text: {student_text_captured[:50] if student_text_captured else 'None'}")

            has_student_text = bool(
                student_text_captured and str(student_text_captured).strip().lower() not in ["null", "none", ""]
            )

            q_score_params = {
                "evaluation_result_id": eval_result.id,
                "awarded_marks": final_marks,
                "feedback": feedback,
                "student_answer": student_text_captured
            }
            if isinstance(target, SubQuestion):
                q_score_params["sub_question_id"] = target.id
            else:
                q_score_params["question_id"] = target.id

            q_score = QuestionScore(**q_score_params)
            self.db.add(q_score)

            if part_name not in part_scores:
                part_scores[part_name] = {}

            is_leaf = True
            main_q = target
            if isinstance(target, SubQuestion):
                if getattr(target, 'children', []): is_leaf = False
                main_q = target.question
            else:
                if getattr(target, 'sub_questions', []): is_leaf = False

            # New attempt tracking: count only rows that actually captured student text,
            # so selection logic reflects real attempts instead of placeholder rows.
            main_q_id = str(main_q.id) if main_q else "orphan"
            if main_q_id not in part_scores[part_name]:
                part_scores[part_name][main_q_id] = {"marks": [], "max_marks": [], "attempted": 0}

            if has_student_text:
                part_scores[part_name][main_q_id]["attempted"] += 1

            if is_leaf:
                part_scores[part_name][main_q_id]["marks"].append(final_marks)
                part_scores[part_name][main_q_id]["max_marks"].append(
                    Decimal(str(item_for_persistence.get("max_marks") or 0))
                )

        self.db.flush()

        # 4. Final total score using selection rules
        total_score_val = Decimal(0)
        norm_selection_map = {k.lower().replace(" ", "_"): v for k, v in selection_map.items()}

        for part_name, section_map in part_scores.items():
            main_q_entries = []
            for mq_id, data in section_map.items():
                total = sum(data["marks"])
                attempted = data["attempted"]
                main_q_entries.append((mq_id, total, attempted))

            part_rules = norm_selection_map.get(part_name) or {}
            mode = str(part_rules.get('mode', 'all')).lower()
            # Support both 'any' and 'choose_any' as synonyms
            is_selection_mode = mode in ['any', 'choose_any', 'choose']
            count = part_rules.get('count') or part_rules.get('total') or part_rules.get('choose_any')

            full_part_max = sum(sum(data.get("max_marks", [])) for data in section_map.values())
            configured_part_total = (config_info_by_part.get(part_name) or {}).get("total_marks")
            known_config_total = sum(
                info["total_marks"]
                for info in config_info_by_part.values()
                if info.get("total_marks") is not None
            )
            session_total_marks = getattr(eval_session, "total_marks", None)
            overall_total = Decimal(str(session_total_marks or 0)) if session_total_marks else Decimal(0)
            inferred_part_total = None
            if configured_part_total and configured_part_total > 0:
                inferred_part_total = configured_part_total
            elif overall_total > known_config_total and part_name in config_info_by_part:
                missing_parts = [
                    part_key
                    for part_key, info in config_info_by_part.items()
                    if info.get("total_marks") is None
                ]
                if len(missing_parts) == 1 and missing_parts[0] == part_name:
                    inferred_part_total = overall_total - known_config_total

            if (not is_selection_mode) and inferred_part_total and Decimal(0) < inferred_part_total < full_part_max:
                main_max_values = [sum(data.get("max_marks", [])) for data in section_map.values()]
                main_max_values = [value for value in main_max_values if value > 0]
                typical_main_max = max(set(main_max_values), key=main_max_values.count) if main_max_values else Decimal(0)
                if typical_main_max and inferred_part_total % typical_main_max == 0:
                    inferred_count = int(inferred_part_total / typical_main_max)
                    if 0 < inferred_count < len(main_q_entries):
                        mode = "any"
                        count = inferred_count
                        is_selection_mode = True
                        logger.info(
                            "[INFERRED_SELECTION_RULES] part=%s inferred_total=%s typical_main_max=%s mode=any count=%s",
                            part_name,
                            inferred_part_total,
                            typical_main_max,
                            inferred_count,
                        )
            elif (not is_selection_mode) and part_name != "paper_i" and part_name in config_info_by_part:
                attempted_count = sum(1 for data in section_map.values() if data.get("attempted", 0) > 0)
                has_unattempted = any(data.get("attempted", 0) <= 0 for data in section_map.values())
                if attempted_count > 0 and has_unattempted:
                    mode = "attempted_only"
                    count = attempted_count
                    is_selection_mode = True
                    logger.info(
                        "[INFERRED_SELECTION_RULES] part=%s mode=attempted_only count=%s reason=missing_config_selection",
                        part_name,
                        attempted_count,
                    )

            if is_selection_mode and count and len(main_q_entries) > int(count):
                # New selection rule: prefer genuinely attempted main questions;
                # only backfill with unattempted zero-mark questions when the paper
                # requires more selected slots than were actually answered.
                attempted_entries = [e for e in main_q_entries if e[2] > 0]
                unattempted_entries = [e for e in main_q_entries if e[2] <= 0]
                attempted_entries.sort(key=lambda x: (x[1], x[2]), reverse=True)
                unattempted_entries.sort(key=lambda x: (x[1], x[2]), reverse=True)

                selected = attempted_entries[:int(count)]
                if len(selected) < int(count):
                    selected.extend(unattempted_entries[: int(count) - len(selected)])

                selected_totals = [e[1] for e in selected]
                attempted_count = sum(1 for e in selected if e[2] > 0)
                logger.info(
                    "[GRADING_SELECTION] part=%s mode=%s required_count=%s main_questions=%s selected=%s attempted_selected=%s",
                    part_name,
                    mode,
                    count,
                    len(main_q_entries),
                    [
                        {
                            "main_question_id": e[0],
                            "awarded": str(e[1]),
                            "attempted": e[2],
                        }
                        for e in selected
                    ],
                    attempted_count,
                )
                total_score_val += sum(selected_totals)
            else:
                logger.info(
                    "[GRADING_SELECTION] part=%s mode=%s required_count=%s main_questions=%s selected=all",
                    part_name,
                    mode,
                    count,
                    len(main_q_entries),
                )
                total_score_val += sum(e[1] for e in main_q_entries)

        eval_result.total_score = total_score_val.quantize(Decimal("0.01"))
        logger.info(f"Final Total Score (system-only, selection-aware): {eval_result.total_score}")

        if include_feedback:
            if progress_callback:
                progress_callback("generating_feedback", "Generating overall feedback...", percent=98)
            eval_result.overall_feedback = self._generate_overall_feedback(
                eval_result.total_score, eval_result.id
            )

        if progress_callback:
            progress_callback("completed", "Evaluation completed successfully.", percent=100)

        self.db.commit()
        logger.info(f"Grading transaction committed for result ID: {eval_result.id}")
        return eval_result


    def generate_feedback_for_result(self, answer_doc_id: UUID, user_id: UUID) -> Optional[EvaluationResult]:
        """Generate Gemini feedback text for an existing evaluation result."""
        result = self.db.query(EvaluationResult).filter(EvaluationResult.answer_document_id == answer_doc_id).first()
        if not result:
            logger.error(f"Cannot generate feedback: No evaluation result for {answer_doc_id}")
            return None

        answer_doc = (
            self.db.query(AnswerDocument)
            .filter(AnswerDocument.id == answer_doc_id)
            .first()
        )
        if not answer_doc:
            return None

        eval_session = (
            self.db.query(EvaluationSession)
            .filter(EvaluationSession.id == answer_doc.evaluation_session_id)
            .first()
        )
        if not eval_session:
            return None

        from app.services.evaluation.evaluation_workflow_service import EvaluationWorkflowService
        workflow = EvaluationWorkflowService(self.db)

        syllabus_text, rubric_text, questions = workflow._get_evaluation_context(eval_session.id)
        question_map = workflow._build_question_map_helper(questions)
        reference_map = self._load_confirmed_reference_map(eval_session.id, eval_session.session_id, user_id)
        confirmed_mark_overrides = self._load_confirmed_mark_overrides(
            eval_session_id=eval_session.id,
            chat_session_id=eval_session.session_id,
            questions=questions,
        )

        scores = self.db.query(QuestionScore).filter(QuestionScore.evaluation_result_id == result.id).all()

        scored_items = []
        for qs in scores:
            target = None
            if qs.sub_question_id:
                target = qs.sub_question
            elif qs.question_id:
                target = qs.question

            if not target:
                continue

            # New feedback guard: skip placeholder rows so generated feedback only
            # talks about answers the student actually wrote.
            student_text = (qs.student_answer or "").strip()
            if not student_text or student_text.lower() in ["null", "none"]:
                continue

            key = str(target.id)

            display_number = self._resolve_display_number(target, key)
            reference_text = self._get_saved_reference_text(reference_map, key, target)
            if not reference_text:
                continue

            scored_items.append({
                "key": key,
                "student_text": student_text,
                "reference_text": reference_text,
                "target": target,
                "max_marks": self._resolve_max_marks(target, confirmed_mark_overrides),
                "awarded_marks": qs.awarded_marks,
                "display_number": display_number,
                "db_score_obj": qs
            })

        if not scored_items:
            return result

        logger.info(f"Generating on-demand batch & overall feedback for result {result.id}")

        # Only send essays to Gemini. Keep feedback calls sequential because the
        # paid API can still return transient 503s when several Flash requests
        # are fired at once.
        essay_items = [i for i in scored_items if i["max_marks"] > 2]
        try:
            eval_data_map = self._get_batch_feedback_from_gemini(essay_items)
        except Exception as e:
            logger.error(f"Batch feedback generation failed: {e}")
            eval_data_map = {}

        try:
            overall_feedback = self._generate_overall_feedback(result.total_score, result.id)
        except Exception as e:
            logger.error(f"Overall feedback generation failed: {e}")
            overall_feedback = f"Total Score: {result.total_score}"

        # Generate short-answer feedback locally - improved templates
        for item in scored_items:
            if item["max_marks"] <= 2:
                ratio = float(item["awarded_marks"]) / max(1, item["max_marks"])
                if ratio >= 0.99:
                    fb = "නිවැරදි පිළිතුරයි. ප්‍රධාන කරුණු නිවැරදිව ඉදිරිපත් කර ඇත."
                elif ratio >= 0.49:
                    fb = "පිළිතුර අර්ධ වශයෙන් නිවැරදිය. වඩාත් සම්පූර්ණ පිළිතුරක් සඳහා තවදුරටත් විස්තර කිරීම අවශ්‍ය වේ."
                else:
                    fb = "පිළිතුර නිවැරදි නොවේ හෝ අදාළ නොවේ. විෂය කරුණු නැවත අධ්‍යයනය කරන්න."
                eval_data_map[item["key"]] = {
                    "feedback": f"**ප්‍රශ්නය {item['display_number']}** {fb}"
                }

        current_scores = {qs.id: qs for qs in self.db.query(QuestionScore).filter(QuestionScore.evaluation_result_id == result.id).all()}

        for item in scored_items:
            qs_id = item["db_score_obj"].id
            qs = current_scores.get(qs_id)

            if not qs:
                logger.warning(f"QuestionScore {qs_id} no longer exists.")
                continue

            gemini_data = eval_data_map.get(item["key"], {})
            feedback = gemini_data.get("feedback")
            # IMPORTANT: Only update feedback text, never marks
            if feedback:
                qs.feedback = feedback

        result.overall_feedback = overall_feedback

        try:
            self.db.commit()
            logger.info(f"Feedback successfully saved for result {result.id}")
        except StaleDataError:
            self.db.rollback()
            logger.error(f"StaleDataError while saving feedback for result {result.id}.")
        except Exception as e:
            self.db.rollback()
            logger.error(f"Unexpected error while committing feedback: {e}")
            raise

        return result


    def build_reference_map_for_targets(
        self,
        eval_session_id: UUID,
        syllabus_text: str,
        rubric_text: str,
        targets: List[Dict[str, Any]],
    ) -> Dict[str, str]:
        questions_for_extraction: List[Dict[str, Any]] = []
        for target_info in targets:
            target = target_info["target"]
            question_text = target_info.get("question_text") or ""
            if not question_text.strip():
                continue

            questions_for_extraction.append(
                {
                    "key": target_info["key"],
                    "target": target,
                    "question_text": question_text,
                    "max_marks": target_info.get("max_marks") or self._resolve_max_marks(target),
                    "display_number": target_info.get("question_number") or target_info["key"],
                    "part_name": target_info.get("part_name") or self._resolve_part_name(target),
                    "question_type": getattr(target, "question_type", "short") or "short",
                }
            )

        reference_source = self._build_reference_source_text(syllabus_text, rubric_text)
        cache_source = f"{self.SCHEMA_REFERENCE_FORMAT_VERSION}\n{reference_source}"
        source_hash = sha256(cache_source.encode("utf-8")).hexdigest()[:12]
        cache_key = f"{eval_session_id}:{source_hash}"
        gemini_ref_map: Dict[str, str] = {}
        with GradingService._gemini_ref_cache_lock:
            cached = GradingService._gemini_ref_cache.get(cache_key)

        if cached is not None:
            gemini_ref_map = cached
            logger.info("Using cached marking references for session %s (%s refs).", cache_key, len(gemini_ref_map))
        elif reference_source and questions_for_extraction:
            gemini_ref_map = self._batch_extract_reference_points(questions_for_extraction, reference_source)
            with GradingService._gemini_ref_cache_lock:
                GradingService._gemini_ref_cache[cache_key] = gemini_ref_map

        final_map: Dict[str, str] = {}
        for target_info in targets:
            target = target_info["target"]
            question_info = {
                "key": target_info["key"],
                "target": target,
                "question_text": target_info.get("question_text") or "",
                "max_marks": target_info.get("max_marks") or self._resolve_max_marks(target),
                "display_number": target_info.get("question_number") or target_info["key"],
                "part_name": target_info.get("part_name") or self._resolve_part_name(target),
                "question_type": getattr(target, "question_type", "short") or "short",
            }
            extracted_reference = gemini_ref_map.get(target_info["key"], "")
            shaped_reference = self._shape_reference_for_schema(extracted_reference, question_info)
            if shaped_reference:
                final_map[target_info["key"]] = shaped_reference
                continue

            final_map[target_info["key"]] = ""
        return final_map

    def _build_reference_source_text(self, syllabus_text: str, rubric_text: str) -> str:
        return (syllabus_text or "").strip()

    def _get_saved_reference_text(self, reference_map: Dict[str, str], key: str, target: Any) -> str:
        candidates = [str(key), self._normalize_question_lookup_key(key)]
        target_id = getattr(target, "id", None)
        if target_id:
            candidates.append(str(target_id))

        display_number = self._resolve_display_number(target, key)
        candidates.append(str(display_number))
        candidates.append(self._normalize_question_lookup_key(display_number))

        for candidate in candidates:
            if candidate in reference_map and reference_map[candidate]:
                return reference_map[candidate]
        return ""

    def _normalize_question_lookup_key(self, value: str) -> str:
        return re.sub(r"[\s().]", "", str(value or "").lower())

    def _load_confirmed_reference_map(self, eval_session_id: UUID, session_id: UUID, user_id: UUID) -> Dict[str, str]:
        from app.services.evaluation.marking_schema_service import MarkingSchemaService

        schema_service = MarkingSchemaService(self.db)
        try:
            return schema_service.get_confirmed_reference_map(eval_session_id, user_id)
        except Exception:
            return schema_service.get_confirmed_reference_map(session_id, user_id)


    # ----------------------------------------------------------
    # SCORING HELPERS
    # ----------------------------------------------------------

    def _semantic_similarity(self, student_text: str, reference_text: str, student_emb=None, reference_emb=None, max_marks: int = 2) -> float:
        """
        Semantic similarity using sentence-level max-match.
        XLM-R on Sinhala OCR text vs syllabus text.

        CALIBRATION (confirmed from actual marks breakdown, March 2026):
        - Correct SHORT answers (Paper I, <=2 marks): cosine sim ~ 0.36–0.48
        - Correct ESSAY answers (Paper II, >2 marks): cosine sim ~ 0.28–0.38
          Essays land LOWER because:
          (a) Multi-sentence answers dilute the embedding vs a short syllabus sentence
          (b) Sinhala OCR noise is compounded across longer text
        - Partial:  sim ~ 0.20–0.30
        - Wrong:    sim ~ 0.10–0.20
        """
        if not reference_text or not student_text:
            return 0.0

        sentences = self._split_reference_concepts(reference_text)
        if not sentences:
            with ml_semaphore:
                emb1 = student_emb if student_emb is not None else _embedding_cache.get(student_text)
                if emb1 is None: emb1 = get_xlmr_model().encode(student_text, convert_to_tensor=True)

                emb2 = reference_emb if reference_emb is not None else _embedding_cache.get(reference_text)
                if emb2 is None: emb2 = get_xlmr_model().encode(reference_text, convert_to_tensor=True)

            raw_sim = _cosine_util().cos_sim(emb1, emb2).item()
            return self._sinhala_sigmoid_boost(raw_sim, max_marks)

        s_emb = student_emb if student_emb is not None else _embedding_cache.get(student_text)
        if s_emb is None:
            with ml_semaphore:
                s_emb = get_xlmr_model().encode(student_text, convert_to_tensor=True)

        missing = [s for s in sentences if s not in _embedding_cache]
        if missing:
            ensure_sentences_cached(missing)

        ref_embs = torch.stack([_embedding_cache[s] for s in sentences])
        sims = _cosine_util().cos_sim(s_emb, ref_embs)[0]

        # Blend max-match (70%) with top-3 average (30%) for stability
        k = min(3, len(sims))
        topk_values = torch.topk(sims, k=k).values
        combined_sim = (0.7 * sims.max().item()) + (0.3 * topk_values.mean().item())

        logger.info(f"[SIM] max_sim={sims.max().item():.4f} combined_sim={combined_sim:.4f} max_marks={max_marks}")
        return self._sinhala_sigmoid_boost(combined_sim, max_marks)


    def _sinhala_sigmoid_boost(self, sim: float, max_marks: int = 2) -> float:
        """
        Maps XLM-R cosine similarity → [0, 1] score.
        Uses SEPARATE calibrated thresholds for short answers vs essays.
        """
        if max_marks > 2:
            # ESSAY thresholds (calibrated for paper II)
            if sim >= 0.40:   return 1.0
            if sim >= 0.30:   return 0.50 + (sim - 0.30) * (0.50 / 0.10)   
            if sim >= 0.15:   return 0.0  + (sim - 0.15) * (0.50 / 0.15)   
            return 0.0
        else:
            # SHORT ANSWER thresholds (Paper I or dynamic marks)
            # KEY FIX: More generous for full marks (0.30 instead of 0.35) 
            # as short MCQ answers have lower cosine sim variance.
            if sim >= 0.30:   return 1.0
            if sim >= 0.18:   return 0.50 + (sim - 0.18) * (0.50 / 0.12)   
            if sim >= 0.08:   return 0.0  + (sim - 0.08) * (0.50 / 0.10)   
            return 0.0


    # ----------------------------------------------------------
    # MARKING BANDS
    # ----------------------------------------------------------
    def _apply_discrete_bands(self, ratio: float, max_marks: int) -> float:
        """
        Snaps the continuous system_score_ratio to clean 0.5 mark steps.

        THRESHOLD DESIGN (Improved Fairness):
        - Paper I / Short: Below 0.20 gives zero
        - Paper II / Essay: Below 0.15 gives zero (more lenient for depth coverage)
        """
        # New essay calibration: do not auto-promote essays to full marks from
        # merely "good" similarity. Paper II should keep separation between
        # partial, good, and excellent answers.
        high_threshold = 0.78 if max_marks <= 2 else 0.97
        low_threshold = 0.20 if max_marks <= 2 else 0.15

        if ratio >= high_threshold: return 1.0
        if ratio < low_threshold: return 0.0

        actual_marks = ratio * max_marks

        # Always use 0.5 mark steps to avoid inflation from rounding
        step = Decimal("0.5")
        snapped_marks = (Decimal(str(actual_marks)) / step).quantize(
            Decimal("1"), rounding=ROUND_HALF_UP
        ) * step
        if max_marks > 2 and ratio < high_threshold:
            snapped_marks = min(snapped_marks, Decimal(str(max_marks)) - step)

        final_ratio = float(snapped_marks / Decimal(str(max_marks)))
        return min(1.0, final_ratio)


    def _apply_marking_band(self, semantic: float, coverage: float) -> float:
        combined = (0.6 * semantic) + (0.4 * coverage)
        if combined >= 0.85: return 1.0
        if combined >= 0.70: return 0.75
        if combined >= 0.55: return 0.50
        if combined >= 0.40: return 0.25
        return 0.0


    # ----------------------------------------------------------
    # CONTEXT & COVERAGE
    # ----------------------------------------------------------
    def _is_placeholder_answer(self, text: str) -> bool:
        """Detects useless placeholder answers generated during extraction."""
        if not text: return True
        
        placeholders = [
            "විස්තර කිරීම", "පිළිබඳව", "පිටුව", "හමුවන්නේ නැත", 
            "සඳහන් වේ", "මෙම ප්‍රශ්නයට අදාළ තොරතුරු", 
            "Syllabus mentions", "page", "description of"
        ]
        
        cleaned = text.strip()
        if len(cleaned) < 5: return True
        
        # If the length is very small and contains these patterns
        if len(cleaned) < 100:
            for p in placeholders:
                if p in cleaned:
                    return True
        return False

    def _get_reference_context(
        self,
        question,
        syllabus_text: str,
        rubric_text: str,
        prefer_gold_standard: bool = True,
    ) -> str:
        correct = getattr(question, "correct_answer", None)
        if prefer_gold_standard and correct and not self._is_placeholder_answer(correct):
            logger.info(f"Using Gold Standard context for question {getattr(question, 'id')}")
            return correct.strip()

        q_number = getattr(question, "question_number", "") or getattr(question, "label", "")
        q_text = getattr(question, "question_text", "") or getattr(question, "sub_question_text", "")

        source = syllabus_text
        if not source:
            if correct and not self._is_placeholder_answer(correct):
                logger.info(f"Using Gold Standard context for question {getattr(question, 'id')}")
                return correct.strip()
            return ""

        chunks = [c.strip() for c in source.split("\n") if len(c.strip()) > 5]
        search_query = f"{q_number} {q_text}"
        bm25 = BM25Okapi([c.split() for c in chunks])
        top_chunks = bm25.get_top_n(search_query.split(), chunks, n=15)

        prio_chunks = [c for c in top_chunks if re.match(rf"^(?:\()?0?{q_number}(?:\.|\)|[\s:]|$)", c.strip())]

        if prio_chunks:
            context = "\n\n".join(prio_chunks[:5])
        else:
            backup_prio = [c for c in top_chunks if f"{q_number}" in c]
            if backup_prio:
                context = "\n\n".join(backup_prio[:5])
            else:
                context = "\n\n".join(top_chunks[:5])

        if len(context) > 2000:
            context = context[:2000] + "... [Context Clipped]"
        return context

    def _build_reference_context_block(self, question_info: Dict[str, Any], syllabus_text: str) -> str:
        target = question_info.get("target")
        question_text = (question_info.get("question_text") or "").strip()
        display_number = str(question_info.get("display_number") or "")

        context = ""
        if target is not None:
            context = self._get_reference_context(
                target,
                syllabus_text,
                rubric_text="",
                prefer_gold_standard=False,
            )

        if not context and syllabus_text:
            chunks = [c.strip() for c in syllabus_text.split("\n") if len(c.strip()) > 5]
            if chunks:
                search_query = f"{display_number} {question_text}".strip()
                bm25 = BM25Okapi([c.split() for c in chunks])
                top_chunks = bm25.get_top_n(search_query.split(), chunks, n=8)
                context = "\n\n".join(top_chunks[:4])

        context = (context or "").strip()
        if len(context) > 2200:
            context = context[:2200] + "... [Context Clipped]"
        return context

    def _build_syllabus_fallback_reference(self, question_info: Dict[str, Any], syllabus_text: str) -> str:
        context = self._build_reference_context_block(question_info, syllabus_text)
        if not context:
            return ""

        cleaned = context.replace("[Context Clipped]", "")
        cleaned = re.sub(r"---\s*(?:PAGE|TABLE)[^-]*---", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if not cleaned or self._is_placeholder_answer(cleaned):
            return ""
        return self._shape_reference_for_schema(cleaned, question_info)

    def _safe_reference_json_loads(self, text: str) -> Any:
        if not text:
            return {}

        clean_text = re.sub(r'^```json\s*|\s*```$', '', text.strip(), flags=re.MULTILINE)
        try:
            return json.loads(clean_text)
        except json.JSONDecodeError:
            repaired = self._repair_reference_json(clean_text)
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                recovered = self._recover_reference_pairs(clean_text)
                if recovered:
                    logger.warning(
                        "[GEMINI_REF] Recovered %s key(s) from malformed JSON response.",
                        len(recovered),
                    )
                    return recovered
                raise

    def _repair_reference_json(self, text: str) -> str:
        repaired = (text or "").strip()
        if not repaired:
            return "{}"
        repaired = re.sub(r",(\s*[}\]])", r"\1", repaired)
        repaired = re.sub(r",\s*$", "", repaired)
        if repaired.count('"') % 2 != 0:
            repaired += '"'

        stack = []
        in_string = False
        escaped = False
        for char in repaired:
            if escaped:
                escaped = False
                continue
            if char == "\\" and in_string:
                escaped = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == "{":
                stack.append("}")
            elif char == "[":
                stack.append("]")
            elif char == "}" and stack and stack[-1] == "}":
                stack.pop()
            elif char == "]" and stack and stack[-1] == "]":
                stack.pop()

        while stack:
            repaired += stack.pop()
        return re.sub(r",(\s*[}\]])", r"\1", repaired)

    def _recover_reference_pairs(self, text: str) -> Dict[str, str]:
        recovered: Dict[str, str] = {}
        pair_pattern = re.compile(
            r'"(?P<key>ref_\d+)"\s*:\s*"(?P<value>(?:\\.|[^"\\])*)"',
            flags=re.DOTALL,
        )
        for match in pair_pattern.finditer(text or ""):
            try:
                recovered[match.group("key")] = json.loads(f'"{match.group("value")}"')
            except Exception:
                recovered[match.group("key")] = match.group("value")
        return recovered

    def _sanitize_extracted_reference(self, value: str) -> str:
        cleaned = (value or "").strip()
        if not cleaned:
            return ""

        disallowed_patterns = [
            r"(?i)\bsyllabus content does not\b",
            r"(?i)\busing internal knowledge\b",
            r"(?i)\bthis document does not contain\b",
            r"(?i)\bnot covered\b",
            r"අඩංගු නොවේ",
            r"සඳහන් නොවේ",
            r"මෙම සාක්ෂිය",
            r"ලබා දී ඇති සාක්ෂි",
            r"මෙම සාධක",
            r"මෙම ලේඛනයේ.+?තොරතුරු අඩංගු නොවේ",
        ]
        for pattern in disallowed_patterns:
            if re.search(pattern, cleaned):
                return ""

        return cleaned[:1500]

    def _normalize_reference_key(self, value: str) -> str:
        cleaned = re.sub(r"\s+", "", str(value or ""))
        return cleaned.replace("[", "(").replace("]", ")")

    def _build_reference_key_aliases(self, idx_key: str, display_number: str) -> set[str]:
        aliases = {self._normalize_reference_key(idx_key)}
        normalized_display = self._normalize_reference_key(display_number)
        if not normalized_display:
            return aliases

        aliases.add(normalized_display)
        dotted_match = re.match(r"^(\d+)[\.\-]([^\.\-]+)$", normalized_display)
        paren_match = re.match(r"^(\d+)\((.+)\)$", normalized_display)
        if dotted_match:
            main, sub = dotted_match.groups()
            aliases.add(f"{main}({sub})")
            aliases.add(f"{main}{sub}")
        elif paren_match:
            main, sub = paren_match.groups()
            aliases.add(f"{main}.{sub}")
            aliases.add(f"{main}{sub}")

        return {alias for alias in aliases if alias}

    def _split_reference_source_chunks(self, source: str) -> List[str]:
        if not source:
            return []

        normalized = re.sub(r"---\s*(?:PAGE|TABLE)[^-]*---", "\n", source, flags=re.IGNORECASE)
        normalized = normalized.replace("\r", "\n")
        raw_segments = re.split(r"\n{2,}", normalized)
        if len(raw_segments) <= 1:
            raw_segments = normalized.split("\n")

        disallowed_markers = [
            "ක්රියාකාරකම",
            "ක්‍රියාකාරකම",
            "à¶šà·Šà¶»à·’à¶ºà·à¶šà·à¶»à¶šà¶¸",
            "පහත ප්රශ්න",
            "පහත ප්‍රශ්න",
            "විග්‍රහාත්මක රචනා ප්‍රශ්න",
            "ශී ලංකා ජාතික ගීය",
            "ගරු අධයාපන අමාත්යතුමාගේ පණිවුඩය",
            "තිළිණය ලෙසින් රජයෙන්",
            "අධයාපන පේ්කාශන දෙපාර්තමේන්තුව",
            "වෙබ් අඩවියට පිවිසෙන්න",
        ]

        chunks: List[str] = []
        for segment in raw_segments:
            cleaned = re.sub(r"\s+", " ", segment).strip()
            if len(cleaned) < 40:
                continue
            if any(marker in cleaned for marker in disallowed_markers):
                continue
            chunks.append(cleaned)
        return chunks

    def _is_placeholder_answer(self, text: str) -> bool:
        """Detects useless placeholder answers generated during extraction."""
        if not text:
            return True

        cleaned = text.strip()
        lowered = cleaned.lower()
        if len(cleaned) < 5:
            return True

        placeholder_markers = [
            "syllabus mentions",
            "description of",
            "[context clipped]",
            "--- page",
            "--- table",
            "this syllabus evidence does not contain",
            "this document does not contain",
            "not covered",
            "අඩංගු නොවේ",
            "සඳහන් නොවේ",
            "සපයා නැත",
            "මෙම සාක්ෂි",
            "මෙම සාධක",
            "මෙම ලේඛනයේ",
            "ලබා දී ඇති සාක්ෂි",
            "පිළිබඳව තොරතුරු",
        ]
        return any(marker in lowered or marker in cleaned for marker in placeholder_markers)

    def _get_reference_context(
        self,
        question,
        syllabus_text: str,
        rubric_text: str,
        prefer_gold_standard: bool = True,
    ) -> str:
        correct = getattr(question, "correct_answer", None)
        if prefer_gold_standard and correct and not self._is_placeholder_answer(correct):
            logger.info(f"Using Gold Standard context for question {getattr(question, 'id')}")
            return correct.strip()

        q_number = getattr(question, "question_number", "") or getattr(question, "label", "")
        q_text = getattr(question, "question_text", "") or getattr(question, "sub_question_text", "")

        source = syllabus_text
        if not source:
            if correct and not self._is_placeholder_answer(correct):
                logger.info(f"Using Gold Standard context for question {getattr(question, 'id')}")
                return correct.strip()
            return ""

        chunks = self._split_reference_source_chunks(source)
        if not chunks:
            return ""

        search_query = q_text or q_number
        if not search_query.strip():
            return ""

        bm25 = BM25Okapi([c.split() for c in chunks])
        ranked_chunks: List[str] = []
        seen_chunks = set()
        for query in self._build_reference_search_queries(
            {"question_text": q_text, "display_number": q_number}
        ):
            top_chunks = bm25.get_top_n(query.split(), chunks, n=8)
            for chunk in top_chunks:
                if chunk in seen_chunks or self._is_placeholder_answer(chunk):
                    continue
                seen_chunks.add(chunk)
                ranked_chunks.append(chunk)
        context = "\n\n".join(ranked_chunks[:6])

        if len(context) > 2000:
            context = context[:2000] + "... [Context Clipped]"
        return context

    def _build_reference_context_block(self, question_info: Dict[str, Any], syllabus_text: str) -> str:
        target = question_info.get("target")
        question_text = (question_info.get("question_text") or "").strip()
        display_number = str(question_info.get("display_number") or "")

        context = ""
        if target is not None:
            context = self._get_reference_context(
                target,
                syllabus_text,
                rubric_text="",
                prefer_gold_standard=False,
            )

        if not context and syllabus_text:
            chunks = self._split_reference_source_chunks(syllabus_text)
            if chunks:
                bm25 = BM25Okapi([c.split() for c in chunks])
                ranked_chunks: List[str] = []
                seen_chunks = set()
                for query in self._build_reference_search_queries(question_info):
                    top_chunks = bm25.get_top_n(query.split(), chunks, n=8)
                    for chunk in top_chunks:
                        if chunk in seen_chunks or self._is_placeholder_answer(chunk):
                            continue
                        seen_chunks.add(chunk)
                        ranked_chunks.append(chunk)
                context = "\n\n".join(ranked_chunks[:6])

        context = (context or "").strip()
        if len(context) > 2200:
            context = context[:2200] + "... [Context Clipped]"
        return context

    def _build_reference_search_queries(self, question_info: Dict[str, Any]) -> List[str]:
        raw_text = str(question_info.get("question_text") or "")
        display_number = str(question_info.get("display_number") or "")
        cleaned = re.sub(r"\([^)]*[A-Za-z0-9\[\]:/]{4,}[^)]*\)", " ", raw_text)
        cleaned = re.sub(r"[^\w\s\u0D80-\u0DFF]", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        queries = [f"{display_number} {cleaned}".strip(), cleaned]
        lower = cleaned.lower()

        topic_queries = []
        if "1960" in lower and "භාෂා" in lower:
            topic_queries.append("1960 රාජ්‍ය භාෂා සිංහල දෙමළ විශේෂ විධිවිධාන පනත")
        if "1977" in lower and ("විවෘත" in lower or "ආර්ථික" in lower):
            topic_queries.append("1977 විවෘත ආර්ථික ප්‍රතිපත්තිය පෞද්ගලික අංශය අපනයන ආනයන විදේශ ආයෝජන")
        if "1972" in lower and "මූලික" in lower:
            topic_queries.append("1972 මූලික අයිතිවාසිකම් නීතිය ඉදිරියේ සමානත්වය සිතීමේ නිදහස ආගමික නිදහස")
        if "1978" in lower and "විධායක" in lower and "ජනාධිපති" in lower:
            topic_queries.append("1978 විධායක ජනාධිපති අගමැති ජනාධිපති ධුරය")
        if "බ්‍රිතාන්" in lower and "අවධානය" in lower:
            topic_queries.append("බ්‍රිතාන්‍ය ශ්‍රී ලංකාව අවධානය ඉන්දියාව බෙංගාල බොක්ක ත්‍රිකුණාමලය කුරුඳු")
        if "කෝල්බ" in lower and "ආර්ථික" in lower:
            topic_queries.append("කෝල්බෲක් ආර්ථික අනිවාර්ය සේවා රාජකාරී රජයේ වෙළෙඳ ඒකාධිකාරය කුරුඳු ලුණු ඉඩම්")
        if "වතු" in lower and ("ආර්ථික" in lower or "වගා" in lower):
            topic_queries.append("වතු වගා ආර්ථිකය කෝපි තේ රබර් ඉන්දියානු ශ්‍රමිකයන් දුම්රිය මහාමාර්ග")
        if "ඩොනමෝර්" in lower and ("සර්ව" in lower or "කාරක" in lower):
            topic_queries.append("ඩොනමෝර් සර්වජන ඡන්ද බලය විධායක කාරක සභා රාජ්‍ය මන්ත්‍රණ සභාව")
        if "සෝල්බ" in lower:
            topic_queries.append("සෝල්බරි ආණ්ඩුක්‍රමය නිදහස අග්‍රාණ්ඩුකාර පාර්ලිමේන්තුව කැබිනට් මණ්ඩලය")
        if "නිදහස්" in lower and "අධ්" in lower:
            topic_queries.append("නිදහස් අධ්‍යාපන ප්‍රතිපත්තිය පාසල් විශ්වවිද්‍යාල අධ්‍යාපන")
        if "විශ්වවිද්" in lower:
            topic_queries.append("නව විශ්වවිද්‍යාල විද්‍යෝදය විද්‍යාලංකාර පේරාදෙණිය")

        queries.extend(topic_queries)
        return [query for query in self._dedupe_reference_items(queries) if query]

    def _sanitize_extracted_reference(self, value: str) -> str:
        cleaned = (value or "").strip()
        if not cleaned:
            return ""

        disallowed_patterns = [
            r"(?i)\bsyllabus content does not\b",
            r"(?i)\bthis syllabus evidence does not contain\b",
            r"(?i)\busing internal knowledge\b",
            r"(?i)\bthis document does not contain\b",
            r"(?i)\bnot covered\b",
        ]
        for pattern in disallowed_patterns:
            if re.search(pattern, cleaned):
                return ""

        disallowed_substrings = [
            "අඩංගු නොවේ",
            "සඳහන් නොවේ",
            "සපයා නැත",
            "මෙම සාක්ෂි",
            "ලබා දී ඇති සාක්ෂි",
            "මෙම සාධක",
            "මෙම ලේඛනයේ",
            "--- PAGE",
            "--- TABLE",
            "[Context Clipped]",
        ]
        if any(marker in cleaned for marker in disallowed_substrings):
            return ""

        return cleaned[:1500]

    def _sanitize_extracted_reference(self, value: str) -> str:
        cleaned = (value or "").strip()
        if not cleaned:
            return ""

        disallowed_patterns = [
            r"(?i)\bsyllabus content does not\b",
            r"(?i)\bthis syllabus evidence does not contain\b",
            r"(?i)\busing internal knowledge\b",
            r"(?i)\bthis document does not contain\b",
            r"(?i)\bnot covered\b",
        ]
        for pattern in disallowed_patterns:
            if re.search(pattern, cleaned):
                return ""

        disallowed_substrings = [
            "\u0d85\u0da9\u0d82\u0d9c\u0dd4 \u0db1\u0ddc\u0dc0\u0dda",
            "\u0dc3\u0db3\u0dc4\u0db1\u0dca \u0db1\u0ddc\u0dc0\u0dda",
            "\u0dc3\u0db4\u0dba\u0dcf \u0db1\u0dd0\u0dad",
            "\u0db8\u0dd9\u0db8 \u0dc3\u0dcf\u0d9a\u0dca\u0dc2\u0dd2",
            "\u0dbd\u0db6\u0dcf \u0daf\u0dd3 \u0d87\u0dad\u0dd2 \u0dc3\u0dcf\u0d9a\u0dca\u0dc2\u0dd2",
            "\u0db8\u0dd9\u0db8 \u0dc3\u0dcf\u0db0\u0d9a",
            "\u0db8\u0dd9\u0db8 \u0dbd\u0dda\u0d9b\u0db1\u0dba\u0dda",
            "--- PAGE",
            "--- TABLE",
            "[Context Clipped]",
        ]
        if any(marker in cleaned for marker in disallowed_substrings):
            return ""

        return cleaned[:1500]

    def _reference_value_to_text(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            return "\n".join(
                text
                for text in (self._reference_value_to_text(item).strip() for item in value)
                if text
            )
        if isinstance(value, dict):
            priority_keys = [
                "points",
                "covering_points",
                "key_points",
                "answer",
                "expected_answer",
                "reference",
                "reference_answer",
            ]
            parts: List[str] = []
            for key in priority_keys:
                if key in value:
                    text = self._reference_value_to_text(value.get(key)).strip()
                    if text:
                        parts.append(text)
            if parts:
                return "\n".join(parts)
            fallback_parts = []
            for item in value.values():
                text = self._reference_value_to_text(item).strip()
                if text:
                    fallback_parts.append(text)
            return "\n".join(fallback_parts)
        return str(value)

    def _shape_reference_for_schema(self, value: Any, question_info: Dict[str, Any]) -> str:
        raw_text = self._reference_value_to_text(value)
        if self._is_invalid_schema_reference(raw_text, question_info):
            return ""

        cleaned = self._sanitize_extracted_reference(raw_text)
        if not cleaned:
            return ""
        if self._is_invalid_schema_reference(cleaned, question_info):
            return ""

        if self._is_short_schema_question(question_info):
            shaped = self._compact_short_reference(cleaned, question_info)
            if not self._short_reference_satisfies_required_count(shaped, question_info):
                return ""
        else:
            shaped = self._format_covering_points_reference(cleaned, question_info)

        if self._is_invalid_schema_reference(shaped, question_info):
            return ""
        return shaped

    def _is_invalid_schema_reference(self, text: str, question_info: Optional[Dict[str, Any]] = None) -> bool:
        cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
        if not cleaned:
            return True

        lowered = cleaned.lower()
        invalid_markers = [
            "not_found",
            "not found",
            "reference points for this question will appear here",
            "will appear here",
            "placeholder",
            "[no matching evidence extracted]",
        ]
        if any(marker in lowered for marker in invalid_markers):
            return True

        if len(cleaned) <= 4 and cleaned.endswith("."):
            return True
        if re.fullmatch(r"(?:[A-Za-z\u0D80-\u0DFF]\.\s*){1,4}", cleaned):
            return True
        if re.search(r"(?:^|\s)[A-Za-z\u0D80-\u0DFF]\.$", cleaned):
            return True

        question_text = str((question_info or {}).get("question_text") or "")
        if "විධායක ජනාධිපති" in question_text and "ගොපල්ලව" in cleaned:
            return True
        return False

    def _is_short_schema_question(self, question_info: Dict[str, Any]) -> bool:
        part_name = str(question_info.get("part_name") or "").lower()
        question_type = str(question_info.get("question_type") or "").lower()
        try:
            max_marks = int(question_info.get("max_marks") or 0)
        except (TypeError, ValueError):
            max_marks = 0

        if "paper_i" in part_name or "paper i" in part_name:
            return True
        if max_marks and max_marks <= 2:
            return True
        return question_type in {"short", "short_answer", "mcq"} and max_marks <= 2

    def _compact_short_reference(self, text: str, question_info: Dict[str, Any]) -> str:
        try:
            max_marks = int(question_info.get("max_marks") or 1)
        except (TypeError, ValueError):
            max_marks = 1

        required_items = self._required_short_answer_count(question_info, max_marks)
        max_items = max(1, min(4, required_items or max_marks or 1))
        limit = 180 if max_items == 1 else 260
        items = self._split_reference_items(text)
        if len(items) <= 1:
            items = self._split_reference_sentences(text)
            max_items = 1

        if required_items == 1 and len(items) == 1 and self._question_explicitly_asks_one(question_info):
            items = [self._first_single_answer_candidate(items[0])]

        if not items:
            return ""

        compact_items = [
            self._truncate_reference_item(self._strip_reference_item_marker(item), limit)
            for item in items[:max_items]
        ]
        compact_items = [item for item in compact_items if item]
        compact = "; ".join(compact_items).strip()
        if len(compact) > limit:
            compact = self._truncate_reference_item(compact, limit)
        return compact

    def _required_short_answer_count(self, question_info: Dict[str, Any], max_marks: int = 1) -> int:
        text = re.sub(r"\s+", " ", str(question_info.get("question_text") or "")).lower()

        if any(term in text for term in ["හතර", "හතරක්", "four"]):
            return 4
        if any(term in text for term in ["තුන", "තුනක්", "three"]):
            return 3
        if any(term in text for term in ["දෙක", "දෙකක්", "two"]):
            return 2

        asks_person_and_year = (
            any(term in text for term in ["කවුද", "කවුරුන්", "who"])
            and any(term in text for term in ["වර්ෂ", "වසර", "year"])
        )
        if asks_person_and_year:
            return 2

        asks_definition_and_aim = "යනු" in text and any(term in text for term in ["අරමුණ", "aim", "objective"])
        if asks_definition_and_aim:
            return 2

        if any(term in text for term in ["එක්", "එකක්", "කවුද", "කුමක්ද", "කුමන", "one"]):
            return 1

        return max(1, min(3, max_marks or 1))

    def _question_explicitly_asks_one(self, question_info: Dict[str, Any]) -> bool:
        text = re.sub(r"\s+", " ", str(question_info.get("question_text") or "")).lower()
        return any(term in text for term in ["එක්", "එකක්", "one"])

    def _first_single_answer_candidate(self, text: str) -> str:
        item = str(text or "").strip()
        parts = [part.strip() for part in re.split(r"\s*,\s*", item) if part.strip()]
        if len(parts) <= 1:
            return item
        first = parts[0]
        if len(first) >= 8 and first.count("(") == first.count(")"):
            return first
        return item

    def _short_reference_satisfies_required_count(self, text: str, question_info: Dict[str, Any]) -> bool:
        if not text:
            return False
        required = self._required_short_answer_count(
            question_info,
            int(question_info.get("max_marks") or 1),
        )
        if required <= 1:
            return True

        question_text = str(question_info.get("question_text") or "")
        if (
            required == 2
            and any(term in question_text for term in ["වර්ෂ", "වසර", "year"])
            and re.search(r"\b(?:18|19|20)\d{2}\b", text)
            and len(re.sub(r"\b(?:18|19|20)\d{2}\b", "", text).strip(" -:,;")) >= 5
        ):
            return True

        count = self._reference_item_count(text)
        return count >= required

    def _reference_item_count(self, text: str) -> int:
        items = self._split_reference_items(text)
        if len(items) <= 1:
            items = self._split_reference_sentences(text)
        if len(items) <= 1 and "," in str(text or ""):
            items = [part.strip() for part in str(text or "").split(",") if part.strip()]
        return len([item for item in items if item])

    def _format_covering_points_reference(self, text: str, question_info: Dict[str, Any]) -> str:
        try:
            max_marks = int(question_info.get("max_marks") or 4)
        except (TypeError, ValueError):
            max_marks = 4

        max_points = max(3, min(8, max_marks or 4))
        items = self._split_reference_items(text)
        if len(items) <= 1:
            items = self._split_reference_sentences(text)

        cleaned_items = []
        for item in items:
            cleaned = self._truncate_reference_item(self._strip_reference_item_marker(item), 420)
            if cleaned:
                cleaned_items.append(cleaned)

        cleaned_items = self._dedupe_reference_items(cleaned_items)[:max_points]
        if not cleaned_items:
            return ""
        if len(cleaned_items) == 1:
            return cleaned_items[0]
        return "\n".join(f"{index}. {item}" for index, item in enumerate(cleaned_items, start=1))

    def _split_reference_items(self, text: str) -> List[str]:
        normalized = (text or "").replace("\r", "\n")
        normalized = re.sub(r"\s*[;；]\s*", "\n", normalized)
        normalized = re.sub(r"\s+(?=(?:\d{1,2}|[A-Za-z])[\.)]\s+)", "\n", normalized)
        raw_items = [line.strip() for line in normalized.split("\n") if line.strip()]
        return self._dedupe_reference_items(
            self._strip_reference_item_marker(item)
            for item in raw_items
            if self._strip_reference_item_marker(item)
        )

    def _split_reference_sentences(self, text: str) -> List[str]:
        normalized = re.sub(r"\s+", " ", text or "").strip()
        if not normalized:
            return []

        dot_placeholder = "__REF_DOT__"
        protected = re.sub(
            r"(?<!\S)([A-Za-z\u0D80-\u0DFF]{1,4})\.",
            lambda match: f"{match.group(1)}{dot_placeholder}",
            normalized,
        )
        sentences = [
            item.replace(dot_placeholder, ".").strip()
            for item in re.split(r"(?<=[.!?\u0964])\s+", protected)
            if item.strip()
        ]
        if len(sentences) <= 1 and len(normalized) > 220:
            sentences = [item.strip() for item in re.split(r"\s*;\s*", normalized) if item.strip()]
        return self._dedupe_reference_items(sentences)

    def _strip_reference_item_marker(self, text: str) -> str:
        cleaned = re.sub(
            r"^\s*(?:[-*\u2022]+|\(?\d{1,2}\)?[\.)-]|\(?[A-Za-z]\)?[\.)-])\s*",
            "",
            str(text or ""),
        ).strip()
        cleaned = re.sub(
            r"^(?:answer|expected answer|direct answer|reference answer|covering points?)\s*[:\-]\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        return re.sub(r"\s+", " ", cleaned).strip()

    def _truncate_reference_item(self, text: str, limit: int) -> str:
        cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
        if len(cleaned) <= limit:
            return cleaned
        clipped = cleaned[:limit].rsplit(" ", 1)[0].strip()
        return clipped.rstrip(" ,;:-")

    def _dedupe_reference_items(self, items) -> List[str]:
        seen = set()
        unique_items: List[str] = []
        for item in items:
            cleaned = str(item or "").strip()
            if not cleaned:
                continue
            key = re.sub(r"\s+", " ", cleaned).lower()
            if key in seen:
                continue
            seen.add(key)
            unique_items.append(cleaned)
        return unique_items

    def _parse_reference_response_by_alias(self, data: Any, questions: List[Dict[str, Any]]) -> Dict[str, str]:
        if isinstance(data, list):
            merged = {}
            for item in data:
                if isinstance(item, dict):
                    merged.update(item)
            data = merged
        if not isinstance(data, dict):
            return {}

        alias_to_uuid = {}
        question_by_key = {}
        for index, question in enumerate(questions, start=1):
            idx_key = f"ref_{index}"
            question_by_key[question["key"]] = question
            for alias in self._build_reference_key_aliases(idx_key, question["display_number"]):
                alias_to_uuid[alias] = question["key"]

        result: Dict[str, str] = {}
        for raw_key, value in data.items():
            original_key = alias_to_uuid.get(self._normalize_reference_key(raw_key))
            if not original_key:
                logger.warning(f"[GEMINI_REF] Unexpected key in response: {raw_key}")
                continue

            text_value = self._reference_value_to_text(value).strip()
            if not text_value or text_value.upper() == "NOT_FOUND":
                logger.info(f"[GEMINI_REF] Key={original_key}: NOT_FOUND in syllabus")
                continue

            clean_val = self._shape_reference_for_schema(
                text_value,
                question_by_key.get(original_key, {}),
            )
            if clean_val:
                result[original_key] = clean_val
        return result

    def _trim_reference_evidence_block(self, evidence: str, limit: int = 1400) -> str:
        cleaned = re.sub(r"\n{3,}", "\n\n", (evidence or "").strip())
        if not cleaned:
            return ""
        cleaned = cleaned.replace("[Context Clipped]", "").strip()
        if len(cleaned) <= limit:
            return cleaned

        clipped = cleaned[:limit].rsplit("\n\n", 1)[0].strip()
        if len(clipped) < int(limit * 0.6):
            clipped = cleaned[:limit].rsplit(" ", 1)[0].strip()
        return f"{clipped}\n[Evidence clipped]"

    def _bounded_generate_missing_references(
        self,
        questions: List[Dict[str, Any]],
        syllabus_text: str,
        max_batches: Optional[int] = None,
    ) -> Dict[str, str]:
        if not questions or not syllabus_text.strip():
            return {}

        result: Dict[str, str] = {}
        source = syllabus_text.strip()
        full_source_for_fallback = ""
        batch_size = 8
        batches = [
            questions[index:index + batch_size]
            for index in range(0, len(questions), batch_size)
        ]
        if max_batches is not None:
            batches = batches[:max(0, int(max_batches))]
        if not batches:
            return {}

        for batch_index, batch in enumerate(batches, start=1):
            prompt = """You are creating a golden marking schema from an uploaded Sinhala History syllabus/note.

Your job is to provide the expected answer/covering points for the questions.

STRICT RULES:
1. First, try to extract the answer directly from the provided uploaded-source evidence.
2. If the source does not support the question topic, you MUST write the correct answer using your expert knowledge of the subject.
3. When using your expert knowledge, the answer MUST be fully accurate, hallucination-free, and appropriate for the subject and grade level.
4. Paper I / 1-2 marks: direct short answer only. If the question asks for one item, return exactly one item. If it asks for two/three items, return exactly that many items.
5. Paper II: covering points only, not essay prose.
6. Ignore activities, paper instructions, page headers, and unrelated nearby topics.

OUTPUT: valid JSON only, keyed by the provided ref_N keys.
"""
            if full_source_for_fallback:
                prompt += "\n=== FULL UPLOADED SYLLABUS / NOTE SOURCE ===\n"
                prompt += full_source_for_fallback
                prompt += "\n=== END FULL SOURCE ===\n"
                prompt += "Use the full source above and the focused evidence excerpts below.\n"
            prompt += "\nQuestions needing bounded golden references:\n"

            for index, question in enumerate(batch, start=1):
                try:
                    max_marks = int(question.get("max_marks") or 0)
                except (TypeError, ValueError):
                    max_marks = 0
                evidence = self._trim_reference_evidence_block(
                    self._build_reference_context_block(question, syllabus_text),
                    limit=1800,
                )
                prompt += f"\n--- Key: ref_{index} ---\n"
                prompt += f"Question Number: {question.get('display_number')}\n"
                prompt += f"Paper Part: {question.get('part_name') or 'Unknown'}\n"
                prompt += f"Question: {question.get('question_text')}\n"
                prompt += f"Max Marks: {question.get('max_marks')}\n"
                prompt += f"Required Style: {'covering points only' if max_marks > 2 else 'direct short answer only'}\n"
                if max_marks <= 2:
                    prompt += f"Required Answer Count: {self._required_short_answer_count(question, max_marks)}\n"
                prompt += f"Uploaded Source Evidence:\n{evidence or '[no matching evidence extracted]'}\n"

            try:
                reason = (
                    "bounded_golden_reference_fallback"
                    if len(batches) == 1
                    else f"bounded_golden_reference_fallback_{batch_index}_of_{len(batches)}"
                )
                response_json = gemini_generate_evaluation(
                    prompt,
                    budget=EvaluationGeminiClient.REFERENCE_SCHEMA,
                    json_mode=True,
                    reason=reason,
                )
                if not response_json:
                    continue
                data = self._safe_reference_json_loads(response_json)
                batch_result = self._parse_reference_response_by_alias(data, batch)
                if batch_result:
                    result.update(batch_result)
                    logger.info(
                        "[GEMINI_REF] Bounded golden fallback batch %s/%s generated %s references.",
                        batch_index,
                        len(batches),
                        len(batch_result),
                    )
            except Exception as exc:
                logger.error(f"[GEMINI_REF] Bounded golden fallback batch %s failed: {exc}", batch_index)

        logger.info("[GEMINI_REF] Bounded golden fallback generated %s references.", len(result))
        return result

    def generate_initial_marking_scheme(
        self,
        evaluation_session_id: UUID,
        syllabus_text: str,
        questions: List[Dict[str, Any]]
    ) -> List[MarkingReference]:
        """
        Phase 0 Extra: Extract marking references from Gemini and save for user review.
        'questions' is a list of dicts: {'key': str, 'target': Question|SubQuestion, 'text': str}
        """
        logger.info(f"Generating initial marking scheme for session {evaluation_session_id}")
        
        # 1. Build enrichment data for extraction
        extraction_questions = []
        for q_info in questions:
            target = q_info['target']
            q_id = q_info['key']
            
            # Resolve question number
            q_num = ""
            if isinstance(target, Question):
                q_num = target.question_number or ""
            elif isinstance(target, SubQuestion):
                p_num = getattr(target.question, 'question_number', '') if target.question else ''
                q_num = f"{p_num}.{target.label}" if p_num else (target.label or "")
            
            # Resolve max marks
            max_marks = self._resolve_max_marks(target)
            
            extraction_questions.append({
                "key": q_id,
                "target": target,
                "display_number": q_num,
                "question_text": q_info['text'],
                "max_marks": max_marks
            })

        # 2. Extract from Gemini using existing batch logic
        extracted_refs = self._batch_extract_reference_points(extraction_questions, syllabus_text)
        
        # 3. Save to MarkingReference table
        saved_refs = []
        # Clear existing unapproved ones if any
        self.marking_refs.delete_session_references(evaluation_session_id)
        
        for q_enriched in extraction_questions:
            q_id = q_enriched['key']
            q_num = q_enriched['display_number']
            q_text = q_enriched['question_text']
            ref_answer = extracted_refs.get(q_id)
            
            # Find the original target object
            original_q_info = next((it for it in questions if it['key'] == q_id), None)
            target = original_q_info['target'] if original_q_info else None

            ref_obj = self.marking_refs.create_reference(
                evaluation_session_id=evaluation_session_id,
                question_id=target.id if isinstance(target, Question) else None,
                sub_question_id=target.id if isinstance(target, SubQuestion) else None,
                question_number=q_num,
                question_text=q_text,
                reference_answer=ref_answer
            )
            saved_refs.append(ref_obj)
            
        logger.info("========================================================================")
        logger.info(f" MARKING SCHEME SUMMARY FOR SESSION: {evaluation_session_id}")
        for r in saved_refs:
            preview = (r.reference_answer[:70] + "...") if r.reference_answer else "EMPTY"
            logger.info(f" Q{r.question_number}: {preview}")
        logger.info("========================================================================")
            
        return saved_refs

    def _batch_extract_reference_points(
        self,
        questions: List[Dict],
        syllabus_text: str
    ) -> Dict[str, str]:
        """
        Use Gemini to read the syllabus and extract precise marking points for each question.
        
        This REPLACES BM25 keyword search as the primary reference source.
        Gemini does NOT grade — it only identifies what the correct answer should contain.
        The actual scoring is still done by XLM-R (deterministic).
        
        Returns: map of question_key -> reference_answer_text
        """
        if not questions:
            return {}

        logger.info(f"[GEMINI_EXTRACTION] Starting extraction for {len(questions)} questions.")
        logger.info(f"[GEMINI_EXTRACTION] Syllabus text length: {len(syllabus_text)} chars.")
        if not syllabus_text.strip():
            logger.warning("[GEMINI_EXTRACTION] SYLLABUS TEXT IS EMPTY; reference extraction will return no generated facts.")

        base_prompt = """You are a Sinhala education expert. Your task is to READ the attached syllabus/note evidence and EXTRACT a marking schema for each question.

IMPORTANT RULES:
1. Try to extract answers primarily from the attached syllabus/note evidence provided in this prompt.
2. If the evidence does not contain a usable answer, you MUST write the correct answer using your expert knowledge of the subject.
3. When using your own knowledge, the answer MUST be fully accurate, hallucination-free, and appropriate for the grade level.
4. Short-answer schema (Paper I / 1-2 marks): return only the direct expected answer words, name, date, place, event, or short facts. If the question asks for one item, return exactly one item. If it asks for two/three items, return exactly that many items. Do not write explanations or essay sentences.
5. Paper II / essay-question schema: do NOT write a model essay. Return only the required covering points that the student answer should include, as a concise numbered list.
6. Be SPECIFIC — extract or generate actual facts, names, dates, and concepts.
6. Be SPECIFIC — extract or generate actual facts, names, dates, and concepts.
7. Do not add English explanations or meta commentary.







OUTPUT FORMAT — return ONLY a valid JSON object:
{{
  "<key>": "extracted answer points here...",
  "<key2>": "1. point one\n2. point two\n3. point three"
}}

Questions:
"""
        # Removed replace calls to allow expert knowledge
        base_prompt += """

GOLDEN FALLBACK:
- If an exact marking-answer sentence is not found, synthesize the expected reference using your expert knowledge.
- You may combine multiple facts into a short answer or covering-point list.
- Do not use unrelated activity text, page headers, paper instructions, or nearby but different topics.
"""

        full_syllabus_for_chunk = ""

        def process_chunk(chunk: List[Dict], chunk_index: int, total_chunks: int) -> Dict[str, str]:
            chunk_prompt = base_prompt
            if full_syllabus_for_chunk:
                chunk_prompt += (
                    "\n=== ATTACHED SYLLABUS / NOTE SOURCE ===\n"
                    f"{full_syllabus_for_chunk}\n"
                    "=== END SYLLABUS / NOTE SOURCE ===\n"
                    "Use this source for every question below.\n"
                )
            # Robust mapping: index -> UUID
            index_to_uuid = {}
            alias_to_uuid = {}
            question_by_key = {}
            for i, q in enumerate(chunk):
                idx_key = f"ref_{i+1}"
                index_to_uuid[idx_key] = q['key']
                question_by_key[q["key"]] = q
                for alias in self._build_reference_key_aliases(idx_key, q["display_number"]):
                    alias_to_uuid[alias] = q["key"]
                evidence_block = (
                    "Use the attached syllabus/note source above."
                    if full_syllabus_for_chunk
                    else self._build_reference_context_block(q, syllabus_text)
                )
                if not full_syllabus_for_chunk:
                    evidence_block = self._trim_reference_evidence_block(evidence_block, limit=1400)
                chunk_prompt += f"\n--- Key: {idx_key} ---\n"
                chunk_prompt += f"Question Number: {q['display_number']}\n"
                chunk_prompt += f"Paper Part: {q.get('part_name') or 'Unknown'}\n"
                chunk_prompt += f"Question: {q['question_text']}\n"
                chunk_prompt += f"Max Marks: {q['max_marks']}\n"
                chunk_prompt += f"Type: {'essay' if q['max_marks'] > 2 else 'short answer'}\n"
                chunk_prompt += f"Required Schema Style: {'covering points only, not essay prose' if q['max_marks'] > 2 else 'direct short answer only'}\n"
                if q["max_marks"] <= 2:
                    chunk_prompt += f"Required Answer Count: {self._required_short_answer_count(q, q['max_marks'])}\n"
                chunk_prompt += f"Evidence:\n{evidence_block or '[no matching evidence extracted]'}\n"

            try:
                sent_indices = list(index_to_uuid.keys())
                logger.info(f"[GEMINI_REF] Extracting references for surrogate keys: {sent_indices}")

                response_json = gemini_generate_evaluation(
                    chunk_prompt,
                    budget=EvaluationGeminiClient.REFERENCE_SCHEMA,
                    json_mode=True,
                    reason=f"chunk_{chunk_index}_of_{total_chunks}",
                )
                if not response_json:
                    return {}

                clean_json = re.sub(
                    r'^```json\s*|\s*```$', '', response_json.strip(), flags=re.MULTILINE
                )
                logger.info(f"[GEMINI_EXTRACTION] Raw Response for chunk: {clean_json}")
                data = self._safe_reference_json_loads(clean_json)

                if isinstance(data, list):
                    merged = {}
                    for d in data:
                        if isinstance(d, dict):
                            merged.update(d)
                    data = merged

                # Validate and clean results
                result = {}
                if isinstance(data, dict):
                    for idx_key, value in data.items():
                        # Map back to original UUID key
                        original_key = alias_to_uuid.get(self._normalize_reference_key(idx_key))
                        if not original_key:
                            # Fallback: maybe Gemini returned the original key or something else
                            logger.warning(f"[GEMINI_REF] Unexpected key in response: {idx_key}")
                            continue

                        text_value = self._reference_value_to_text(value)
                        if text_value.strip() and text_value.strip().upper() != "NOT_FOUND":
                            clean_val = self._shape_reference_for_schema(
                                text_value,
                                question_by_key.get(original_key, {}),
                            )
                            if clean_val:
                                result[original_key] = clean_val
                                logger.info(
                                    f"[GEMINI_REF] Key={original_key}: extracted {len(clean_val)} chars"
                                )
                        else:
                            logger.info(f"[GEMINI_REF] Key={original_key}: NOT_FOUND in syllabus")

                received_keys = list(result.keys())
                sent_uuids = list(index_to_uuid.values())
                missing_keys = [k for k in sent_uuids if k not in received_keys]
                if missing_keys:
                    logger.warning(f"[GEMINI_REF] Missing UUIDs in response: {missing_keys}")

                return result

            except json.JSONDecodeError as e:
                logger.error(f"[GEMINI_REF] JSON parse error: {e}")
                return {}
            except Exception as e:
                logger.error(f"[GEMINI_REF] Extraction failed: {e}")
                return {}

        max_requests = max(1, int(settings.EVAL_GEMINI_REFERENCE_SCHEMA_MAX_REQUESTS))
        fallback_request_reserve = 2 if max_requests > 3 and len(questions) > 8 else 0
        primary_request_limit = max(1, max_requests - fallback_request_reserve)
        chunks: List[List[Dict]] = []
        if full_syllabus_for_chunk:
            chunks = [[]]
            current_chars = 0
            max_chunk_chars = 7000
            for question in questions:
                estimated_chars = len(question.get("question_text", "")) + 300
                if (
                    chunks[-1]
                    and current_chars + estimated_chars > max_chunk_chars
                    and len(chunks) < primary_request_limit
                ):
                    chunks.append([])
                    current_chars = 0
                chunks[-1].append(question)
                current_chars += estimated_chars
        else:
            questions_per_chunk = max(1, (len(questions) + primary_request_limit - 1) // primary_request_limit)
            chunks = [
                questions[index:index + questions_per_chunk]
                for index in range(0, len(questions), questions_per_chunk)
            ]

        logger.info(
            "[GEMINI_REF] duty_name=reference_extraction request_budget=%s requests_used=%s fallback_used=%s reason=chunk_plan",
            max_requests,
            len(chunks),
            len(chunks) > 1,
        )

        all_refs: Dict[str, str] = {}
        with ThreadPoolExecutor(max_workers=min(len(chunks), max_requests)) as executor:
            future_to_chunk = {
                executor.submit(process_chunk, chunk, idx + 1, len(chunks)): chunk
                for idx, chunk in enumerate(chunks)
            }
            for future in as_completed(future_to_chunk):
                try:
                    result = future.result()
                    if result:
                        all_refs.update(result)
                except Exception as exc:
                    logger.error(f"[GEMINI_REF] Chunk extraction exception: {exc}")

        missing_for_golden = [
            question
            for question in questions
            if question.get("key") and not all_refs.get(question.get("key"))
        ]
        if missing_for_golden and syllabus_text.strip():
            remaining_requests = max(0, max_requests - len(chunks))
            golden_refs = self._bounded_generate_missing_references(
                missing_for_golden,
                syllabus_text,
                max_batches=remaining_requests,
            )
            if golden_refs:
                all_refs.update(golden_refs)

        logger.info("[GEMINI_REF] Total extracted references: %s", len(all_refs))

        # PERSISTENCE: Save extracted references to DB to heal missing context
        if all_refs:
            try:
                updated_count = 0
                for q_info in questions:
                    key = q_info['key']
                    extracted_text = all_refs.get(key)
                    if not extracted_text:
                        continue
                    
                    target = q_info['target']
                    # Only persist if current correct_answer is empty
                    if not getattr(target, 'correct_answer', None):
                        target.correct_answer = extracted_text
                        updated_count += 1
                
                if updated_count > 0:
                    self.db.commit()
                    logger.info(f"[GEMINI_REF] Persisted {updated_count} references to database.")
            except Exception as e:
                self.db.rollback()
                logger.error(f"[GEMINI_REF] Failed to persist references: {e}")

        return all_refs

    def _batch_clean_student_answers(self, mapped_answers: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use Gemini to rapidly clean OCR noise and structural typos from student answers.
        Does NOT rewrite or expand answers. Simply acts as a spell-checker.
        """
        # Filter valid text answers
        to_clean = {
            k: v for k, v in mapped_answers.items()
            if isinstance(v, str) and str(v).strip() and str(v).strip().lower() not in ["null", "none"]
        }

        if not to_clean:
            return mapped_answers

        prompt = """You are a Sinhala OCR corrector. Below is a JSON dictionary where keys are IDs and values are text written by students that has been extracted via OCR.
The text contains spelling mistakes, garbled characters, and structural spacing issues caused by bad OCR.

YOUR TASK:
1. Fix spelling mistakes and garbled characters so that the Sinhala text is readable.
2. DO NOT rewrite the answer.
3. DO NOT expand the answer or add new information.
4. DO NOT evaluate the answer or say if it is correct or wrong.
5. If the text is completely unreadable or blank, leave it as is or return an empty string.

Return ONLY a valid JSON object mirroring the keys provided.

=== STUDENT ANSWERS ===
"""
        prompt += json.dumps(to_clean, ensure_ascii=False, indent=2)
        prompt += "\n\n=== END ===\nReturn pure JSON dictionary only."

        try:
            logger.info(f"[OCR_CLEAN] Batch cleaning {len(to_clean)} answers via Gemini...")
            start_t = time.time()
            res_data = self.gemini.generate_content(prompt, json_mode=True)
            res_text = res_data.get("text") or "{}"
            clean_json = re.sub(r'^```json\s*|\s*```$', '', res_text.strip(), flags=re.MULTILINE)
            cleaned_map = json.loads(clean_json)

            # Re-merge with original
            final_map = dict(mapped_answers)
            for k, original_text in to_clean.items():
                cleaned_text = cleaned_map.get(k)
                if isinstance(cleaned_text, str) and cleaned_text.strip():
                    final_map[k] = cleaned_text
                    if original_text.strip() != cleaned_text.strip():
                        logger.info(f"[OCR_CLEAN_DIFF] Key '{k}'\nOriginal: {original_text}\nCleaned : {cleaned_text}")
                else:
                    final_map[k] = original_text

            logger.info(f"[OCR_CLEAN] Cleaned in {time.time() - start_t:.2f}s")
            return final_map
        except Exception as e:
            logger.error(f"[OCR_CLEAN] Failed to batch clean OCR: {e}")
            return mapped_answers

    def _split_reference_concepts(self, reference_text: str) -> List[str]:
        concepts = [
            re.sub(r"\s+", " ", item).strip(" -:\t\r\n")
            for item in re.split(r"[;；.!?\n।]+", str(reference_text or ""))
        ]
        return [concept for concept in concepts if len(concept) >= 4]

    def _calculate_coverage_score(self, student_text: str, reference_text: str, max_marks: int, student_emb=None):
        """
        Measures how many key concepts from the reference the student covered.

        The denominator is capped to a mark-aware expected concept count instead
        of the full raw reference sentence count. This means a 4-mark question
        expects roughly 4 distinct covered points, while long BM25 contexts do
        not unfairly dilute otherwise relevant answers.

        Previously, BM25 could return 20 long syllabus sentences as context, and a
        student covering all the required points still got coverage ~0.21 because
        the denominator was the full raw sentence count.
        """
        if not reference_text:
            return 0.0, "No reference"

        sentences = self._split_reference_concepts(reference_text)

        if not sentences:
            return 0.0, "No reference sentences"

        s_emb = student_emb if student_emb is not None else _embedding_cache.get(student_text)
        if s_emb is None:
            with ml_semaphore:
                s_emb = get_xlmr_model().encode(student_text, convert_to_tensor=True)

        missing_sentences = [s for s in sentences if s not in _embedding_cache]
        if missing_sentences:
            ensure_sentences_cached(missing_sentences)

        sentence_embs = torch.stack([_embedding_cache[s] for s in sentences])
        sims = _cosine_util().cos_sim(s_emb, sentence_embs)[0]

        # Threshold: how similar must a reference point be to count as covered.
        # Paper II references are usually semicolon-separated covering points.
        # Keep this stricter than topical similarity so merely writing around
        # the same topic does not count as explaining every point.
        threshold = 0.28 if max_marks <= 2 else 0.46
        hits = int((sims >= threshold).sum().item())

        # Cap the denominator to a fair expected concept count for this question.
        # A 4-mark structured essay should normally need about 4 distinct points
        # for full coverage. The previous max_marks * 0.60 cap let 2-3 loose
        # matches count as perfect coverage and inflated Paper II scores.
        expected_concepts = min(len(sentences), max(1, max_marks))
        if max_marks > 2:
            expected_concepts = max(2, expected_concepts)
        ratio = hits / max(1, expected_concepts)

        return min(1.0, ratio), f"{hits}/{len(sentences)} concepts covered"

    def _calculate_explanation_quality(self, student_text: str, reference_text: str, max_marks: int, coverage: float) -> float:
        if max_marks <= 2:
            return 1.0

        student_words = re.findall(r"\w+", str(student_text or ""))
        if not student_words:
            return 0.0

        concepts = self._split_reference_concepts(reference_text)
        expected_concepts = max(1, min(len(concepts), max_marks))
        words_per_expected_concept = len(student_words) / expected_concepts

        explanation_markers = [
            "නිසා", "හේතුවෙන්", "එමගින්", "මෙමගින්", "ඒ නිසා", "බලපෑ", "වැඩි",
            "වර්ධනය", "දියුණු", "ඇති විය", "පිහිටුව", "ක්‍රියාත්මක", "හඳුන්වා",
            "ලැබුණි", "කරන ලදී", "විය", "සඳහා", "තුළින්",
        ]
        marker_hits = sum(1 for marker in explanation_markers if marker in student_text)

        if words_per_expected_concept >= 10 and marker_hits >= 2:
            return 1.0
        if words_per_expected_concept >= 7 and marker_hits >= 1:
            return 0.90
        if words_per_expected_concept >= 5:
            return 0.78
        if coverage >= 0.75:
            return 0.70
        return 0.60


    # ----------------------------------------------------------
    # QUESTION MAPPING
    # ----------------------------------------------------------
    def _build_question_map(self, questions: List[Question]) -> Dict[str, Any]:
        """Build a robust map of question IDs and human-readable identifiers."""
        q_map = {}
        for q in questions:
            q_id_str = str(q.id)
            q_num = str(q.question_number).lower().replace(".", "").strip()
            part_name = getattr(q, 'part_name', '').lower().replace(" ", "_").strip()

            q_map[q_id_str] = q

            if part_name:
                q_map[f"{part_name}_{q_num}"] = q

            if q_num not in q_map:
                q_map[q_num] = q

            for sq in getattr(q, 'sub_questions', []) or []:
                sq_id_str = str(sq.id)
                sq_label = str(sq.label).lower().replace("(", "").replace(")", "").strip()
                composite_key = f"{q_num}{sq_label}"

                q_map[sq_id_str] = sq
                if part_name:
                    q_map[f"{part_name}_{composite_key}"] = sq

                if composite_key not in q_map:
                    q_map[composite_key] = sq

        return q_map

    def _find_matching_question(self, key: str, q_map: Dict[str, Any]):
        norm = key.lower().replace(" ", "").replace(".", "").replace("(", "").replace(")", "")
        return q_map.get(key) or q_map.get(norm)


    # ----------------------------------------------------------
    # SYSTEM SCORING & RUBRIC WEIGHTS
    # ----------------------------------------------------------
    def _get_rubric_weights(self, eval_session) -> Dict[str, float]:
        """Fetch rubric weights. These apply to the system XLM-R scoring only."""
        default_weights = {"semantic": 0.4, "coverage": 0.4, "relevance": 0.2}

        if not eval_session.rubric_id:
            logger.warning(f"No rubric_id for session {eval_session.id}, using default weights")
            return default_weights

        criteria = (
            self.db.query(RubricCriterion)
            .filter(RubricCriterion.rubric_id == eval_session.rubric_id)
            .all()
        )

        if not criteria:
            logger.warning(f"No criteria found for rubric {eval_session.rubric_id}, using defaults")
            return default_weights

        weights = {c.criterion.lower(): c.weight_percentage for c in criteria}

        for key in default_weights:
            if key not in weights:
                weights[key] = default_weights[key]

        return weights


    def _calculate_depth_penalty(self, student_text: str, reference_text: str, max_marks: int, coverage: float = 0.0) -> float:
        """
        Soft penalty for very short essay answers.
        Only applies to Paper II (max_marks > 2).
        Does NOT apply to short answers — a correct 2-word answer is valid for Paper I.
        """
        if max_marks <= 2:
            return 1.0

        if coverage >= 0.95:
            return 1.0

        student_words = len(re.findall(r'\w+', student_text))

        # Expected words could scale with max_marks: ~2 words per mark (concise bullet points)
        expected_min_words = max_marks * 2

        if student_words >= expected_min_words:
            return 1.0

        if student_words < expected_min_words * 0.3:
            return 0.80
        elif student_words < expected_min_words * 0.5:
            return 0.90
        elif student_words < expected_min_words * 0.7:
            return 0.95

        # Reasonable length: no penalty
        return 1.0


    def _calculate_formal_language_penalty(self, student_text: str, max_marks: int) -> float:
        """
        Penalize spoken Sinhala forms (කටවහර) in essay answers.
        Paper II answers in Sinhala expect formal written language (ලිඛිත ව්‍යවහාරය).
        Answers written in informal style lose marks for poor description style.
        """
        if max_marks <= 2:
            return 1.0

        spoken_keywords = {
            "වුණා", "උණා", "කළා", "හැදුවා", "ගත්තා", "ආවා", "ගියා",
            "තියෙනවා", "තිබුණා", "ඉන්නවා", "හිටියා", "දෙනවා", "දුන්නා",
            "කියනවා", "කිව්වා", "කෙරුවා", "ඕන", "ඕනේ", "නෑ", "බෑ", "එක",
            "හැදුණා", "වෙනවා", "ඉන්න", "ගහනවා"
        }

        # Cleaning punctuation to match words accurately
        words = [w.strip('.,!?;:"()[]') for w in student_text.split()]
        words = [w for w in words if w]
        if not words:
            return 1.0

        spoken_count = 0
        for w in words:
            if any(w.endswith(kw) or w == kw for kw in spoken_keywords):
                spoken_count += 1
                
        ratio = spoken_count / len(words)

        # Moderated punishing modifiers for informal essays to prevent catastrophic score drops
        if ratio > 0.15:
            return 0.80
        elif ratio > 0.08:
            return 0.90
        elif ratio > 0.04:
            return 0.95

        return 1.0


    def _calculate_system_score(
        self,
        student_text: str,
        reference_text: str,
        weights: Dict[str, float],
        max_marks: int = 1,
        student_emb=None,
        reference_emb=None
    ) -> float:
        """
        Calculate the system score using XLM-R semantic similarity, coverage, and keyword relevance.

        This is the SOLE marks-producing function. Gemini is never called here.
        Marks flow: student_text → XLM-R embeddings → similarity scores → weighted sum → final ratio.

        Paper I (max_marks <= 2): semantic + relevance blend, optimized for short factual answers.
        Paper II (max_marks > 2): semantic + coverage + relevance, optimized for essay depth.
        """
        if not reference_text or not student_text:
            return 0.0

        semantic = self._semantic_similarity(student_text, reference_text, student_emb, reference_emb, max_marks)
        
        # KEY FIX: Exact Match Boost for Short Answers (MCQs)
        # If student answer matches reference exactly (ignoring dots/spaces), give full marks.
        if max_marks <= 2:
            s_clean = str(student_text).strip().replace(".", "").replace(" ", "").lower()
            r_clean = str(reference_text).strip().replace(".", "").replace(" ", "").lower()
            if s_clean == r_clean and len(s_clean) > 0:
                logger.info(f"[EXACT_MATCH_BOOST] Q{max_marks} marks. Match: {s_clean} == {r_clean}")
                return 1.0

        # KEY FIX: If semantic score is 0, the answer is completely off-topic.
        # Ensure 'relevance' (keyword matching) doesn't award free points to wrong/irrelevant answers.
        if semantic <= 0.0:
            logger.info(f"[SCORE_COMPONENTS] semantic=0.000 relevance=X.XXX max_marks={max_marks} -> SHORT CIRCUIT ZERO")
            return 0.0

        relevance = self._calculate_relevance_score(student_text, reference_text)

        logger.info(
            f"[SCORE_COMPONENTS] semantic={semantic:.3f} relevance={relevance:.3f} "
            f"max_marks={max_marks} | answer='{student_text[:60]}'"
        )

        # ------------------------------------------------------------------
        # ------------------------------------------------------------------
        # Paper I — Short Factual Answers (max_marks <= 2)
        # ------------------------------------------------------------------
        if max_marks <= 2:
            # Consistent blend for ALL short answers — no gold-standard shortcut.
            # Semantic (80%): dominates because XLM-R structural matching is robust
            # Relevance (20%): tied to keyword matching which is noisy in Sinhala
            combined = (0.80 * semantic) + (0.20 * relevance)
            return float(combined)

        # ------------------------------------------------------------------
        # Paper II — Essay / Structured Answers (max_marks > 2)
        # Coverage measures depth: how many key concepts did the student address?
        # Semantic measures topicality: is the answer on the right subject?
        # Relevance measures keyword overlap: does the answer use correct terms?
        # ------------------------------------------------------------------
        coverage, coverage_debug = self._calculate_coverage_score(
            student_text, reference_text, max_marks, student_emb
        )

        logger.info(f"[COVERAGE] {coverage_debug} -> coverage_ratio={coverage:.3f}")

        explanation = self._calculate_explanation_quality(student_text, reference_text, max_marks, coverage)

        # Paper II should reward covered marking points and explanatory depth.
        # Semantic similarity is only topicality; it must not overpower missing
        # covering points from the confirmed schema.
        score = (0.20 * semantic) + (0.55 * coverage) + (0.15 * relevance) + (0.10 * explanation)

        # Apply depth penalty for very short essays
        depth_mult = self._calculate_depth_penalty(student_text, reference_text, max_marks, coverage)
        
        # Apply formal language penalty for spoken-style essays
        style_mult = self._calculate_formal_language_penalty(student_text, max_marks)
        
        score = score * depth_mult * style_mult

        logger.info(
            f"[ESSAY_SCORE] semantic={semantic:.3f} coverage={coverage:.3f} "
            f"relevance={relevance:.3f} explanation={explanation:.3f} "
            f"depth_mult={depth_mult:.2f} style_mult={style_mult:.2f} -> score={score:.3f}"
        )

        return float(score)


    def _calculate_relevance_score(self, student_text: str, reference_text: str) -> float:
        """
        Keyword recall: what fraction of the student's own words also appear in the reference?
        Lenient — single-character Sinhala particles are included.
        """
        if not reference_text or not student_text:
            return 0.0

        def get_keywords(text):
            words = re.findall(r'\w+', text.lower())
            return set([w for w in words if len(w) >= 1])

        student_words = get_keywords(student_text)
        ref_words = get_keywords(reference_text)

        if not ref_words:
            return 1.0

        overlap = len(student_words & ref_words)

        # Short answers: use precision-style recall to avoid penalizing correct concise answers
        if len(student_words) < 10:
            recall = overlap / max(1, len(student_words))
        else:
            target_len = min(len(student_words) + 5, 12 + len(ref_words) * 0.4)
            recall = overlap / max(1, target_len)

        return min(1.0, recall)


    def _resolve_display_number(self, question, q_number: str) -> str:
        """Resolve a human-readable question number."""
        if hasattr(question, "question_number") and question.question_number:
            return str(question.question_number)
        elif hasattr(question, "label") and question.label:
            if hasattr(question, "question") and question.question and question.question.question_number:
                return f"{question.question.question_number}({question.label})"
            else:
                return str(question.label)
        return q_number

    def _score_band_label(self, item: Dict[str, Any]) -> str:
        final_marks = Decimal(str(item.get("final_marks", item.get("awarded_marks", 0))))
        max_marks = Decimal(str(max(1, item.get("max_marks", 1))))
        ratio = float(final_marks / max_marks)
        if ratio >= 0.875:
            return "excellent"
        if ratio >= 0.625:
            return "good"
        if ratio >= 0.375:
            return "partial"
        if ratio > 0:
            return "limited"
        return "not_creditworthy"

    def _build_score_aligned_feedback(self, item: Dict[str, Any]) -> str:
        display = item.get("display_number", "")
        final_marks = item.get("final_marks", item.get("awarded_marks", 0))
        max_marks = item.get("max_marks", 1)
        band = self._score_band_label(item)

        if band == "excellent":
            body = "ලබා දී ඇති ලකුණු අනුව පිළිතුර ඉතා හොඳ මට්ටමේ ඇත. ප්‍රධාන කරුණු බොහෝ දුරට නිවැරදිව ආවරණය කර ඇත."
        elif band == "good":
            body = "ලබා දී ඇති ලකුණු අනුව පිළිතුර බොහෝ දුරට නිවැරදිය. තවත් නිශ්චිත කරුණු කිහිපයක් එකතු කළහොත් පිළිතුර වඩා සම්පූර්ණ වේ."
        elif band == "partial":
            body = "ලබා දී ඇති ලකුණු අනුව පිළිතුර අර්ධ වශයෙන් නිවැරදිය. ප්‍රශ්නයට අදාළ ප්‍රධාන කරුණු තවදුරටත් පැහැදිලි කළ යුතුය."
        elif band == "limited":
            body = "ලබා දී ඇති ලකුණු අනුව පිළිතුර සීමිත මට්ටමේ ඇත. අදාළ කරුණු පැහැදිලිව සහ ප්‍රමාණවත් ලෙස දැක්වීම අවශ්‍ය වේ."
        else:
            body = "ලබා දී ඇති ලකුණු අනුව පිළිතුරට ලකුණු ලබා දීමට ප්‍රමාණවත් අදාළ කරුණු හමු නොවේ."

        return f"**ප්‍රශ්නය {display}** {body} ({final_marks}/{max_marks})"

    def _feedback_contradicts_score(self, feedback: str, item: Dict[str, Any]) -> bool:
        if not feedback:
            return True

        text = str(feedback).lower()
        unavailable_markers = [
            "gemini",
            "ප්‍රතිපෝෂණ ලබා ගත නොහැක",
            "ලබා ගත නොහැක",
        ]
        if any(marker in text for marker in unavailable_markers):
            return True

        final_marks = Decimal(str(item.get("final_marks", item.get("awarded_marks", 0))))
        max_marks = Decimal(str(max(1, item.get("max_marks", 1))))
        ratio = float(final_marks / max_marks)

        says_fully_correct = any(
            marker in text
            for marker in [
                "සම්පූර්ණයෙන්ම නිවැරදි",
                "සම්පූර්ණ නිවැරදි",
                "fully correct",
                "completely correct",
            ]
        )
        says_wrong = any(
            marker in text
            for marker in [
                "සම්පූර්ණයෙන්ම වැරදි",
                "සම්පූර්ණයෙන්ම වැරදියි",
                "නිවැරදි නොවේ",
                "completely wrong",
                "fully wrong",
            ]
        )

        if ratio <= 0.25 and says_fully_correct:
            return True
        if ratio >= 0.75 and says_wrong:
            return True
        return False


    # ----------------------------------------------------------
    # GEMINI FEEDBACK — TEXT ONLY, NEVER MARKS
    # ----------------------------------------------------------
    def _get_batch_feedback_from_gemini(self, items: List[Dict]) -> Dict[str, Dict]:
        """
        Request Sinhala feedback text from Gemini for essay questions only.

        CONTRACT:
        - Input: scored_items with marks already finalized by XLM-R system.
        - Output: map of key -> { feedback: str }
        - Marks are NEVER read from Gemini output. Only "feedback" text is used.
        - This function must never influence or modify awarded_marks.
        """
        if not items:
            return {}

        base_prompt = """
You are a senior examiner specializing in Sinhala history assessment. 
Your role is to write brief, constructive feedback in Sinhala.
The marks are already finalized. DO NOT suggest marks.

FEEDBACK REQUIREMENTS:
1. Explain **WHY** it is correct, partially correct, or wrong based on the Reference Context.
2. For partial answers, clearly state which missing points should have been included.
3. Keep feedback concise (2-3 sentences).
4. Use Markdown formatting: **bold** for emphasis, bullet points if needed.
5. Language: Simple, professional written Sinhala.
6. The final awarded mark is already fixed. Your feedback MUST match that mark.
7. If the final mark is 0, do not say the answer is correct. If the final mark is high, do not say the answer is completely wrong.

OUTPUT FORMAT (Pure JSON only):
{
  "<key>": { "feedback": "**නිගමනය:** [desc] \n- [point1]\n- [point2]" },
  ...
}

Questions to evaluate:
"""

        def process_chunk(chunk: List[Dict]) -> Dict[str, Dict]:
            chunk_prompt = base_prompt
            for item in chunk:
                q_text = getattr(item["target"], "question_text", "") or getattr(
                    item["target"], "sub_question_text", ""
                )
                chunk_prompt += f"\n--- Key: {item['key']} ---\n"
                chunk_prompt += f"Question Number: {item['display_number']}\n"
                chunk_prompt += f"Question: {q_text}\n"
                chunk_prompt += f"Max Marks: {item['max_marks']}\n"
                chunk_prompt += f"Final Awarded Marks: {item.get('final_marks', item['awarded_marks'])}/{item['max_marks']}\n"
                chunk_prompt += f"Score Band: {self._score_band_label(item)}\n"
                chunk_prompt += f"Student Answer: {str(item['student_text'])[:500]}\n"
                chunk_prompt += f"Reference Context: {str(item['reference_text'])[:600]}\n"

            try:
                sent_keys = [it["key"] for it in chunk]
                logger.info(f"Sending Gemini feedback batch for keys: {sent_keys}")

                res_data = self.gemini.generate_content(
                    chunk_prompt,
                    json_mode=True,
                    max_retries=2,
                    model_name=settings.EVAL_GEMINI_QUESTION_PARSING_MODEL,
                )
                response_json = res_data.get("text") or "{}"
                try:
                    clean_json = re.sub(r'^```json\s*|\s*```$', '', response_json.strip(), flags=re.MULTILINE)
                    data = json.loads(clean_json, strict=False)

                    if isinstance(data, list):
                        merged_data = {}
                        for item_dict in data:
                            if isinstance(item_dict, dict):
                                merged_data.update(item_dict)
                        data = merged_data

                    received_keys = list(data.keys()) if isinstance(data, dict) else []
                    missing_keys = [k for k in sent_keys if k not in received_keys]
                    if missing_keys:
                        logger.warning(f"Gemini missed feedback keys: {missing_keys}")

                    # Sanitize feedback text — cap length and remove corruption patterns
                    if isinstance(data, dict):
                        for q_key, q_val in data.items():
                            if isinstance(q_val, dict):
                                fb = q_val.get("feedback", "")
                                if fb:
                                    fb = re.sub(r'(.)\1{10,}', r'\1\1\1', fb)  # Remove repeated chars
                                    q_val["feedback"] = fb[:600]

                    return data
                except Exception as e:
                    logger.error(f"Failed to parse Gemini feedback JSON: {e}. Raw: {response_json}")
                    return {}
            except Exception as e:
                logger.error(f"Gemini feedback batch failed: {e}")
                return {}

        # Smaller chunks recover better when one model is overloaded or denied.
        batch_size = 4
        chunks = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]

        evaluation_data = {}
        for chunk in chunks:
            try:
                result = process_chunk(chunk)
                if result:
                    evaluation_data.update(result)
            except Exception as exc:
                logger.error(f"Gemini feedback chunk failed: {exc}")
            time.sleep(0.5)

        # Build final map — ONLY extract feedback text, never any numeric value
        final_map = {}
        for item in items:
            key = item["key"]
            eval_res_json = evaluation_data.get(key, {})
            raw_feedback = eval_res_json.get("feedback", "(ප්‍රතිපෝෂණ ලබා ගත නොහැක)")

            final_map[key] = {
                "feedback": f"**ප්‍රශ්නය {item['display_number']}** {raw_feedback}"
                # NOTE: No "marks", "score", or numeric value is ever stored here.
                # awarded_marks is set exclusively by _apply_discrete_bands().
            }
            if not raw_feedback or raw_feedback.startswith("("):
                raw_feedback = "ප්‍රතිපෝෂණ ලබා ගත නොහැක. Gemini සේවාව තාවකාලිකව අසාර්ථක විය."
            final_map[key]["feedback"] = f"**ප්‍රශ්නය {item['display_number']}** {raw_feedback}"

        for item in items:
            key = item["key"]
            feedback = (final_map.get(key) or {}).get("feedback")
            if not feedback or self._feedback_contradicts_score(feedback, item):
                logger.warning(
                    "Replacing missing or score-contradictory Gemini feedback for Q%s. final_marks=%s/%s",
                    item["display_number"],
                    item.get("final_marks", item["awarded_marks"]),
                    item["max_marks"],
                )
                final_map[key] = {"feedback": self._build_score_aligned_feedback(item)}

        return final_map


    def _generate_overall_feedback(self, total_score, result_id):
        """Generate a brief overall feedback summary in Sinhala. Text only, no marks logic."""
        try:
            prompt = f"""
Summary Results: Total Score is {total_score}.
Evaluate the performance based on this score and generate a quality overall feedback in Sinhala (2-3 sentences).
High quality feedback highlights strengths and provides specific advice for improvement.
Use Markdown for presentation. Use simple and direct language.
"""
            res_data = self.gemini.generate_content(
                prompt,
                max_retries=2,
                model_name=settings.EVAL_GEMINI_QUESTION_PARSING_MODEL,
            )
            if isinstance(res_data, dict):
                return res_data.get("text") or f"Total Score: {total_score}"
            return f"Total Score: {total_score}"
        except Exception as e:
            logger.error(f"Overall feedback generation failed: {e}")
            return f"Total Score: {total_score}"
