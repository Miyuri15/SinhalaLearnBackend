#app/services/evaluation/grading_service.py

import logging
import re
import json
import time
from math import floor
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
from app.shared.models.evaluation_session import EvaluationSession, PaperConfig
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
    _gemini_ref_cache: Dict[str, Dict[str, str]] = {}
    _gemini_ref_cache_lock = __import__('threading').Lock()

    def __init__(self, db: Session):
        self.db = db
        self.gemini = GeminiClient()
        from app.repositories.evaluation.marking_reference_repository import MarkingReferenceRepository
        self.marking_refs = MarkingReferenceRepository(db)


    # ----------------------------------------------------------
    # MARK RESOLUTION
    # ----------------------------------------------------------
    def _resolve_max_marks(self, question):
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

            max_marks = self._resolve_max_marks(target)
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

            max_marks = self._resolve_max_marks(target)
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
            (PaperConfig.evaluation_session_id == eval_session_id) |
            (PaperConfig.chat_session_id == chat_session_id)
        ).all()

        selection_map = {}
        for c in paper_configs:
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
                    "max_marks": self._resolve_max_marks(target),
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
                part_scores[part_name][main_q_id] = {"marks": [], "attempted": 0}

            if has_student_text:
                part_scores[part_name][main_q_id]["attempted"] += 1

            if is_leaf:
                part_scores[part_name][main_q_id]["marks"].append(final_marks)

        self.db.flush()

        # 4. Final total score using selection rules
        total_score_val = Decimal(0)
        norm_selection_map = {k.lower().replace(" ", "_"): v for k, v in selection_map.items()}

        for part_name, section_map in part_scores.items():
            main_q_entries = []
            for mq_id, data in section_map.items():
                total = sum(data["marks"])
                attempted = data["attempted"]
                main_q_entries.append((total, attempted))

            part_rules = norm_selection_map.get(part_name) or {}
            mode = str(part_rules.get('mode', 'all')).lower()
            # Support both 'any' and 'choose_any' as synonyms
            is_selection_mode = mode in ['any', 'choose_any', 'choose']
            count = part_rules.get('count') or part_rules.get('total') or part_rules.get('choose_any')

            if is_selection_mode and count and len(main_q_entries) > int(count):
                # New selection rule: prefer genuinely attempted main questions;
                # only backfill with unattempted zero-mark questions when the paper
                # requires more selected slots than were actually answered.
                attempted_entries = [e for e in main_q_entries if e[1] > 0]
                unattempted_entries = [e for e in main_q_entries if e[1] <= 0]
                attempted_entries.sort(key=lambda x: (x[0], x[1]), reverse=True)
                unattempted_entries.sort(key=lambda x: (x[0], x[1]), reverse=True)

                selected = attempted_entries[:int(count)]
                if len(selected) < int(count):
                    selected.extend(unattempted_entries[: int(count) - len(selected)])

                selected_totals = [e[0] for e in selected]
                attempted_count = sum(1 for e in selected if e[1] > 0)
                logger.info(
                    f"Part {part_name}: Selected best {count}/{len(main_q_entries)} questions "
                    f"({attempted_count} actually attempted)."
                )
                total_score_val += sum(selected_totals)
            else:
                total_score_val += sum(e[0] for e in main_q_entries)

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
                "max_marks": self._resolve_max_marks(target),
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
                    "question_type": getattr(target, "question_type", "short") or "short",
                }
            )

        cache_key = str(eval_session_id)
        gemini_ref_map: Dict[str, str] = {}
        with GradingService._gemini_ref_cache_lock:
            cached = GradingService._gemini_ref_cache.get(cache_key)

        if cached is not None:
            gemini_ref_map = cached
            logger.info("Using cached marking references for session %s (%s refs).", cache_key, len(gemini_ref_map))
        elif syllabus_text and questions_for_extraction:
            gemini_ref_map = self._batch_extract_reference_points(questions_for_extraction, syllabus_text)
            with GradingService._gemini_ref_cache_lock:
                GradingService._gemini_ref_cache[cache_key] = gemini_ref_map

        final_map: Dict[str, str] = {}
        for target_info in targets:
            target = target_info["target"]
            extracted_reference = gemini_ref_map.get(target_info["key"], "")
            if extracted_reference and not self._is_placeholder_answer(extracted_reference):
                final_map[target_info["key"]] = extracted_reference
                continue

            syllabus_reference = self._get_reference_context(
                target,
                syllabus_text,
                rubric_text,
                prefer_gold_standard=False,
            )
            if syllabus_reference and not self._is_placeholder_answer(syllabus_reference):
                final_map[target_info["key"]] = syllabus_reference
                continue

            final_map[target_info["key"]] = self._get_reference_context(
                target,
                syllabus_text,
                rubric_text,
                prefer_gold_standard=True,
            )
        return final_map

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

        sentences = [s.strip() for s in re.split(r'[.!?\n]', reference_text) if len(s.strip()) > 10]
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
        top_chunks = bm25.get_top_n(search_query.split(), chunks, n=15)
        top_chunks = [chunk for chunk in top_chunks if not self._is_placeholder_answer(chunk)]
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
            chunks = self._split_reference_source_chunks(syllabus_text)
            if chunks:
                search_query = question_text or display_number
                bm25 = BM25Okapi([c.split() for c in chunks])
                top_chunks = bm25.get_top_n(search_query.split(), chunks, n=8)
                top_chunks = [chunk for chunk in top_chunks if not self._is_placeholder_answer(chunk)]
                context = "\n\n".join(top_chunks[:4])

        context = (context or "").strip()
        if len(context) > 2200:
            context = context[:2200] + "... [Context Clipped]"
        return context

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
            logger.warning("[GEMINI_EXTRACTION] SYLLABUS TEXT IS EMPTY! Gemini will fallback to generation mode.")

        base_prompt = """You are a Sinhala education expert. Your task is to READ the syllabus evidence provided for each question and EXTRACT the specific answer points that question is looking for.

IMPORTANT RULES:
1. Use ONLY the syllabus evidence block given for that question.
2. Do NOT use internal knowledge.
3. Do NOT say "not covered", "not in the document", or similar phrases.
4. If the evidence is partial, extract the closest useful answer points from that evidence anyway.
5. For short-answer questions (1-2 marks): provide the exact expected answer in 1-2 Sinhala sentences.
6. Be SPECIFIC — extract or generate actual facts, names, dates, and concepts.
7. Keep the answer grounded in the provided evidence. Do not add English explanations or meta commentary.







OUTPUT FORMAT — return ONLY a valid JSON object:
{{
  "<key>": "extracted answer points here...",
  "<key2>": "1. point one\n2. point two\n3. point three"
}}

Questions:
"""
        base_prompt = base_prompt.replace(
            "extract or generate actual facts, names, dates, and concepts.",
            "extract actual facts, names, dates, and concepts from the evidence block.",
        )
        base_prompt = base_prompt.replace(
            'Do not add English explanations or meta commentary.\n',
            'Do not add English explanations or meta commentary.\n8. If you cannot find a usable answer in the evidence, return the exact string "NOT_FOUND" for that key.\n',
        )

        def process_chunk(chunk: List[Dict], chunk_index: int, total_chunks: int) -> Dict[str, str]:
            chunk_prompt = base_prompt
            # Robust mapping: index -> UUID
            index_to_uuid = {}
            alias_to_uuid = {}
            for i, q in enumerate(chunk):
                idx_key = f"ref_{i+1}"
                index_to_uuid[idx_key] = q['key']
                for alias in self._build_reference_key_aliases(idx_key, q["display_number"]):
                    alias_to_uuid[alias] = q["key"]
                evidence_block = self._build_reference_context_block(q, syllabus_text)
                chunk_prompt += f"\n--- Key: {idx_key} ---\n"
                chunk_prompt += f"Question Number: {q['display_number']}\n"
                chunk_prompt += f"Question: {q['question_text']}\n"
                chunk_prompt += f"Max Marks: {q['max_marks']}\n"
                chunk_prompt += f"Type: {'essay' if q['max_marks'] > 2 else 'short answer'}\n"
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
                data = json.loads(clean_json)

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

                        if isinstance(value, str) and value.strip() and value.strip() != "NOT_FOUND":
                            clean_val = self._sanitize_extracted_reference(value)
                            if clean_val:
                                result[original_key] = clean_val
                                logger.info(
                                    f"[GEMINI_REF] Key={original_key}: extracted {len(clean_val)} chars"
                                )
                        elif isinstance(value, dict):
                            # Handle if Gemini returns nested object
                            text_val = value.get("answer", "") or value.get("points", "") or str(value)
                            clean_val = self._sanitize_extracted_reference(text_val)
                            if clean_val:
                                result[original_key] = clean_val
                        else:
                            logger.info(f"[GEMINI_REF] Key={original_key}: NOT_FOUND in syllabus")

                received_keys = list(result.keys())
                sent_uuids = list(index_to_uuid.values())
                missing_keys = [k for k in sent_uuids if k not in received_keys]
                if missing_keys:
                    logger.warning(f"[GEMINI_REF] Missing UUIDs in response: {missing_keys}")
                    for missing_key in missing_keys:
                        question_info = next((q for q in chunk if q["key"] == missing_key), None)
                        if not question_info:
                            continue
                        fallback_context = self._build_reference_context_block(question_info, syllabus_text)
                        if fallback_context and not self._is_placeholder_answer(fallback_context):
                            result[missing_key] = fallback_context

                return result

            except json.JSONDecodeError as e:
                logger.error(f"[GEMINI_REF] JSON parse error: {e}")
                return {}
            except Exception as e:
                logger.error(f"[GEMINI_REF] Extraction failed: {e}")
                return {}

        max_requests = max(1, int(settings.EVAL_GEMINI_REFERENCE_SCHEMA_MAX_REQUESTS))
        chunks: List[List[Dict]] = [[]]
        current_chars = 0
        max_chunk_chars = 2500
        for question in questions:
            evidence_preview = self._build_reference_context_block(question, syllabus_text)
            estimated_chars = (
                len(question.get("question_text", ""))
                + len(evidence_preview or "")
                + 300
            )
            if (
                chunks[-1]
                and current_chars + estimated_chars > max_chunk_chars
                and len(chunks) < max_requests
            ):
                chunks.append([])
                current_chars = 0
            chunks[-1].append(question)
            current_chars += estimated_chars

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

        logger.info(f"[GEMINI_REF] Total extracted: {len(all_refs)} references")

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

        sentences = re.split(r"(?<=[.!?])\s+", reference_text)
        sentences = [s.strip() for s in sentences if len(s.strip()) >= 4]

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

        # Threshold: how similar must a sentence be to count as "covered"
        # Relaxed from 0.32 to 0.28 to reward pooled concise bullet items without dilution.
        threshold = 0.28 if max_marks <= 2 else 0.28
        hits = int((sims >= threshold).sum().item())

        # Cap the denominator to a fair expected concept count for this question.
        # A 4-mark structured essay should normally need about 4 distinct points
        # for full coverage. The previous max_marks * 0.60 cap let 2-3 loose
        # matches count as perfect coverage and inflated Paper II scores.
        expected_concepts = min(len(sentences), max(1, max_marks))
        ratio = hits / max(1, expected_concepts)

        return min(1.0, ratio), f"{hits}/{len(sentences)} concepts covered"


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

        # Recalibrated blend for Paper II (Fairness Update March 2026)
        if coverage > 0.60:
            # High quality answer: value concepts (coverage) and keywords (relevance) more
            e_s_w, e_c_w, e_r_w = 0.40, 0.45, 0.15
        else:
            # Lower quality or vague: semantic topicality is the safeguard
            e_s_w, e_c_w, e_r_w = 0.60, 0.30, 0.10

        score = (e_s_w * semantic) + (e_c_w * coverage) + (e_r_w * relevance)

        # Apply depth penalty for very short essays
        depth_mult = self._calculate_depth_penalty(student_text, reference_text, max_marks, coverage)
        
        # Apply formal language penalty for spoken-style essays
        style_mult = self._calculate_formal_language_penalty(student_text, max_marks)
        
        score = score * depth_mult * style_mult

        logger.info(
            f"[ESSAY_SCORE] semantic={semantic:.3f} coverage={coverage:.3f} "
            f"relevance={relevance:.3f} depth_mult={depth_mult:.2f} style_mult={style_mult:.2f} -> score={score:.3f}"
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
                    data = json.loads(clean_json)

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
