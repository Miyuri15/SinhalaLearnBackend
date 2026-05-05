import json
import logging
import re
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy.orm import Session

from app.repositories.evaluation.marking_schema_repository import MarkingSchemaRepository
from app.schemas.evaluation import (
    MarkingSchemaQuestionResponse,
    MarkingSchemaResponse,
    MarkingSchemaUpdateRequest,
)
from app.services.evaluation.evaluation_session_service import EvaluationSessionService
from app.services.resource_service import ResourceService
from app.services.session_resource_service import SessionResourceService
from app.services.chat_session_service import ChatSessionService
from app.shared.models.evaluation_session import MarkingSchema
from app.shared.models.question_papers import Question, SubQuestion

logger = logging.getLogger(__name__)


class MarkingSchemaService:
    SESSION_RESOURCE_LABEL = "marking_schema"

    def __init__(self, db: Session):
        self.db = db
        self.repository = MarkingSchemaRepository(db)
        self.sessions = EvaluationSessionService(db)
        self.resources = ResourceService(db)
        self.session_resources = SessionResourceService(db)
        self.chat_sessions = ChatSessionService(db)

    def get_or_create_schema(self, session_id: UUID, user_id: UUID) -> MarkingSchemaResponse:
        from app.services.evaluation.grading_service import GradingService

        eval_session = self._resolve_eval_session(session_id, user_id)
        schema = self.repository.get_marking_schema_by_session(eval_session.id)
        if schema:
            if not schema.is_confirmed and self._schema_needs_regeneration(schema):
                GradingService.clear_reference_cache(eval_session.id)
                generated_items = self._generate_marking_schema_items(eval_session.id)
                self.repository.replace_marking_schema_items(schema.id, generated_items)
                self._sync_schema_resource(schema.id, eval_session.session_id, user_id)
                logger.info("Regenerated unsafe draft marking schema for session %s", eval_session.id)
                return self._to_response(self.repository.get_marking_schema_by_session(eval_session.id))
            logger.info("Loaded existing marking schema for session %s", eval_session.id)
            return self._to_response(schema)

        generated_items = self._generate_marking_schema_items(eval_session.id)
        schema = self.repository.create_marking_schema(eval_session.id, is_confirmed=False)
        self.repository.replace_marking_schema_items(schema.id, generated_items)
        self._sync_schema_resource(schema.id, eval_session.session_id, user_id)
        logger.info("Generated marking schema for session %s", eval_session.id)
        return self._to_response(self.repository.get_marking_schema_by_session(eval_session.id))

    def save_schema(
        self,
        session_id: UUID,
        payload: Any,
        user_id: UUID,
        confirmed: bool = False,
    ) -> MarkingSchemaResponse:
        from app.services.evaluation.grading_service import GradingService

        eval_session = self._resolve_eval_session(session_id, user_id)
        GradingService.clear_reference_cache(eval_session.id)
        schema = self.repository.get_marking_schema_by_session(eval_session.id)
        if not schema:
            schema = self.repository.create_marking_schema(eval_session.id, is_confirmed=confirmed)

        normalized_items = self._normalize_items(self._extract_questions_payload(payload))
        self.repository.replace_marking_schema_items(schema.id, normalized_items)
        schema = self.repository.update_marking_schema(schema.id, is_confirmed=confirmed)
        self._sync_schema_resource(schema.id, eval_session.session_id, user_id)
        schema = self.repository.get_marking_schema(schema.id)

        if confirmed:
            logger.info("Confirmed marking schema for session %s", eval_session.id)
        else:
            logger.info("Saved draft marking schema for session %s", eval_session.id)

        return self._to_response(schema)

    def confirm_schema(self, session_id: UUID, payload: Any, user_id: UUID) -> MarkingSchemaResponse:
        return self.save_schema(session_id, payload, user_id, confirmed=True)

    def delete_schema(self, session_id: UUID, user_id: UUID) -> bool:
        from app.services.evaluation.grading_service import GradingService

        eval_session = self._resolve_eval_session(session_id, user_id)
        schema = self.repository.get_marking_schema_by_session(eval_session.id)
        if not schema:
            return False

        GradingService.clear_reference_cache(eval_session.id)
        resource_id = schema.resource_id
        deleted = self.repository.delete_marking_schema(eval_session.id)
        self.session_resources.detach_resources_by_label(eval_session.session_id, self.SESSION_RESOURCE_LABEL)
        if resource_id:
            try:
                self.resources.delete_resource(resource_id, user_id)
            except Exception:
                logger.warning("Failed to delete marking schema resource %s for session %s", resource_id, eval_session.id, exc_info=True)

        logger.info("Deleted marking schema for session %s", eval_session.id)
        return deleted

    def ensure_schema_confirmed(self, session_id: UUID, user_id: UUID) -> MarkingSchemaResponse:
        eval_session = self._resolve_eval_session(session_id, user_id)
        schema = self.repository.get_marking_schema_by_session(eval_session.id)
        if not schema:
            logger.info("Blocked grading because marking schema is missing for session %s", eval_session.id)
            raise ValueError("Marking schema must be confirmed before grading")
        if not schema.is_confirmed:
            logger.info("Blocked grading because marking schema not confirmed for session %s", eval_session.id)
            raise ValueError("Marking schema must be confirmed before grading")
        return self._to_response(schema)

    def get_confirmed_reference_map(self, session_id: UUID, user_id: UUID) -> Dict[str, str]:
        schema = self.ensure_schema_confirmed(session_id, user_id)
        reference_map: Dict[str, str] = {}
        for item in schema.questions:
            if item.question_id:
                reference_map[str(item.question_id)] = item.reference_text
            reference_map[item.question_number] = item.reference_text
            normalized_number = self._normalize_question_key(item.question_number)
            reference_map[normalized_number] = item.reference_text
        return reference_map

    def _resolve_eval_session(self, session_id: UUID, user_id: UUID):
        eval_session = self.sessions.get_evaluation_session(session_id)
        if eval_session:
            if not self.chat_sessions.validate_ownership(eval_session.session_id, user_id):
                raise PermissionError("You don't have permission to access this session")
            return eval_session

        if self.chat_sessions.validate_ownership(session_id, user_id):
            eval_sessions = self.sessions.get_evaluation_sessions_by_chat_session(session_id)
            if eval_sessions:
                return sorted(eval_sessions, key=lambda item: item.created_at, reverse=True)[0]

        raise ValueError(f"No evaluation session found for ID {session_id}")

    def _generate_marking_schema_items(self, eval_session_id: UUID) -> List[Dict[str, Any]]:
        from app.services.evaluation.evaluation_workflow_service import EvaluationWorkflowService
        from app.services.evaluation.grading_service import GradingService

        workflow = EvaluationWorkflowService(self.db)
        syllabus_text, rubric_text, questions = workflow._get_evaluation_context(eval_session_id)
        grader = GradingService(self.db)
        leaf_targets = self._flatten_leaf_targets(questions)
        reference_map = grader.build_reference_map_for_targets(
            eval_session_id=eval_session_id,
            syllabus_text=syllabus_text,
            rubric_text=rubric_text,
            targets=leaf_targets,
        )

        items: List[Dict[str, Any]] = []
        for index, target in enumerate(leaf_targets):
            reference_text = reference_map.get(target["key"], "")
            items.append(
                {
                    "question_id": target["question_id"],
                    "question_number": target["question_number"],
                    "question_text": target["question_text"],
                    "reference_text": reference_text,
                    "max_marks": target["max_marks"],
                    "part_name": target["part_name"],
                    "sort_order": index,
                }
            )
        return items

    def _flatten_leaf_targets(self, questions: List[Question]) -> List[Dict[str, Any]]:
        targets: List[Dict[str, Any]] = []
        sort_order = 0

        def add_sub_questions(question: Question, sub_questions: List[SubQuestion], prefix: str):
            nonlocal sort_order
            for sub_question in sorted(sub_questions, key=lambda item: (item.label or "")):
                label = (sub_question.label or "").strip()
                number = f"{prefix}({label})" if prefix and label else prefix or label
                if sub_question.children:
                    add_sub_questions(question, sub_question.children, number)
                    continue

                targets.append(
                    {
                        "key": str(sub_question.id),
                        "question_id": sub_question.id,
                        "question_number": number,
                        "question_text": self._clean_schema_question_text(sub_question.sub_question_text or ""),
                        "max_marks": self._resolve_schema_max_marks(
                            sub_question.max_marks,
                            question.part_name,
                            sub_question.sub_question_text or "",
                        ),
                        "part_name": question.part_name,
                        "target": sub_question,
                        "sort_order": sort_order,
                    }
                )
                sort_order += 1

        for question in questions:
            if question.sub_questions:
                root_children = [sub for sub in question.sub_questions if sub.parent_sub_question_id is None]
                add_sub_questions(question, root_children, str(question.question_number or ""))
                continue

            targets.append(
                {
                    "key": str(question.id),
                    "question_id": question.id,
                    "question_number": str(question.question_number or ""),
                    "question_text": self._clean_schema_question_text(question.question_text or ""),
                    "max_marks": self._resolve_schema_max_marks(
                        question.max_marks,
                        question.part_name,
                        question.question_text or "",
                    ),
                    "part_name": question.part_name,
                    "target": question,
                    "sort_order": sort_order,
                }
            )
            sort_order += 1

        return targets

    def _clean_schema_question_text(self, text: str) -> str:
        cleaned = str(text or "").replace("\r", " ").replace("\n", " ")
        cleaned = re.sub(r"---\s*PAGE\s+\d+\s*---", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.split(
            r"(?:\bPaper\s*II\b|II\s+කොටස|2\s+කොටස|විග්‍රහාත්මක\s+රචනා\s+ප්‍රශ්න|à·€à·’à¶œà·Š)",
            cleaned,
            maxsplit=1,
        )[0]
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def _resolve_schema_max_marks(self, value: Optional[int], part_name: str, question_text: str) -> Optional[int]:
        text_mark = self._extract_marks_from_text(question_text)
        if text_mark:
            return text_mark

        if self._is_paper_i_part(part_name):
            try:
                parsed_value = int(value) if value else None
            except (TypeError, ValueError):
                parsed_value = None
            if parsed_value in {1, 2}:
                return parsed_value
            return 2

        if value:
            try:
                return int(value)
            except (TypeError, ValueError):
                return value
        return value

    def _extract_marks_from_text(self, text: str) -> Optional[int]:
        patterns = [
            r"\(\s*0?(\d{1,2})\s*(?:ලකුණු|marks|à¶½à¶šà·”à¶«à·”)",
            r"(?:ලකුණු|marks|à¶½à¶šà·”à¶«à·”)\s*0?(\d{1,2})",
        ]
        for pattern in patterns:
            match = re.search(pattern, str(text or ""), flags=re.IGNORECASE)
            if match:
                try:
                    mark = int(match.group(1))
                    return mark if mark > 0 else None
                except (TypeError, ValueError):
                    return None
        return None

    def _is_paper_i_part(self, part_name: str) -> bool:
        normalized = str(part_name or "").strip().lower().replace(" ", "_")
        return normalized in {"paper_i", "part_i", "i"} or "paper_i" in normalized or "part_i" in normalized

    def _schema_needs_regeneration(self, schema: MarkingSchema) -> bool:
        items = self.repository.get_marking_schema_items(schema.id)
        for item in items:
            reference = str(item.reference_text or "")
            question_text = str(item.question_text or "")
            if self._is_paper_i_part(item.part_name) and item.max_marks and item.max_marks > 2:
                return True
            if self._is_paper_i_part(item.part_name) and len(reference) > 260:
                return True
            if re.search(r"NOT_FOUND|Reference points for this question will appear here|will appear here", reference, flags=re.IGNORECASE):
                return True
            if self._is_paper_i_part(item.part_name) and re.fullmatch(r"(?:[A-Za-z\u0D80-\u0DFF]\.\s*){1,4}", reference.strip()):
                return True
            if re.search(r"---\s*PAGE\s+\d+\s*---", question_text, flags=re.IGNORECASE):
                return True
            if re.search(r"---\s*(?:PAGE|TABLE)\b|\[Context Clipped\]", reference, flags=re.IGNORECASE):
                return True
            if any(marker in reference for marker in ["ක්රියාකාරකම", "ක්‍රියාකාරකම", "à¶šà·Šà¶»à·’à¶ºà·à¶šà·à¶»à¶šà¶¸"]):
                return True
        return False

    def _extract_questions_payload(self, payload: Any) -> List[Dict[str, Any]]:
        if isinstance(payload, MarkingSchemaUpdateRequest):
            return [
                question.model_dump() if hasattr(question, "model_dump") else question.dict()
                for question in payload.questions
            ]
        if hasattr(payload, "questions"):
            return [
                question.model_dump() if hasattr(question, "model_dump") else question.dict()
                for question in payload.questions
            ]
        return payload

    def _normalize_items(self, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for index, question in enumerate(questions):
            normalized.append(
                {
                    "question_id": question.get("question_id"),
                    "question_number": str(question.get("question_number") or "").strip(),
                    "question_text": str(question.get("question_text") or "").strip(),
                    "reference_text": str(question.get("reference_text") or "").strip(),
                    "max_marks": question.get("max_marks"),
                    "part_name": question.get("part_name"),
                    "sort_order": index,
                }
            )
        return normalized

    def _sync_schema_resource(self, schema_id: UUID, chat_session_id: UUID, user_id: UUID) -> None:
        schema_model = self.repository.get_marking_schema(schema_id)
        if not schema_model:
            return

        schema_response = self._to_response(schema_model)
        payload = schema_response.model_dump() if hasattr(schema_response, "model_dump") else schema_response.dict()
        serialized = json.dumps(payload, default=str, ensure_ascii=False)

        resource = None
        if schema_model.resource_id:
            resource = self.resources.get_resource(schema_model.resource_id)

        if resource:
            self.resources.update_resource_extracted_text(resource.id, serialized)
        else:
            resource = self.resources.upload_resource(
                user_id=user_id,
                original_filename=f"marking_schema_{chat_session_id}.json",
                storage_path=None,
                mime_type="application/json",
                size_bytes=len(serialized.encode("utf-8")),
                source_type="system",
            )
            self.resources.update_resource_extracted_text(resource.id, serialized)
            schema_model = self.repository.update_marking_schema(schema_id, resource_id=resource.id)

        self.session_resources.upsert_session_resource(
            session_id=chat_session_id,
            resource_id=resource.id,
            label=self.SESSION_RESOURCE_LABEL,
        )
        if schema_model and schema_model.resource_id != resource.id:
            self.repository.update_marking_schema(schema_id, resource_id=resource.id)

    def _to_response(self, schema: MarkingSchema) -> MarkingSchemaResponse:
        items = self.repository.get_marking_schema_items(schema.id)
        return MarkingSchemaResponse(
            id=schema.id,
            session_id=schema.evaluation_session_id,
            resource_id=schema.resource_id,
            is_confirmed=schema.is_confirmed,
            created_at=schema.created_at,
            updated_at=schema.updated_at,
            questions=[
                MarkingSchemaQuestionResponse(
                    id=str(item.id),
                    question_id=item.question_id,
                    question_number=item.question_number,
                    question_text=item.question_text,
                    reference_text=item.reference_text,
                    max_marks=item.max_marks,
                    part_name=item.part_name,
                )
                for item in items
            ],
        )

    def _normalize_question_key(self, value: str) -> str:
        return re.sub(r"[\s().]", "", (value or "").lower())
