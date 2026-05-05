# app/schemas/evaluation.py

from pydantic import AliasChoices, BaseModel, ConfigDict, Field
from typing import Optional, Dict, List, Union, Any
from uuid import UUID
from datetime import datetime
from decimal import Decimal
from enum import Enum


class EvaluationStatus(str, Enum):
    pending = "pending"
    processing = "processing"
    completed = "completed"
    failed = "failed"


class EvaluationResourceRole(str, Enum):
    syllabus = "syllabus"
    question_paper = "question_paper"
    answer_script = "answer_script"
    rubric = "rubric"


# Evaluation Session Schemas
class EvaluationSessionCreate(BaseModel):
    session_id: UUID
    rubric_id: Optional[UUID] = None


class EvaluationSessionUpdate(BaseModel):
    status: Optional[EvaluationStatus] = None
    rubric_id: Optional[UUID] = None


class EvaluationSessionResponse(BaseModel):
    id: UUID
    session_id: UUID
    rubric_id: Optional[UUID] = None
    status: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


# Evaluation Resource Schemas
class EvaluationResourceAttach(BaseModel):
    resource_id: UUID
    role: EvaluationResourceRole


from app.schemas.resource import ResourceFileResponse
from app.schemas.rubric import RubricResponse

class UserEvaluationContextResponse(BaseModel):
    syllabus: Optional[ResourceFileResponse] = None
    question_paper: Optional[ResourceFileResponse] = None
    rubric: Optional[RubricResponse] = None
    paper_config: Optional[List[Dict]] = None


class StartEvaluationRequest(BaseModel):
    chat_session_id: UUID
    answer_resource_ids: List[UUID]
    run_grading: bool = True


class ProcessDocumentsRequest(BaseModel):
    chat_session_id: UUID
    answer_resource_ids: List[UUID]
    force_reprocess: bool = False


class DocumentProcessingStatus(BaseModel):
    resource_id: UUID
    role: str
    status: str  # "processed", "already_processed", "failed"
    message: Optional[str] = None


class ProcessDocumentsResponse(BaseModel):
    results: List[DocumentProcessingStatus]


class AnswerMappingResponse(BaseModel):
    answer_document_id: UUID
    mapped_answers: Optional[Dict[str, Any]] = None
    extracted_text: Optional[str] = None


class SyllabusContentResponse(BaseModel):
    resource_id: UUID
    extracted_text: Optional[str] = None


class RubricContentResponse(BaseModel):
    resource_id: UUID
    extracted_text: Optional[str] = None


class MarkingSchemaQuestionBase(BaseModel):
    question_id: Optional[UUID] = None
    question_number: str
    question_text: str
    reference_text: str
    max_marks: Optional[int] = None
    part_name: Optional[str] = None


class MarkingSchemaQuestionUpdate(MarkingSchemaQuestionBase):
    id: str


class MarkingSchemaQuestionResponse(MarkingSchemaQuestionUpdate):
    pass


class MarkingSchemaUpdateRequest(BaseModel):
    questions: List[MarkingSchemaQuestionUpdate]


class MarkingSchemaResponse(BaseModel):
    id: UUID
    session_id: UUID
    resource_id: Optional[UUID] = None
    is_confirmed: bool
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    questions: List[MarkingSchemaQuestionResponse] = Field(default_factory=list)


class EvaluationResourceResponse(BaseModel):
    id: UUID
    evaluation_session_id: UUID
    resource_id: UUID
    role: Optional[str] = None

    class Config:
        from_attributes = True


# Paper Config Schemas
class PaperConfigCreate(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    paper_part: Optional[str] = Field(default=None, validation_alias=AliasChoices("paper_part", "paperPart"))
    question_type: Optional[str] = None  # e.g., 'short', 'essay'
    subject_name: Optional[str] = Field(default=None, validation_alias=AliasChoices("subject_name", "subjectName"))
    medium: Optional[str] = None
    total_marks: Optional[int] = Field(default=None, validation_alias=AliasChoices("total_marks", "totalMarks"))
    weightage: Optional[Decimal] = None
    total_main_questions: Optional[int] = Field(default=None, validation_alias=AliasChoices("total_main_questions", "totalMainQuestions"))
    selection_rules: Optional[Dict[str, Any]] = Field(default=None, validation_alias=AliasChoices("selection_rules", "selectionRules"))
    is_confirmed: Optional[bool] = Field(default=False, validation_alias=AliasChoices("is_confirmed", "isConfirmed"))

    def get_question_type(self):
        # Default: Paper/Part I = mcq, II = short, III = essay
        if self.question_type:
            return self.question_type
        if self.paper_part:
            part = self.paper_part.strip().lower()
            if part in ["part i", "paper i"]:
                return "mcq"
            if part in ["part ii", "paper ii"]:
                return "short"
            if part in ["part iii", "paper iii"]:
                return "essay"
        return None


class PaperConfigUpdate(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    paper_part: Optional[str] = Field(default=None, validation_alias=AliasChoices("paper_part", "paperPart"))
    question_type: Optional[str] = None
    subject_name: Optional[str] = Field(default=None, validation_alias=AliasChoices("subject_name", "subjectName"))
    medium: Optional[str] = None
    total_marks: Optional[int] = Field(default=None, validation_alias=AliasChoices("total_marks", "totalMarks"))
    weightage: Optional[Decimal] = None
    total_main_questions: Optional[int] = Field(default=None, validation_alias=AliasChoices("total_main_questions", "totalMainQuestions"))
    selection_rules: Optional[Dict[str, Any]] = Field(default=None, validation_alias=AliasChoices("selection_rules", "selectionRules"))
    is_confirmed: Optional[bool] = Field(default=None, validation_alias=AliasChoices("is_confirmed", "isConfirmed"))


class PaperConfigResponse(BaseModel):
    id: UUID
    chat_session_id: Optional[UUID] = None
    evaluation_session_id: Optional[UUID] = None
    paper_part: Optional[str] = None
    question_type: Optional[str] = None  # e.g., 'mcq', 'short', 'essay'
    subject_name: Optional[str] = None
    medium: Optional[str] = None
    total_marks: Optional[int] = None
    weightage: Optional[Decimal] = None
    total_main_questions: Optional[int] = None
    selection_rules: Optional[Dict[str, Any]] = None
    is_confirmed: Optional[bool] = None
    created_at: datetime

    def get_question_type(self):
        # Default: Paper/Part I = mcq, II = short, III = essay
        if self.question_type:
            return self.question_type
        if self.paper_part:
            part = self.paper_part.strip().lower()
            if part in ["part i", "paper i"]:
                return "mcq"
            if part in ["part ii", "paper ii"]:
                return "short"
            if part in ["part iii", "paper iii"]:
                return "essay"
        return None

    class Config:
        from_attributes = True


# Answer Document Schemas
class AnswerDocumentCreate(BaseModel):
    resource_id: UUID
    student_identifier: Optional[str] = None


class AnswerDocumentResponse(BaseModel):
    id: UUID
    evaluation_session_id: UUID
    resource_id: UUID
    student_identifier: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


# Question Paper Schemas
class QuestionPaperCreate(BaseModel):
    resource_id: UUID
    extracted_text: Optional[str] = None


class QuestionPaperResponse(BaseModel):
    id: UUID
    evaluation_session_id: UUID
    resource_id: UUID
    extracted_text: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


# Sub Question Schemas
class SubQuestionCreate(BaseModel):
    label: Optional[str] = None
    sub_question_text: Optional[str] = None
    max_marks: Optional[int] = None
    question_type: Optional[str] = None  # e.g., 'essay', 'short', 'mcq'
    correct_answer: Optional[str] = None  # For MCQ/short answer


class SubQuestionResponse(BaseModel):
    id: UUID
    question_id: UUID
    parent_sub_question_id: Optional[UUID] = None
    label: Optional[str] = None
    sub_question_text: Optional[str] = None
    max_marks: Optional[int] = None
    question_type: Optional[str] = None  # e.g., 'essay', 'short', 'mcq'
    correct_answer: Optional[str] = None  # For MCQ/short answer
    children: List['SubQuestionResponse'] = []

    class Config:
        from_attributes = True


# Question Schemas
class QuestionCreate(BaseModel):
    question_number: Optional[str] = None
    question_text: Optional[str] = None
    max_marks: Optional[int] = None
    question_type: Optional[str] = None  # e.g., 'essay', 'short', 'mcq'
    correct_answer: Optional[str] = None  # For MCQ/short answer


class QuestionResponse(BaseModel):
    id: UUID
    question_paper_id: UUID
    question_number: Optional[str] = None
    question_text: Optional[str] = None
    max_marks: Optional[int] = None
    question_type: Optional[str] = None  # e.g., 'essay', 'short', 'mcq'
    correct_answer: Optional[str] = None  # For MCQ/short answer
    sub_questions: List[SubQuestionResponse] = Field(default=[], validation_alias="root_sub_questions")

    class Config:
        from_attributes = True


# Evaluation Result Schemas
class EvaluationResultCreate(BaseModel):
    answer_document_id: UUID
    total_score: Optional[Decimal] = None
    overall_feedback: Optional[str] = None


class EvaluationResultResponse(BaseModel):
    id: UUID
    answer_document_id: UUID
    total_score: Optional[Decimal] = None
    overall_feedback: Optional[str] = None
    evaluated_at: datetime

    class Config:
        from_attributes = True


# Question Score Schemas
class QuestionScoreCreate(BaseModel):
    question_id: Optional[UUID] = None
    sub_question_id: Optional[UUID] = None
    awarded_marks: Optional[Decimal] = None
    feedback: Optional[str] = None


class QuestionScoreResponse(BaseModel):
    id: UUID
    evaluation_result_id: UUID
    question_id: Optional[UUID] = None
    sub_question_id: Optional[UUID] = None
    question_label: Optional[str] = None
    main_question_key: Optional[str] = None
    sub_label: Optional[str] = None
    paper_part: Optional[str] = None
    paper_part_display: Optional[str] = None
    is_leaf: Optional[bool] = None
    awarded_marks: Optional[Decimal] = None
    max_marks: Optional[Decimal] = None
    feedback: Optional[str] = None

    class Config:
        from_attributes = True


# Combined Evaluation Result with Details
class EvaluationResultDetail(BaseModel):

    id: UUID
    answer_document_id: UUID
    total_score: Optional[Decimal] = None
    total_max: Optional[float] = None
    percentage_score: Optional[float] = None
    grade: Optional[str] = None
    overall_feedback: Optional[str] = None
    improvement_points: Optional[List[str]] = None
    feedback: Optional[dict] = None  # {overall_feedback, missing_concepts, improvement_points}
    marks_summary: Optional[Dict[str, List[Dict[str, Any]]]] = None
    marking_schema: Optional[Dict[str, Any]] = None
    answer_mapping: Optional[Dict[str, Any]] = None
    evaluated_at: datetime
    question_feedbacks: list[QuestionScoreResponse] = []

    class Config:
        from_attributes = True

