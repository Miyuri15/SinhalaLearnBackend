
import logging
from fastapi import APIRouter, Depends, HTTPException, status
from typing import List, Optional, Dict, Any, Union
from uuid import UUID
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.schemas.evaluation import (
    EvaluationSessionCreate,
    EvaluationSessionUpdate,
    EvaluationSessionResponse,
    EvaluationResourceAttach,
    EvaluationResourceResponse,
    PaperConfigCreate,
    PaperConfigUpdate,
    PaperConfigResponse,
    AnswerDocumentCreate,
    AnswerDocumentResponse,
    QuestionResponse,
    EvaluationResultDetail,
    UserEvaluationContextResponse,
    AnswerMappingResponse,
    SyllabusContentResponse,
    RubricContentResponse,
    StartEvaluationRequest,
    ProcessDocumentsRequest,
    ProcessDocumentsResponse,
    EvaluationResultResponse,
    MarkingSchemaResponse,
    MarkingSchemaUpdateRequest,
)

from app.services.evaluation.evaluation_workflow_service import EvaluationWorkflowService
from app.services.evaluation.user_context_service import UserContextService
from app.core.database import get_db
from app.core.security import get_current_user
from app.shared.models.user import User


router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/answers/{answer_id}/score", response_model=Dict[str, Any])
def get_answer_score(
    answer_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Get only the evaluated score for a given answer document.
    """
    service = EvaluationWorkflowService(db)
    result = service.get_evaluation_result(answer_id, current_user.id)
    if not result:
        raise HTTPException(status_code=404, detail="Evaluation result not found")
    return {
        "answer_document_id": answer_id,
        "total_score": result.get("total_score"),
        "percentage_score": result.get("percentage_score"),
    }

@router.get("/answers/{answer_id}/feedback", response_model=Dict[str, Any])
def get_answer_feedback(
    answer_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Get only the feedback for a given answer document.
    """
    service = EvaluationWorkflowService(db)
    result = service.get_evaluation_result(answer_id, current_user.id)
    if not result:
        raise HTTPException(status_code=404, detail="Evaluation result not found")
    feedback_obj = result.get("feedback", {})
    overall_feedback = feedback_obj.get("overall_feedback")
    improvement_points = feedback_obj.get("improvement_points", [])
    return {
        "answer_document_id": answer_id,
        "overall_feedback": overall_feedback,
        "improvement_points": improvement_points,
        "evaluated_at": result.get("evaluated_at"),
    }

@router.get("/sessions/{evaluation_id}/results", response_model=List[Dict[str, Any]])
def list_evaluation_results(
    evaluation_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Get evaluation results for all answer documents in a session.
    Returns actual and percentage scores for each answer sheet.
    """
    service = EvaluationWorkflowService(db)
    try:
        answer_docs = service.list_answer_documents(evaluation_id, current_user.id)
        logger.debug(f"Session {evaluation_id}: Found {len(answer_docs)} answer documents.")
        results = []
        for ad in answer_docs:
            logger.debug(f"Processing answer document: {ad.id} (student_identifier={getattr(ad, 'student_identifier', None)})")
            result = service.get_evaluation_result(ad.id, current_user.id)
            logger.debug(f"Result for answer document {ad.id}: {result}")
            entry = {
                "answer_document_id": ad.id,
                "student_identifier": getattr(ad, 'student_identifier', None),
                "total_score": None,
                "percentage_score": None,
                "overall_feedback": None,
                "evaluated_at": None,
            }
            if result:
                total_score = result.get("total_score") or 0
                max_marks = result.get("total_max") or 0
                percent = result.get("percentage_score")
                if percent is None and max_marks:
                    percent = float(total_score) / float(max_marks) * 100
                entry.update({
                    "total_score": float(total_score),
                    "percentage_score": round(percent, 2) if percent is not None else None,
                    "overall_feedback": result.get("overall_feedback", None),
                    "evaluated_at": result.get("evaluated_at", None),
                })
            results.append(entry)
        logger.debug(f"Returning results for session {evaluation_id}: {results}")
        return results
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))
    except Exception as exc:
        logger.error(f"Failed to list evaluation results for session {evaluation_id}: {exc}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to list evaluation results")



@router.post("/context/paper-config", response_model=UserEvaluationContextResponse)
def update_active_paper_config(
    config_data: List[dict],
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Update the active paper configuration for the user.
    """
    service = UserContextService(db)
    service.update_paper_config(current_user.id, config_data)
    return service.get_context_details(current_user.id)


@router.post("/process-documents", response_model=ProcessDocumentsResponse)
def process_documents(
    payload: ProcessDocumentsRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Process the syllabus, question paper, and answer sheets before evaluation.
    Checks if they are already processed.
    """
    service = EvaluationWorkflowService(db)
    try:
        return service.process_documents(
            chat_session_id=payload.chat_session_id,
            answer_resource_ids=payload.answer_resource_ids,
            user_id=current_user.id,
            force_reprocess=payload.force_reprocess,
        )
    except Exception as exc:
        logger.error(f"Failed to process documents: {exc}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to process documents")


@router.post("/process-documents/stream")
def process_documents_stream(
    payload: ProcessDocumentsRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Stream progress updates for document processing (SSE).
    """
    service = EvaluationWorkflowService(db)
    return StreamingResponse(
        service.process_documents_generator(
            chat_session_id=payload.chat_session_id,
            answer_resource_ids=payload.answer_resource_ids,
            user_id=current_user.id,
            force_reprocess=payload.force_reprocess,
        ),
        media_type="text/event-stream"
    )


from fastapi import BackgroundTasks
from app.core.database import SessionLocal

def run_evaluation_background_task(session_id: UUID, answer_resource_ids: List[UUID], user_id: UUID):
    """
    Background task wrapper to handle DB session lifecycle.
    """
    db = SessionLocal()
    try:
        service = EvaluationWorkflowService(db)
        service.execute_evaluation_process(session_id, answer_resource_ids, user_id)
    finally:
        db.close()

@router.post("/start", response_model=EvaluationSessionResponse)
def start_evaluation(
    payload: StartEvaluationRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Start a new evaluation session using the active context and provided answer scripts.
    Processing happens in the background.
    """
    service = EvaluationWorkflowService(db)
    try:
        # Initialize session synchronously
        session = service.initialize_evaluation_session(
            chat_session_id=payload.chat_session_id,
            answer_resource_ids=payload.answer_resource_ids,
            user_id=current_user.id
        )
        
        # Offload processing to background only if requested
        if payload.run_grading:
            background_tasks.add_task(
                run_evaluation_background_task,
                session.id,
                payload.answer_resource_ids,
                current_user.id
            )
        
        return session
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception as exc:
        logger.error(f"Failed to start evaluation: {exc}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to start evaluation")


@router.post("/start/stream")
def start_evaluation_stream(
    payload: StartEvaluationRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Start a new evaluation session and stream progress updates (SSE).
    """
    service = EvaluationWorkflowService(db)
    try:
        # Initialize session synchronously
        session = service.initialize_evaluation_session(
            chat_session_id=payload.chat_session_id,
            answer_resource_ids=payload.answer_resource_ids,
            user_id=current_user.id
        )
        
        # Check if we should actually run the evaluation or just initialize/return session
        if not payload.run_grading:
             import json
             # If we only wanted initialization, we just return the session details in a single generator event
             def init_only_generator():
                 yield f"data: {json.dumps({'step': 'completed', 'progress': 100, 'session': str(session.id)})}\n\n"
             return StreamingResponse(init_only_generator(), media_type="text/event-stream")

        return StreamingResponse(
            service.execute_evaluation_process_generator(
                session_id=session.id,
                answer_resource_ids=payload.answer_resource_ids,
                user_id=current_user.id
            ),
            media_type="text/event-stream"
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception as exc:
        logger.error(f"Failed to start evaluation stream: {exc}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to start evaluation stream")


@router.get("/context", response_model=UserEvaluationContextResponse)
def get_user_evaluation_context(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get the current active evaluation context (Syllabus, Question Paper, Rubric) for the user.
    """
    service = UserContextService(db)
    return service.get_context_details(current_user.id)



@router.post("/sessions", response_model=EvaluationSessionResponse)
def create_evaluation_session(
    payload: EvaluationSessionCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Start a new evaluation session with a rubric.
    """
    service = EvaluationWorkflowService(db)
    try:
        return service.create_session(payload, current_user.id)
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))
    except ValueError as exc:
        error_text = str(exc)
        status_code = status.HTTP_400_BAD_REQUEST if "Marking schema must be confirmed before grading" in error_text else status.HTTP_404_NOT_FOUND
        raise HTTPException(status_code=status_code, detail=error_text)
    except Exception as exc:
        logger.error(f"Failed to create evaluation session for chat {payload.session_id}: {exc}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create evaluation session")


@router.get("/sessions", response_model=List[EvaluationSessionResponse])
def list_evaluation_sessions(
    session_id: Optional[UUID] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    List all evaluation sessions for the current user (or a specific chat session).
    """
    service = EvaluationWorkflowService(db)
    try:
        return service.list_sessions(current_user.id, session_id)
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))
    except Exception as exc:
        logger.error(f"Failed to list evaluation sessions: {exc}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to list evaluation sessions")


@router.get("/sessions/{evaluation_id}", response_model=EvaluationSessionResponse)
def get_evaluation_session(
    evaluation_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Get evaluation session details.
    """
    service = EvaluationWorkflowService(db)
    try:
        return service.get_session(evaluation_id, current_user.id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))


@router.put("/sessions/{evaluation_id}", response_model=EvaluationSessionResponse)
def update_evaluation_session(
    evaluation_id: UUID,
    payload: EvaluationSessionUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Update evaluation session.
    """
    service = EvaluationWorkflowService(db)
    try:
        updated = service.update_session(evaluation_id, payload, current_user.id)
        if not updated:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Evaluation session not found")
        return updated
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))





@router.get("/sessions/{session_id}/syllabus", response_model=SyllabusContentResponse)
def get_syllabus_content(
    session_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Get the extracted content of the syllabus attached to the session.
    """
    service = EvaluationWorkflowService(db)
    try:
        return service.get_syllabus_content(session_id, current_user.id)
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))
    except ValueError as exc:
        error_text = str(exc)
        status_code = status.HTTP_400_BAD_REQUEST if "Marking schema must be confirmed before grading" in error_text else status.HTTP_404_NOT_FOUND
        raise HTTPException(status_code=status_code, detail=error_text)
    except Exception as exc:
        logger.error(f"Failed to get syllabus content: {exc}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to get syllabus content")


@router.get("/sessions/{session_id}/rubric", response_model=RubricContentResponse)
def get_rubric_content(
    session_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Get the extracted content of the rubric (marking scheme) attached to the session.
    """
    service = EvaluationWorkflowService(db)
    try:
        return service.get_rubric_content(session_id, current_user.id)
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except Exception as exc:
        logger.error(f"Failed to get rubric content: {exc}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to get rubric content")


@router.post("/sessions/{session_id}/parse-question-paper")
def parse_question_paper(
    session_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Parse the question paper for the given session and save the structure.
    """
    service = EvaluationWorkflowService(db)
    try:
        qp = service.parse_question_paper(session_id, current_user.id)
        return {"message": "Question paper parsed successfully", "question_paper_id": qp["id"]}
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))
    except Exception as exc:
        logger.error(f"Failed to parse question paper: {exc}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to parse question paper")


@router.get("/sessions/{session_id}/questions", response_model=List[QuestionResponse])
def get_parsed_questions(
    session_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Get parsed questions from question paper for a chat session.
    """
    service = EvaluationWorkflowService(db)
    try:
        return service.get_parsed_questions(session_id, current_user.id)
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))
    except Exception as exc:
        logger.error(f"Failed to fetch parsed questions for session {session_id}: {exc}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to fetch parsed questions")


@router.get("/sessions/{session_id}/marking-schema", response_model=MarkingSchemaResponse)
def get_marking_schema(
    session_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    service = EvaluationWorkflowService(db)
    try:
        return service.get_marking_schema(session_id, current_user.id)
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception as exc:
        logger.error(f"Failed to get marking schema for session {session_id}: {exc}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to get marking schema")


@router.put("/sessions/{session_id}/marking-schema", response_model=MarkingSchemaResponse)
def update_marking_schema(
    session_id: UUID,
    payload: MarkingSchemaUpdateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    service = EvaluationWorkflowService(db)
    try:
        return service.save_marking_schema(session_id, payload, current_user.id)
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception as exc:
        logger.error(f"Failed to save marking schema for session {session_id}: {exc}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to save marking schema")


@router.post("/sessions/{session_id}/marking-schema/confirm", response_model=MarkingSchemaResponse)
def confirm_marking_schema(
    session_id: UUID,
    payload: MarkingSchemaUpdateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    service = EvaluationWorkflowService(db)
    try:
        return service.confirm_marking_schema(session_id, payload, current_user.id)
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception as exc:
        logger.error(f"Failed to confirm marking schema for session {session_id}: {exc}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to confirm marking schema")


@router.delete("/sessions/{session_id}/marking-schema", response_model=Dict[str, bool])
def delete_marking_schema(
    session_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    service = EvaluationWorkflowService(db)
    try:
        deleted = service.delete_marking_schema(session_id, current_user.id)
        return {"deleted": deleted}
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception as exc:
        logger.error(f"Failed to delete marking schema for session {session_id}: {exc}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to delete marking schema")


@router.post("/sessions/{session_id}/paper-config", response_model=List[PaperConfigResponse])
def save_paper_config(
    session_id: UUID,
    payload: List[PaperConfigCreate],
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Save paper marking configuration for a chat session.
    """
    service = EvaluationWorkflowService(db)
    try:
        return service.save_paper_config(session_id, payload, current_user.id)
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception as exc:
        logger.error(f"Failed to save paper config for session {session_id}: {exc}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to save paper config")


@router.get("/sessions/{session_id}/paper-config", response_model=List[PaperConfigResponse])
def get_paper_config(
    session_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Get paper marking configuration for a chat session.
    """
    service = EvaluationWorkflowService(db)
    try:
        config = service.get_paper_config(session_id, current_user.id)
        if not config:
            return []
        return config if isinstance(config, list) else [config]
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))


@router.post("/sessions/{session_id}/paper-config/confirm", response_model=List[PaperConfigResponse])
def confirm_paper_config(
    session_id: UUID,
    payload: Union[PaperConfigUpdate, List[PaperConfigUpdate]],
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Confirm the paper marking configuration for this chat session.
    """
    service = EvaluationWorkflowService(db)
    try:
        configs = service.confirm_paper_config(session_id, payload, current_user.id)
        if not configs:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Paper config not found")
        return configs
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))


@router.post("/sessions/{evaluation_id}/answers", response_model=AnswerDocumentResponse)
def register_answer_document(
    evaluation_id: UUID,
    payload: AnswerDocumentCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Register a student answer script.
    """
    service = EvaluationWorkflowService(db)
    try:
        return service.register_answer_document(evaluation_id, payload, current_user.id)
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except Exception as exc:
        logger.error(f"Failed to register answer for evaluation {evaluation_id}: {exc}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to register answer document")


@router.get("/sessions/{evaluation_id}/answers", response_model=List[AnswerDocumentResponse])
def list_answer_documents(
    evaluation_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    List all answer documents for an evaluation session.
    """
    service = EvaluationWorkflowService(db)
    try:
        return service.list_answer_documents(evaluation_id, current_user.id)
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))
    except Exception as exc:
        logger.error(f"Failed to list answers for evaluation {evaluation_id}: {exc}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to list answer documents")


@router.post("/answers/{answer_id}/parse", response_model=Dict[str, Any])
def parse_answer_document(
    answer_id: UUID,
    force_remap: bool = False,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Parse and map student answers to questions (Debug/Verification endpoint).
    """
    service = EvaluationWorkflowService(db)
    try:
        return service.parse_answer_document(answer_id, current_user.id, force_remap=force_remap)
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except Exception as exc:
        logger.error(f"Failed to parse answer {answer_id}: {exc}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to parse answer document")


@router.post("/answers/{answer_id}/remap", response_model=AnswerMappingResponse)
def remap_answer_document(
    answer_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Force rebuild the student-answer mapping without re-uploading the answer sheet.
    """
    service = EvaluationWorkflowService(db)
    try:
        service.parse_answer_document(answer_id, current_user.id, force_remap=True)
        return service.get_answer_mapping(answer_id, current_user.id)
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except Exception as exc:
        logger.error(f"Failed to remap answer {answer_id}: {exc}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to remap answer document")


@router.get("/answers/{answer_id}/mapping", response_model=AnswerMappingResponse)
def get_answer_mapping(
    answer_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Get the stored AI mapping for an answer document.
    """
    service = EvaluationWorkflowService(db)
    try:
        return service.get_answer_mapping(answer_id, current_user.id)
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except Exception as exc:
        logger.error(f"Failed to get answer mapping: {exc}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to get answer mapping")


@router.get("/answers/{answer_id}/mapping-details")
def get_answer_mapping_details(
    answer_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Get detailed answer mapping with question text and numbering.
    """
    service = EvaluationWorkflowService(db)
    try:
        return service.get_answer_mapping_details(answer_id, current_user.id)
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except Exception as exc:
        logger.error(f"Failed to get answer mapping details: {exc}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to get answer mapping details")


@router.post("/answers/{answer_id}/evaluate/stream")
def evaluate_answer_stream(
    answer_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Evaluate an answer document and stream progress updates (SSE).
    """
    service = EvaluationWorkflowService(db)
    try:
        def sse_generator():
            generator = service.evaluate_answer_generator(answer_id, current_user.id)
            import json
            for stage, message, percent in generator:
                data = {
                    "progress": int(percent) if percent is not None else None,
                    "message": message,
                    "stage": stage,
                    "status": "processing"
                }
                yield json.dumps(data) + "\n"
        
        return StreamingResponse(sse_generator(), media_type="text/event-stream")

    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except Exception as exc:
        logger.error(f"Failed to stream evaluation for answer {answer_id}: {exc}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to start evaluation stream")


@router.post("/answers/{answer_id}/evaluate")
def evaluate_answer(
    answer_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Evaluate an answer document using rubric.
    """
    service = EvaluationWorkflowService(db)
    try:
        result = service.evaluate_answer(answer_id, current_user.id)
        if not result:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Answer not found for evaluation")
        return {"evaluation_result_id": result.id}
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Failed to evaluate answer {answer_id}: {exc}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to evaluate answer")


@router.post("/answers/{answer_id}/generate-feedback", response_model=Dict[str, Any])
def generate_feedback(
    answer_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Trigger on-demand feedback generation for an answer document.
    """
    service = EvaluationWorkflowService(db)
    try:
        result = service.generate_feedback(answer_id, current_user.id)
        if not result:
            raise HTTPException(status_code=404, detail="Evaluation result not found or generation failed")
        return {"message": "Feedback generated successfully", "evaluation_result_id": result.id}
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except Exception as exc:
        logger.error(f"Failed to generate feedback for answer {answer_id}: {exc}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to generate feedback")


@router.get("/answers/{answer_id}/result", response_model=EvaluationResultDetail)
def get_evaluation_result(
    answer_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Get evaluation result for an answer.
    """
    service = EvaluationWorkflowService(db)
    try:
        result = service.get_evaluation_result(answer_id, current_user.id)
        if not result:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Evaluation result not found")
        return result
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))

