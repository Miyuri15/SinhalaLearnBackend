# app/services/api_usage_log_service.py

import logging
from datetime import datetime, timezone

from app.shared.models.api_usage_log import ApiUsageLog

from app.core.database import SessionLocal
logger = logging.getLogger(__name__)


class ApiUsageLogService:
    @staticmethod
    def create_log(
        request_id: str,
        provider: str,
        service_name: str,
        status: str,
        model_name: str | None = None,
        user_id=None,
        session_id=None,
        message_id=None,
        prompt_chars: int = 0,
        response_chars: int = 0,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        attempt_number: int = 1,
        max_retries: int = 0,
        is_retry: bool = False,
        error_type: str | None = None,
        error_message: str | None = None,
        duration_ms: float | None = None,
        metadata_json: dict | None = None,
    ):
        db = SessionLocal()

        try:
            log = ApiUsageLog(
                request_id=request_id,
                provider=provider,
                service_name=service_name,
                model_name=model_name,
                user_id=user_id,
                session_id=session_id,
                message_id=message_id,
                prompt_chars=prompt_chars,
                response_chars=response_chars,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                attempt_number=attempt_number,
                max_retries=max_retries,
                is_retry=is_retry,
                status=status,
                error_type=error_type,
                error_message=error_message,
                duration_ms=duration_ms,
                metadata_json=metadata_json,
                created_at=datetime.now(timezone.utc),
            )

            db.add(log)
            db.commit()
            db.refresh(log)

            return log

        except Exception:
            db.rollback()
            logger.exception("Failed to save API usage log")
            return None

        finally:
            db.close()