# app/shared/models/api_usage_log.py

import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, String, Integer, Float, Text, DateTime, Boolean
from sqlalchemy.dialects.postgresql import UUID, JSONB

from app.core.database import Base

class ApiUsageLog(Base):
    __tablename__ = "api_usage_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Request tracking
    request_id = Column(String(100), index=True, nullable=False)
    provider = Column(String(50), nullable=False)        # gemini, openai, google_stt, etc.
    service_name = Column(String(100), nullable=False)   # chat_generation, feedback_generation, summary_generation
    model_name = Column(String(100), nullable=True)      # gemini-2.5-flash

    # Related app data
    user_id = Column(UUID(as_uuid=True), nullable=True, index=True)
    session_id = Column(UUID(as_uuid=True), nullable=True, index=True)
    message_id = Column(UUID(as_uuid=True), nullable=True, index=True)

    # Request/response size
    prompt_chars = Column(Integer, default=0)
    response_chars = Column(Integer, default=0)

    # Token usage
    prompt_tokens = Column(Integer, default=0)
    completion_tokens = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)

    # Attempt / retry details
    attempt_number = Column(Integer, default=1)
    max_retries = Column(Integer, default=0)
    is_retry = Column(Boolean, default=False)

    # Status
    status = Column(String(30), nullable=False)          # success, failed, retry, rate_limited
    error_type = Column(String(100), nullable=True)      # rate_limited, overloaded, auth_failed
    error_message = Column(Text, nullable=True)

    # Performance
    duration_ms = Column(Float, nullable=True)

    # Extra flexible data
    metadata_json = Column(JSONB, nullable=True)

    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        index=True,
    )