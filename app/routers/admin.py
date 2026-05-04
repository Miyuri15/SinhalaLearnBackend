from datetime import datetime, date, time, timezone
from typing import Optional, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, case
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import require_admin_user
from app.shared.models.api_usage_log import ApiUsageLog


router = APIRouter(
    prefix="/api-usage",
    dependencies=[Depends(require_admin_user)],
)


# -------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------

def parse_date_start(value: Optional[date]) -> Optional[datetime]:
    if value is None:
        return None
    return datetime.combine(value, time.min).replace(tzinfo=timezone.utc)


def parse_date_end(value: Optional[date]) -> Optional[datetime]:
    if value is None:
        return None
    return datetime.combine(value, time.max).replace(tzinfo=timezone.utc)


def apply_usage_filters(
    query,
    from_date: Optional[date] = None,
    to_date: Optional[date] = None,
    provider: Optional[str] = None,
    service_name: Optional[str] = None,
    model_name: Optional[str] = None,
    status_value: Optional[str] = None,
    user_id: Optional[UUID] = None,
    session_id: Optional[UUID] = None,
):
    start_dt = parse_date_start(from_date)
    end_dt = parse_date_end(to_date)

    if start_dt:
        query = query.filter(ApiUsageLog.created_at >= start_dt)

    if end_dt:
        query = query.filter(ApiUsageLog.created_at <= end_dt)

    if provider:
        query = query.filter(ApiUsageLog.provider == provider)

    if service_name:
        query = query.filter(ApiUsageLog.service_name == service_name)

    if model_name:
        query = query.filter(ApiUsageLog.model_name == model_name)

    if status_value:
        query = query.filter(ApiUsageLog.status == status_value)

    if user_id:
        query = query.filter(ApiUsageLog.user_id == user_id)

    if session_id:
        query = query.filter(ApiUsageLog.session_id == session_id)

    return query


def serialize_log(log: ApiUsageLog) -> dict:
    return {
        "id": str(log.id),
        "request_id": log.request_id,
        "provider": log.provider,
        "service_name": log.service_name,
        "model_name": log.model_name,
        "user_id": str(log.user_id) if log.user_id else None,
        "session_id": str(log.session_id) if log.session_id else None,
        "message_id": str(log.message_id) if log.message_id else None,
        "prompt_chars": log.prompt_chars,
        "response_chars": log.response_chars,
        "prompt_tokens": log.prompt_tokens,
        "completion_tokens": log.completion_tokens,
        "total_tokens": log.total_tokens,
        "attempt_number": log.attempt_number,
        "max_retries": log.max_retries,
        "is_retry": log.is_retry,
        "status": log.status,
        "error_type": log.error_type,
        "error_message": log.error_message,
        "duration_ms": log.duration_ms,
        "metadata_json": log.metadata_json,
        "created_at": log.created_at.isoformat() if log.created_at else None,
    }


# -------------------------------------------------------------------
# 1. Summary cards
# -------------------------------------------------------------------

@router.get("/summary")
def get_api_usage_summary(
    from_date: Optional[date] = Query(None),
    to_date: Optional[date] = Query(None),
    provider: Optional[str] = Query(None),
    service_name: Optional[str] = Query(None),
    model_name: Optional[str] = Query(None),
    user_id: Optional[UUID] = Query(None),
    session_id: Optional[UUID] = Query(None),
    db: Session = Depends(get_db),
):
    query = db.query(ApiUsageLog)

    query = apply_usage_filters(
        query=query,
        from_date=from_date,
        to_date=to_date,
        provider=provider,
        service_name=service_name,
        model_name=model_name,
        user_id=user_id,
        session_id=session_id,
    )

    result = query.with_entities(
        func.count(ApiUsageLog.id).label("total_requests"),

        func.sum(
            case((ApiUsageLog.status == "success", 1), else_=0)
        ).label("successful_requests"),

        func.sum(
            case((ApiUsageLog.status == "failed", 1), else_=0)
        ).label("failed_requests"),

        func.sum(
            case((ApiUsageLog.status == "retry", 1), else_=0)
        ).label("retry_requests"),

        func.coalesce(func.sum(ApiUsageLog.prompt_tokens), 0).label("prompt_tokens"),
        func.coalesce(func.sum(ApiUsageLog.completion_tokens), 0).label("completion_tokens"),
        func.coalesce(func.sum(ApiUsageLog.total_tokens), 0).label("total_tokens"),

        func.coalesce(func.sum(ApiUsageLog.prompt_chars), 0).label("prompt_chars"),
        func.coalesce(func.sum(ApiUsageLog.response_chars), 0).label("response_chars"),

        func.coalesce(func.avg(ApiUsageLog.duration_ms), 0).label("avg_duration_ms"),
        func.coalesce(func.max(ApiUsageLog.duration_ms), 0).label("max_duration_ms"),
    ).first()

    total_requests = result.total_requests or 0
    successful_requests = result.successful_requests or 0
    failed_requests = result.failed_requests or 0
    retry_requests = result.retry_requests or 0

    success_rate = 0
    failure_rate = 0

    if total_requests > 0:
        success_rate = round((successful_requests / total_requests) * 100, 2)
        failure_rate = round((failed_requests / total_requests) * 100, 2)

    return {
        "total_requests": total_requests,
        "successful_requests": successful_requests,
        "failed_requests": failed_requests,
        "retry_requests": retry_requests,
        "success_rate": success_rate,
        "failure_rate": failure_rate,
        "prompt_tokens": result.prompt_tokens or 0,
        "completion_tokens": result.completion_tokens or 0,
        "total_tokens": result.total_tokens or 0,
        "prompt_chars": result.prompt_chars or 0,
        "response_chars": result.response_chars or 0,
        "avg_duration_ms": round(float(result.avg_duration_ms or 0), 2),
        "max_duration_ms": round(float(result.max_duration_ms or 0), 2),
    }


# -------------------------------------------------------------------
# 2. Logs table with pagination
# -------------------------------------------------------------------

@router.get("/logs")
def get_api_usage_logs(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    from_date: Optional[date] = Query(None),
    to_date: Optional[date] = Query(None),
    provider: Optional[str] = Query(None),
    service_name: Optional[str] = Query(None),
    model_name: Optional[str] = Query(None),
    status_value: Optional[str] = Query(None, alias="status"),
    user_id: Optional[UUID] = Query(None),
    session_id: Optional[UUID] = Query(None),
    db: Session = Depends(get_db),
):
    query = db.query(ApiUsageLog)

    query = apply_usage_filters(
        query=query,
        from_date=from_date,
        to_date=to_date,
        provider=provider,
        service_name=service_name,
        model_name=model_name,
        status_value=status_value,
        user_id=user_id,
        session_id=session_id,
    )

    total = query.count()

    logs = (
        query
        .order_by(ApiUsageLog.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )

    return {
        "items": [serialize_log(log) for log in logs],
        "page": page,
        "page_size": page_size,
        "total": total,
        "total_pages": (total + page_size - 1) // page_size,
    }


# -------------------------------------------------------------------
# 3. Single log details
# -------------------------------------------------------------------

@router.get("/logs/{log_id}")
def get_api_usage_log_detail(
    log_id: UUID,
    db: Session = Depends(get_db),
):
    log = db.query(ApiUsageLog).filter(ApiUsageLog.id == log_id).first()

    if not log:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API usage log not found",
        )

    return serialize_log(log)


# -------------------------------------------------------------------
# 4. Group by service
# -------------------------------------------------------------------

@router.get("/by-service")
def get_api_usage_by_service(
    from_date: Optional[date] = Query(None),
    to_date: Optional[date] = Query(None),
    provider: Optional[str] = Query(None),
    model_name: Optional[str] = Query(None),
    db: Session = Depends(get_db),
):
    query = db.query(ApiUsageLog)

    query = apply_usage_filters(
        query=query,
        from_date=from_date,
        to_date=to_date,
        provider=provider,
        model_name=model_name,
    )

    rows = (
        query
        .with_entities(
            ApiUsageLog.service_name.label("service_name"),
            func.count(ApiUsageLog.id).label("request_count"),
            func.sum(case((ApiUsageLog.status == "success", 1), else_=0)).label("success_count"),
            func.sum(case((ApiUsageLog.status == "failed", 1), else_=0)).label("failed_count"),
            func.sum(case((ApiUsageLog.status == "retry", 1), else_=0)).label("retry_count"),
            func.coalesce(func.sum(ApiUsageLog.total_tokens), 0).label("total_tokens"),
            func.coalesce(func.avg(ApiUsageLog.duration_ms), 0).label("avg_duration_ms"),
        )
        .group_by(ApiUsageLog.service_name)
        .order_by(func.count(ApiUsageLog.id).desc())
        .all()
    )

    return [
        {
            "service_name": row.service_name,
            "request_count": row.request_count or 0,
            "success_count": row.success_count or 0,
            "failed_count": row.failed_count or 0,
            "retry_count": row.retry_count or 0,
            "total_tokens": row.total_tokens or 0,
            "avg_duration_ms": round(float(row.avg_duration_ms or 0), 2),
        }
        for row in rows
    ]


# -------------------------------------------------------------------
# 5. Group by provider
# -------------------------------------------------------------------

@router.get("/by-provider")
def get_api_usage_by_provider(
    from_date: Optional[date] = Query(None),
    to_date: Optional[date] = Query(None),
    service_name: Optional[str] = Query(None),
    model_name: Optional[str] = Query(None),
    db: Session = Depends(get_db),
):
    query = db.query(ApiUsageLog)

    query = apply_usage_filters(
        query=query,
        from_date=from_date,
        to_date=to_date,
        service_name=service_name,
        model_name=model_name,
    )

    rows = (
        query
        .with_entities(
            ApiUsageLog.provider.label("provider"),
            func.count(ApiUsageLog.id).label("request_count"),
            func.sum(case((ApiUsageLog.status == "success", 1), else_=0)).label("success_count"),
            func.sum(case((ApiUsageLog.status == "failed", 1), else_=0)).label("failed_count"),
            func.sum(case((ApiUsageLog.status == "retry", 1), else_=0)).label("retry_count"),
            func.coalesce(func.sum(ApiUsageLog.total_tokens), 0).label("total_tokens"),
            func.coalesce(func.avg(ApiUsageLog.duration_ms), 0).label("avg_duration_ms"),
        )
        .group_by(ApiUsageLog.provider)
        .order_by(func.count(ApiUsageLog.id).desc())
        .all()
    )

    return [
        {
            "provider": row.provider,
            "request_count": row.request_count or 0,
            "success_count": row.success_count or 0,
            "failed_count": row.failed_count or 0,
            "retry_count": row.retry_count or 0,
            "total_tokens": row.total_tokens or 0,
            "avg_duration_ms": round(float(row.avg_duration_ms or 0), 2),
        }
        for row in rows
    ]


# -------------------------------------------------------------------
# 6. Usage trend
# -------------------------------------------------------------------

@router.get("/trend")
def get_api_usage_trend(
    group_by: Literal["day", "hour", "month"] = Query("day"),
    from_date: Optional[date] = Query(None),
    to_date: Optional[date] = Query(None),
    provider: Optional[str] = Query(None),
    service_name: Optional[str] = Query(None),
    model_name: Optional[str] = Query(None),
    status_value: Optional[str] = Query(None, alias="status"),
    db: Session = Depends(get_db),
):
    query = db.query(ApiUsageLog)

    query = apply_usage_filters(
        query=query,
        from_date=from_date,
        to_date=to_date,
        provider=provider,
        service_name=service_name,
        model_name=model_name,
        status_value=status_value,
    )

    if group_by == "hour":
        period_expr = func.date_trunc("hour", ApiUsageLog.created_at)
    elif group_by == "month":
        period_expr = func.date_trunc("month", ApiUsageLog.created_at)
    else:
        period_expr = func.date_trunc("day", ApiUsageLog.created_at)

    rows = (
        query
        .with_entities(
            period_expr.label("period"),
            func.count(ApiUsageLog.id).label("request_count"),
            func.sum(case((ApiUsageLog.status == "success", 1), else_=0)).label("success_count"),
            func.sum(case((ApiUsageLog.status == "failed", 1), else_=0)).label("failed_count"),
            func.sum(case((ApiUsageLog.status == "retry", 1), else_=0)).label("retry_count"),
            func.coalesce(func.sum(ApiUsageLog.total_tokens), 0).label("total_tokens"),
            func.coalesce(func.avg(ApiUsageLog.duration_ms), 0).label("avg_duration_ms"),
        )
        .group_by(period_expr)
        .order_by(period_expr.asc())
        .all()
    )

    return [
        {
            "period": row.period.isoformat() if row.period else None,
            "request_count": row.request_count or 0,
            "success_count": row.success_count or 0,
            "failed_count": row.failed_count or 0,
            "retry_count": row.retry_count or 0,
            "total_tokens": row.total_tokens or 0,
            "avg_duration_ms": round(float(row.avg_duration_ms or 0), 2),
        }
        for row in rows
    ]