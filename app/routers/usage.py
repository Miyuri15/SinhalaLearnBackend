# app/routers/usage.py

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import get_current_user
from app.schemas.usage import UsageSummaryResponse
from app.services.usage_service import UsageService
from app.shared.models.user import User

router = APIRouter()


@router.get("/me", response_model=UsageSummaryResponse)
def get_my_usage(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return UsageService(db).get_usage_summary(current_user.id)
