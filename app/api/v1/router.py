# app/api/v1/router.py
from fastapi import APIRouter

from app.components.voice_qa.routers.voice_router import router as voice_router
from app.routers import (
    chat_sessions,
    messages,
    resources,
    evaluation,
    rubrics,
    users,
    auth,
    pricing,
    usage,
    admin
)

from app.components.voice_qa.routers.voice_router import router as voice_router
from app.components.evaluation.temporary.temporary_evaluation_router import router as temporary_evaluation_router

api_router = APIRouter()

api_router.include_router(chat_sessions.router, prefix="/chat", tags=["Chat"])
api_router.include_router(messages.router, prefix="/messages", tags=["Messages"])
api_router.include_router(resources.router, prefix="/resources", tags=["Resources"])
api_router.include_router(evaluation.router, prefix="/evaluation", tags=["Evaluation"])
api_router.include_router(rubrics.router, prefix="/rubrics", tags=["Rubrics"])
api_router.include_router(users.router, prefix="/users", tags=["Users"]) 
api_router.include_router(auth.router, prefix="/auth", tags=["Auth"]) 
api_router.include_router(pricing.router, prefix="/pricing", tags=["Pricing"])
api_router.include_router(usage.router, prefix="/usage", tags=["Usage"])
api_router.include_router(admin.router, prefix="/admin", tags=["Admin"])

api_router.include_router(voice_router, prefix="/voice", tags=["Voice Q&A"])

print("API Router configured with all components")
