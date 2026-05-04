# app/main.py

from app.logging_config import setup_logging

setup_logging()

import logging
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from app.routers import (
    websockets,
)
from app.core.whisper_loader import WhisperLoader
from app.components.document_processing.services.classify_text_type_ml import TextTypeClassifierLoader
from app.api.v1.router import api_router
from app.core.config import settings
from app.core.database import Base, engine

logger = logging.getLogger(__name__)

cors_origins = [
    "http://localhost:64792",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

if settings.FRONTEND_URL and settings.FRONTEND_URL not in cors_origins:
    cors_origins.append(settings.FRONTEND_URL)

app = FastAPI(
    title="Sinhala Educational Assistant API",
    version="1.0.0",
    swagger_ui_parameters={"persistAuthorization": True},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(websockets.router)
app.include_router(api_router, prefix="/api/v1")

@app.on_event("startup")
def on_startup():
    logger.info("Starting Sinhala Educational Assistant API...")

    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created/verified successfully")
    except Exception as e:
        logger.warning(
            "Could not connect to database during startup: %s",
            str(e)
        )

    if settings.LOAD_WHISPER_ON_STARTUP:
        try:
            logger.info("Loading Whisper model at startup...")
            WhisperLoader.load()
            logger.info("Whisper model loaded successfully.")
        except Exception as e:
            logger.error("Whisper model failed to load: %s", str(e))
    else:
        logger.info("Skipping Whisper startup load; model will load on first use.")

    try:
        logger.info("Loading MobileNet text type classifier at startup...")
        app.state.text_type_classifier = TextTypeClassifierLoader.load()
        logger.info("MobileNet text type classifier loaded successfully.")
    except Exception as e:
        logger.error("MobileNet text type classifier failed to load: %s", str(e))

def custom_openapi():
    """Add Bearer auth security scheme to OpenAPI and apply to protected endpoints."""
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    # Replace any auto-generated schemes with a single BearerAuth scheme
    openapi_schema.setdefault("components", {})["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
        }
    }

    # Public endpoints that don't require auth
    public_paths = {
        "/api/v1/auth/signup",
        "/api/v1/auth/signin",
        "/api/v1/auth/forgot-password",
        "/api/v1/auth/reset-password",
        "/api/v1/auth/refresh",
        "/",
    }

    # HTTP methods to check
    http_methods = {"get", "post", "put", "delete", "patch", "options", "head"}

    # Apply security only to protected endpoints
    for path, path_item in openapi_schema.get("paths", {}).items():
        for method in http_methods:
            if method in path_item and path not in public_paths:
                operation = path_item[method]
                if isinstance(operation, dict):
                    operation["security"] = [{"BearerAuth": []}]

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi

@app.get("/")
def root():
    return {"status": "OK"}
