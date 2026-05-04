#app/core/config.py
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # General app info
    APP_NAME: str = "Sinhala Educational Assistant"
    ENV: str = "development"

    # Gemini
    GOOGLE_API_KEY: Optional[str] = None
    GOOGLE_API_KEY_V2: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None
    GEMINI_LIGHT_API_KEY: Optional[str] = None
    EVAL_GEMINI_OCR_CORRECTION_MAX_REQUESTS: int = 1
    EVAL_GEMINI_ANSWER_MAPPING_MAX_REQUESTS: int = 10
    EVAL_GEMINI_REFERENCE_SCHEMA_MAX_REQUESTS: int = 10
    EVAL_GEMINI_OCR_MODEL: str = "gemini-2.5-flash"
    EVAL_GEMINI_QUESTION_PARSING_MODEL: str = "gemini-2.5-flash"
    EVAL_GEMINI_ANSWER_MAPPING_MODEL: str = "gemini-2.5-flash"
    EVAL_GEMINI_REFERENCE_SCHEMA_MODEL: str = "gemini-2.5-flash"

    # Database (optional)
    DATABASE_URL: Optional[str] = None

    # Auth
    JWT_SECRET_KEY: str = "change-me"  # override in .env
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 # 24 hours for development
    REFRESH_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7
    PASSWORD_RESET_TOKEN_EXPIRE_MINUTES: int = 15
    ADMIN_BOOTSTRAP_TOKEN: Optional[str] = None

    # Embedding model for RAG
    MODEL_EMBEDDING_NAME: str = "sentence-transformers/paraphrase-xlm-r-multilingual-v1"

    # HuggingFace Hub token (optional, suppresses rate-limit warnings)
    HF_TOKEN: Optional[str] = None
    LOAD_WHISPER_ON_STARTUP: bool = False

    # Email (SMTP)
    MAIL_MAILER: str = "smtp"
    MAIL_HOST: str = "smtp.gmail.com"
    MAIL_PORT: int = 587
    MAIL_USERNAME: Optional[str] = None
    MAIL_PASSWORD: Optional[str] = None
    MAIL_ENCRYPTION: str = "tls"
    MAIL_FROM_ADDRESS: str = "admin@sinhalalearn.online"
    MAIL_FROM_NAME: str = "Sinhala Educational Assistant"
    FRONTEND_URL: str = "http://localhost:3000"  # for reset link

    # Evaluation Limits
    EVALUATION_LIMIT_NORMAL: int = 100  # 5 evaluations as normal, set to 1 for force subscription testing
    EVALUATION_LIMIT_BASIC: int = 20
    EVALUATION_LIMIT_CLASSROOM: int = 60
    EVALUATION_LIMIT_INSTITUTION: int = -1  # Unlimited
    EVALUATION_LIMIT_DURATION_HOURS: int = 12

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "allow"   # <- IMPORTANT: prevents "extra inputs forbidden" errors


settings = Settings()
