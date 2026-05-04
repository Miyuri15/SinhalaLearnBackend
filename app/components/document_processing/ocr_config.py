# app\components\document_processing\ocr_config.py

import os
import logging

logger = logging.getLogger(__name__)

OCR_LANG = "sin+eng"
OCR_CONFIG_EXTRA = ""

TROCR_MODEL_NAME = os.getenv(
    "TROCR_MODEL_NAME",
    "hasindu-k/sinhala-handwritten-notes-v2",
)
TROCR_BACKEND = os.getenv("TROCR_BACKEND", "local").lower()
TROCR_DEVICE = os.getenv("TROCR_DEVICE", "auto")
TROCR_MAX_NEW_TOKENS = int(os.getenv("TROCR_MAX_NEW_TOKENS", "256"))
TROCR_ENDPOINT_URL = os.getenv("TROCR_ENDPOINT_URL", "")
TROCR_API_KEY = os.getenv("TROCR_API_KEY", "")
TROCR_REQUEST_TIMEOUT_SECONDS = float(
    os.getenv("TROCR_REQUEST_TIMEOUT_SECONDS", "30")
)

custom_model_name = "sin_custom_v3"
custom_model_filename = f"{custom_model_name}.traineddata"

utils_dir = os.path.join(os.getcwd(), "utils")
custom_model_path = os.path.join(utils_dir, custom_model_filename)

if os.path.exists(custom_model_path):
    logger.info(f"Custom Tesseract model found: {custom_model_path}")

    OCR_LANG = custom_model_name

    # SET TESSDATA PREFIX
    os.environ["TESSDATA_PREFIX"] = os.path.abspath(utils_dir)

else:
    logger.warning(
        f"Custom Tesseract model not found at {custom_model_path}. Using default language: {OCR_LANG}"
    )
