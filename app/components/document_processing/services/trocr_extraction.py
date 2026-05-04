"""TrOCR abstraction layer for handwritten Sinhala OCR."""

import base64
import io
import json
import logging
from functools import lru_cache
from typing import Optional
from urllib import request
from urllib.error import HTTPError, URLError

import numpy as np
from PIL import Image

from app.components.document_processing.ocr_config import (
    TROCR_API_KEY,
    TROCR_BACKEND,
    TROCR_DEVICE,
    TROCR_ENDPOINT_URL,
    TROCR_MAX_NEW_TOKENS,
    TROCR_MODEL_NAME,
    TROCR_REQUEST_TIMEOUT_SECONDS,
)

logger = logging.getLogger(__name__)


def _resolve_device() -> str:
    if TROCR_DEVICE and TROCR_DEVICE != "auto":
        return TROCR_DEVICE

    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


@lru_cache(maxsize=1)
def _load_trocr_model():
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    device = _resolve_device()
    logger.info("Loading TrOCR model '%s' on %s", TROCR_MODEL_NAME, device)

    processor = TrOCRProcessor.from_pretrained(TROCR_MODEL_NAME)
    model = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL_NAME)
    model.to(device)
    model.eval()

    return processor, model, device


def _to_pil_rgb(image_crop: np.ndarray) -> Image.Image:
    if image_crop.ndim == 2:
        image = Image.fromarray(image_crop)
    else:
        image = Image.fromarray(image_crop[..., ::-1])
    return image.convert("RGB")


def _image_to_base64_png(image_crop: np.ndarray) -> str:
    buffer = io.BytesIO()
    _to_pil_rgb(image_crop).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _predict_local(image_crop: np.ndarray) -> str:
    import torch

    processor, model, device = _load_trocr_model()
    image = _to_pil_rgb(image_crop)
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            pixel_values,
            max_new_tokens=TROCR_MAX_NEW_TOKENS,
        )

    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


def _predict_http(image_crop: np.ndarray) -> str:
    if not TROCR_ENDPOINT_URL:
        raise ValueError("TROCR_ENDPOINT_URL is required when TROCR_BACKEND=http")

    payload = {
        "image_base64": _image_to_base64_png(image_crop),
        "image_format": "png",
        "model": TROCR_MODEL_NAME,
        "max_new_tokens": TROCR_MAX_NEW_TOKENS,
    }
    body = json.dumps(payload).encode("utf-8")

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if TROCR_API_KEY:
        headers["Authorization"] = f"Bearer {TROCR_API_KEY}"

    http_request = request.Request(
        TROCR_ENDPOINT_URL,
        data=body,
        headers=headers,
        method="POST",
    )

    try:
        with request.urlopen(
            http_request,
            timeout=TROCR_REQUEST_TIMEOUT_SECONDS,
        ) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"TrOCR HTTP backend failed with status {e.code}: {error_body}"
        ) from e
    except URLError as e:
        raise RuntimeError(f"TrOCR HTTP backend request failed: {e}") from e

    text = response_payload.get("text")
    if text is None:
        text = response_payload.get("prediction")

    if text is None:
        raise ValueError("TrOCR HTTP backend response must include `text`")

    return str(text).strip()


def trocr_predict(image_crop: np.ndarray, tess_lang: Optional[str] = None) -> str:
    """Predict handwritten text using the configured TrOCR backend."""
    del tess_lang

    if image_crop is None or image_crop.size == 0:
        return ""
    
    logger.info("Predicting text with TrOCR backend '%s'", TROCR_BACKEND)
    
    if TROCR_BACKEND == "local":
        return _predict_local(image_crop)
    if TROCR_BACKEND in {"http", "modal"}:
        return _predict_http(image_crop)

    raise ValueError(
        f"Unsupported TROCR_BACKEND '{TROCR_BACKEND}'. Use 'local', 'http', or 'modal'."
    )
