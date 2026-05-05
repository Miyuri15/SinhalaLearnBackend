# scripts/modal_trocr_service.py

import base64
import io
import os
from functools import lru_cache

import modal
from fastapi import Request


MODEL_NAME = os.environ.get("TROCR_MODEL_NAME", "hasindu-k/sinhala-handwritten-notes-v3")

app = modal.App("sinhala-trocr-service")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "accelerate",
        "fastapi[standard]",
        "pillow",
        "sentencepiece",
        "torch",
        "transformers",
    )
)


@lru_cache(maxsize=1)
def load_model():
    import torch
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = TrOCRProcessor.from_pretrained(TROCR_MODEL_NAME)
    model = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL_NAME)
    model.to(device)
    model.eval()
    return processor, model, device


@app.function(
    image=image,
    gpu="T4",
    scaledown_window=180,
    timeout=120,
)
@modal.fastapi_endpoint(method="POST")
async def predict(request: Request):
    import torch
    from fastapi import HTTPException
    from PIL import Image

    auth_token = os.environ.get("TROCR_API_KEY")
    if auth_token:
        authorization = request.headers.get("Authorization", "")
        if authorization != f"Bearer {auth_token}":
            raise HTTPException(status_code=401, detail="Unauthorized")

    payload = await request.json()
    image_base64 = payload.get("image_base64")
    if not image_base64:
        raise HTTPException(status_code=400, detail="image_base64 is required")

    max_new_tokens = int(payload.get("max_new_tokens", 256))
    image_bytes = base64.b64decode(image_base64)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    processor, model, device = load_model()
    pixel_values = processor(
        images=image,
        return_tensors="pt",
    ).pixel_values.to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            pixel_values,
            max_new_tokens=max_new_tokens,
        )

    text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )[0].strip()

    return {"text": text}
