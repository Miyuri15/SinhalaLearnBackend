# app/shared/ai/gemini_client.py

import socket
from google import genai
from google.genai import types

from app.core.config import settings
from app.core.gemini_client import GeminiClient
from app.services.evaluation.gemini_cost_policy import EvaluationGeminiClient, GeminiDutyBudget

import logging
import time

logger = logging.getLogger(__name__)

MODEL_NAME = "gemini-2.5-flash"

# Shared safety settings
SAFETY_SETTINGS = [
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=types.HarmBlockThreshold.BLOCK_NONE,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=types.HarmBlockThreshold.BLOCK_NONE,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=types.HarmBlockThreshold.BLOCK_NONE,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=types.HarmBlockThreshold.BLOCK_NONE,
    ),
]


def gemini_generate(
    prompt: str,
    *,
    json_mode: bool = False,
    model_name: str = None,
    max_retries: int = 3,
) -> str:
    if not prompt or not prompt.strip():
        return ""

    selected_model = model_name or MODEL_NAME

    try:
        result = GeminiClient.generate_content(
            prompt,
            max_retries=max_retries,
            safety_settings=SAFETY_SETTINGS,
            json_mode=json_mode,
            model_name=selected_model,
        )
        print(
            f"DEBUG: gemini_generate called (fixed version). JSON Mode: {json_mode}, Model: {selected_model}"
        )
        return (result.get("text") if isinstance(result, dict) else "") or ""
    except Exception as e:
        print(f"Error during Gemini generation (model: {selected_model}): {e}")
        return ""


def gemini_generate_evaluation(
    prompt: str,
    *,
    budget: GeminiDutyBudget,
    json_mode: bool = False,
    reason: str | None = None,
) -> str:
    return EvaluationGeminiClient.generate_once(
        prompt,
        budget=budget,
        json_mode=json_mode,
        reason=reason,
    )


def gemini_generate_lightweight(content: str, prompt_template: str = None) -> str:
    """
    Use lightweight Gemini model for fast, simple tasks with minimal rate impact.
    Perfect for title generation, quick classifications, etc.
    Handles concurrent request rate limiting automatically.
    """
    if not content or not content.strip():
        return ""

    if prompt_template is None:
        prompt_template = """Act as an educational content classifier.
Task: Create a 4-word title for the content provided below.

Rules:
- Exactly 4 words.
- Use the same language as the input (English or Sinhala).
- No quotes, no periods, no introductory text.

Examples:
Input: "How do I solve quadratic equations using the formula?"
Title: Math Equations Help Guide
Input: "à·à·Šâ€à¶»à·“ à¶½à¶‚à¶šà·à·€à·š à¶»à¶¢à·€à¶»à·”à¶±à·Š à¶œà·à¶± à¶­à·œà¶»à¶­à·”à¶»à·”"
Title: à¶‰à¶­à·’à·„à·à·ƒà¶º à·à·Šâ€à¶»à·“ à¶½à¶‚à¶šà·à·€ à¶»à¶¢à·€à¶»à·”

Input: {content}
Title:"""

    prompt = prompt_template.format(content=content)

    max_retries = 3

    for attempt in range(max_retries):
        try:
            api_key = getattr(settings, "GEMINI_LIGHT_API_KEY", None)
            client = genai.Client(api_key=api_key)

            response = client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="text/plain",
                    safety_settings=[
                        types.SafetySetting(
                            category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                            threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
                        ),
                        types.SafetySetting(
                            category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                            threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
                        ),
                    ],
                ),
            )

            return (response.text or "").strip()
        
        except socket.gaierror as e:
            logger.warning(f"DNS error (getaddrinfo failed): {e}")
            return ""

        except Exception as e:
            error_msg = str(e).lower()
            is_rate_limit = "429" in error_msg or "resource_exhausted" in error_msg

            if is_rate_limit and attempt < max_retries - 1:
                logger.debug(
                    f"Rate limited on lightweight request (attempt {attempt + 1}), waiting 1s..."
                )
                time.sleep(1)
                continue

            logger.warning(f"Lightweight Gemini generation failed: {e}")
            return ""

    return ""
