"""Groq/Gemini API client helpers, retries, and safe chat completion wrapper."""

from __future__ import annotations

import json
import logging
from typing import Any

from openai import APIConnectionError, APIError, OpenAI, RateLimitError as OpenAIRateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from applylytics.config import api_key_help_message, resolve_groq_api_key, settings

logger = logging.getLogger("applylytics")


class RateLimitError(Exception):
    """Raised when Groq returns rate-limit or quota errors (legacy)."""


class ApplylyticsLLMError(Exception):
    """User-facing LLM failure after retries and fallbacks are exhausted."""

    def __init__(self, message: str, *, cause: BaseException | None = None) -> None:
        super().__init__(message)
        self.user_message = message
        self.cause = cause


def _groq_rate_limit_message_text(msg: str) -> bool:
    lo = msg.lower()
    return (
        "error code: 429" in lo
        or "rate limit" in lo
        or "rate_limit_exceeded" in lo
        or "tokens per day" in lo
        or "tokens per minute" in lo
    )


def groq_quota_user_message(exc: BaseException) -> str:
    """Plain-text user message for Groq 429 / token or rate-limit errors."""
    detail = str(exc)
    if isinstance(exc, APIError):
        try:
            body = exc.body
            if isinstance(body, dict):
                err = body.get("error")
                if isinstance(err, dict) and err.get("message"):
                    detail = str(err["message"])
        except Exception:
            pass
    import re

    wait_hint = ""
    m = re.search(r"try again in ([\d.]+)s", detail, re.IGNORECASE)
    if m:
        wait_hint = f" Groq suggests waiting about {float(m.group(1)):.0f} seconds."

    hint = (
        "We retried the request several times with backoff. "
        "Set GROQ_MODEL=llama-3.1-8b-instant in .env, wait for the minute window to reset, "
        "or add a valid GEMINI_API_KEY for fallback. Restart Streamlit after editing .env."
        f"{wait_hint}"
    )
    return f"{detail}\n\n{hint}"


def is_groq_rate_limit_error(obj: Any) -> bool:
    """True for RateLimitError / ApplylyticsLLMError instances or legacy rate-limit payloads."""
    if isinstance(obj, (RateLimitError, ApplylyticsLLMError)):
        return True
    if isinstance(obj, dict):
        if obj.get("rate_limited"):
            return True
        err = obj.get("error")
        if isinstance(err, str):
            return _groq_rate_limit_message_text(err)
        return False
    if isinstance(obj, str):
        return _groq_rate_limit_message_text(obj)
    return False


def groq_api_key() -> str:
    """Return configured Groq API key or raise EnvironmentError."""
    key = resolve_groq_api_key()
    if not key:
        raise EnvironmentError(f"GROQ_API_KEY is not set. {api_key_help_message()}")
    return key


def require_groq_api_key() -> str | None:
    """Return API key or None after showing a Streamlit error (no st.stop)."""
    import streamlit as st

    try:
        return groq_api_key()
    except EnvironmentError as e:
        st.error(str(e))
        return None


def get_groq_client() -> OpenAI:
    """OpenAI-compatible client pointed at Groq (SDK retries disabled; we use tenacity)."""
    return OpenAI(
        api_key=groq_api_key(),
        base_url="https://api.groq.com/openai/v1",
        max_retries=0,
        timeout=120.0,
    )


@retry(
    retry=retry_if_exception_type((OpenAIRateLimitError, APIConnectionError)),
    wait=wait_exponential(multiplier=1, min=1, max=32),
    stop=stop_after_attempt(4),
    reraise=True,
)
def retryable_chat_create(client: OpenAI, **kwargs: Any) -> Any:
    """Groq/Gemini chat completion with exponential backoff (4 attempts)."""
    return client.chat.completions.create(**kwargs)


def safe_groq_chat_create(client: OpenAI, **kwargs: Any) -> tuple[Any, str | None]:
    """
    Call Groq chat completions with retries.

    Returns (response, None) on success, (None, error_message) on non-rate API errors.
    Raises ApplylyticsLLMError when quota is exhausted after retries.
    """
    try:
        return retryable_chat_create(client, **kwargs), None
    except (OpenAIRateLimitError, APIConnectionError) as exc:
        raise ApplylyticsLLMError(
            groq_quota_user_message(exc),
            cause=exc,
        ) from exc
    except APIError as exc:
        status = getattr(exc, "status_code", None)
        if status == 429 or str(status) == "429":
            raise ApplylyticsLLMError(
                groq_quota_user_message(exc),
                cause=exc,
            ) from exc
        if status == 413 or str(status) == "413":
            return None, (
                "Groq API error (413): Request payload too large. "
                "Resume or job description was trimmed automatically on retry; "
                "if this persists, use a shorter PDF or job posting."
            )
        return None, f"Groq API error ({getattr(exc, 'status_code', '?')}): {exc}"


def parse_json_from_ai(raw: str | None) -> dict:
    """Parse JSON from model output; strip markdown fences if present."""
    if not raw or not str(raw).strip():
        return {"error": "Empty AI response"}
    s = str(raw).strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return {"error": "AI returned invalid JSON", "raw": s[:2000]}
