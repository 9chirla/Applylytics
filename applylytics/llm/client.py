"""Groq API client helpers and safe chat completion wrapper."""

from __future__ import annotations

import json
import logging
from typing import Any

from openai import APIError, OpenAI, RateLimitError as OpenAIRateLimitError

from applylytics.config import api_key_help_message, resolve_groq_api_key, settings

logger = logging.getLogger("applylytics")


class RateLimitError(Exception):
    """Raised when Groq returns rate-limit or quota errors."""


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
    hint = (
        "Groq applies token and request limits per organization. "
        "Wait for the window to reset, upgrade billing, or use a smaller model (e.g. llama-3.1-8b-instant)."
    )
    return f"{detail}\n\n{hint}"


def is_groq_rate_limit_error(obj: Any) -> bool:
    """True for RateLimitError instances or legacy rate-limit payloads."""
    if isinstance(obj, RateLimitError):
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
    """OpenAI-compatible client pointed at Groq."""
    return OpenAI(api_key=groq_api_key(), base_url="https://api.groq.com/openai/v1")


def safe_groq_chat_create(client: OpenAI, **kwargs: Any) -> tuple[Any, str | None]:
    """
    Call Groq chat completions.

    Returns (response, None) on success, (None, error_message) on non-rate API errors.
    Raises RateLimitError on quota / 429 responses.
    """
    try:
        return client.chat.completions.create(**kwargs), None
    except OpenAIRateLimitError as exc:
        raise RateLimitError(groq_quota_user_message(exc)) from exc
    except APIError as exc:
        status = getattr(exc, "status_code", None)
        if status == 429 or str(status) == "429":
            raise RateLimitError(groq_quota_user_message(exc)) from exc
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
