"""Tests for LLM client JSON parsing and rate-limit handling."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from openai import RateLimitError as OpenAIRateLimitError

from applylytics.llm.client import (
    ApplylyticsLLMError,
    RateLimitError,
    is_groq_rate_limit_error,
    parse_json_from_ai,
    safe_groq_chat_create,
)


def test_parse_json_from_ai_raw_object():
    data = {"name": "Test", "key_skills": ["Python"]}
    assert parse_json_from_ai(json.dumps(data)) == data


def test_parse_json_from_ai_markdown_fence():
    raw = '```json\n{"name": "Test"}\n```'
    assert parse_json_from_ai(raw) == {"name": "Test"}


def test_parse_json_from_ai_invalid_returns_error():
    result = parse_json_from_ai("not json at all")
    assert "error" in result
    assert "raw" in result


def test_is_groq_rate_limit_error_exception_type():
    assert is_groq_rate_limit_error(RateLimitError("quota")) is True


def test_safe_groq_chat_create_raises_applylytics_llm_error():
    client = MagicMock()
    client.chat.completions.create.side_effect = OpenAIRateLimitError(
        "rate limit",
        response=MagicMock(status_code=429),
        body=None,
    )
    with pytest.raises(ApplylyticsLLMError):
        safe_groq_chat_create(client, model="llama-3.3-70b-versatile", messages=[])


def test_safe_groq_retries_fast_model_on_429(monkeypatch):
    """When the primary model is TPM-limited, retry with GROQ_FAST_MODEL."""
    from applylytics.llm import groq_optimizer

    monkeypatch.setattr(groq_optimizer, "get_gemini_client", lambda: None)
    client = MagicMock()
    ok_response = MagicMock()
    rate_exc = OpenAIRateLimitError(
        "rate limit",
        response=MagicMock(status_code=429),
        body=None,
    )

    def _create(**kwargs):
        if kwargs.get("model") == "llama-3.1-8b-instant":
            return ok_response
        raise rate_exc

    client.chat.completions.create.side_effect = _create
    resp, err = groq_optimizer.safe_groq_chat_create(
        client,
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=4096,
    )
    assert err is None
    assert resp is ok_response
    models_used = [
        c.kwargs.get("model") for c in client.chat.completions.create.call_args_list
    ]
    assert "llama-3.1-8b-instant" in models_used


def test_ui_handler_catches_applylytics_llm_error():
    """Simulate panel pattern: catch ApplylyticsLLMError and surface warning flag."""

    def panel_action() -> str:
        raise ApplylyticsLLMError("quota")

    warning_shown = False
    try:
        panel_action()
    except ApplylyticsLLMError:
        warning_shown = True

    assert warning_shown
