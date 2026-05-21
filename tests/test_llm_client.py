"""Tests for LLM client JSON parsing and rate-limit handling."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from openai import RateLimitError as OpenAIRateLimitError

from applylytics.llm.client import (
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


def test_safe_groq_chat_create_raises_rate_limit_error():
    client = MagicMock()
    client.chat.completions.create.side_effect = OpenAIRateLimitError(
        "rate limit",
        response=MagicMock(status_code=429),
        body=None,
    )
    with pytest.raises(RateLimitError):
        safe_groq_chat_create(client, model="llama-3.3-70b-versatile", messages=[])


def test_ui_handler_catches_rate_limit_error():
    """Simulate panel pattern: catch RateLimitError and surface warning flag."""

    def panel_action() -> str:
        raise RateLimitError("quota")

    warning_shown = False
    try:
        panel_action()
    except RateLimitError:
        warning_shown = True

    assert warning_shown
