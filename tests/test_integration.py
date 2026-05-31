"""Integration tests for ATS scoring, LLM fallback, and DOCX templates."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from openai import RateLimitError as OpenAIRateLimitError

import time

from applylytics.ats.scorer import calculate_ats_score, load_semantic_model, semantic_similarity_score
from applylytics.constants import CV_TEMPLATE_CHOICES
from applylytics.cv.docx_generator import cv_context_from_structured, render_cv
from applylytics.cv.renderer import render_docx_to_bytes

ROOT = Path(__file__).resolve().parent.parent

SAMPLE_JD = """
Senior Python Engineer at Example Corp.
Requirements: Python, FastAPI, PostgreSQL, Docker, AWS, pytest, CI/CD.
You will build APIs and mentor junior developers.
"""

SAMPLE_CV = """
Jane Doe
Software Engineer with 5 years building Python APIs and data pipelines.
Skills: Python, FastAPI, PostgreSQL, Docker, AWS, pytest, Git, Agile.
Experience at TechCo developing REST services and unit tests.
"""


def test_ats_scorer_returns_positive_keyword_score():
    result = calculate_ats_score(SAMPLE_CV, SAMPLE_JD, include_semantic=False)
    assert int(result.get("keyword_score", result.get("score", 0))) > 0
    assert result.get("matched_keywords")


def test_llm_fallback_calls_gemini_after_groq_rate_limit(monkeypatch):
    from applylytics.llm import groq_optimizer

    groq_client = MagicMock()
    gemini_client = MagicMock()
    ok_response = MagicMock()
    gemini_client.chat.completions.create.return_value = ok_response

    groq_client.chat.completions.create.side_effect = OpenAIRateLimitError(
        "rate limit",
        response=MagicMock(status_code=429),
        body=None,
    )

    monkeypatch.setattr(groq_optimizer, "get_gemini_client", lambda: gemini_client)
    monkeypatch.setattr(groq_optimizer, "get_gemini_model", lambda: "gemini-2.0-flash")
    monkeypatch.setattr(
        groq_optimizer,
        "get_gemini_fallback_models",
        lambda: ["gemini-2.0-flash"],
    )

    resp, err = groq_optimizer.safe_groq_chat_create(
        groq_client,
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=128,
    )
    assert err is None
    assert resp is ok_response
    assert gemini_client.chat.completions.create.called


@pytest.mark.parametrize("template_name", list(CV_TEMPLATE_CHOICES.keys()))
def test_docx_export_each_template_valid_size(template_name: str, tmp_path: Path):
    template_path = CV_TEMPLATE_CHOICES[template_name]
    assert template_path.is_file(), f"Missing template: {template_path}"

    sample = {
        "name": "Integration Test",
        "contact_email": "test@example.com",
        "personal_statement": "Experienced engineer.",
        "work_experience": [
            {
                "company": "Co",
                "role": "Engineer",
                "dates": "2020 – Present",
                "bullets": ["Built APIs in Python."],
            }
        ],
        "projects": [],
        "education": [{"institution": "Uni", "degree": "BSc", "dates": "2016"}],
        "skills": ["Python", "FastAPI"],
        "certifications": [],
    }
    context = cv_context_from_structured(sample)
    out = tmp_path / f"{template_name}.docx"
    render_cv(str(template_path), str(out), context)
    data = out.read_bytes()
    assert data[:2] == b"PK"
    assert len(data) > 1024

    via_renderer = render_docx_to_bytes(template_path, context)
    assert len(via_renderer) > 1024


def test_load_semantic_model_returns_same_instance():
    """Embedding model is cached once per process (not re-instantiated)."""
    first = load_semantic_model()
    second = load_semantic_model()
    assert first is second


def test_semantic_similarity_reuses_cached_model(monkeypatch):
    """Two sequential scoring calls stay fast when encode uses the warm cached model."""
    import numpy as np

    model = load_semantic_model()

    def _fast_encode(texts):
        return np.zeros((len(texts), 8), dtype=float)

    monkeypatch.setattr(model, "encode", _fast_encode)
    start = time.perf_counter()
    first = semantic_similarity_score(SAMPLE_CV, SAMPLE_JD)
    second = semantic_similarity_score(SAMPLE_CV, SAMPLE_JD)
    elapsed = time.perf_counter() - start
    assert first is not None and second is not None
    assert elapsed < 2.0
