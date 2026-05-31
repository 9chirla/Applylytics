"""Tests for pure UI logic extracted from panels."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from applylytics.constants import CV_TEMPLATE_CHOICES, DEFAULT_CV_TEMPLATE
from applylytics.llm.client import ApplylyticsLLMError, RateLimitError
from applylytics.ui.logic import (
    RQ_EMAIL_TIMEOUT_SECONDS,
    RQ_WORKER_OFFLINE_MSG,
    ats_band_guidance,
    ats_score_band,
    ats_score_description,
    ats_score_title,
    evaluate_rq_email_job,
    format_llm_warning_message,
    input_signature,
    normalize_job_description_input,
    optimise_queue_signature,
    optimised_hm_cache_key,
    phase_key,
    resolve_template_choice,
    resolve_template_path,
    sanitize_coach_display,
    serialisable_cv_payload,
)


@pytest.mark.parametrize("name", list(CV_TEMPLATE_CHOICES.keys()))
def test_resolve_template_path_exists(name: str):
    path = resolve_template_path(name)
    assert isinstance(path, Path)
    assert path.is_file()
    assert path.suffix == ".docx"


def test_resolve_template_choice_defaults():
    assert resolve_template_choice(None) == DEFAULT_CV_TEMPLATE
    assert resolve_template_choice("NotARealTemplate") == DEFAULT_CV_TEMPLATE
    assert resolve_template_choice("Modern") == "Modern"


def test_format_llm_warning_applylytics_error():
    msg = format_llm_warning_message(ApplylyticsLLMError("Quota exceeded. Wait and retry."))
    assert "Quota exceeded" in msg


def test_format_llm_warning_rate_limit_fallback():
    msg = format_llm_warning_message(RateLimitError(""))
    assert "Groq rate limit" in msg or "rate limit" in msg.lower()


def test_ats_score_band_thresholds():
    assert ats_score_band(10) == "low"
    assert ats_score_band(39) == "low"
    assert ats_score_band(40) == "mid"
    assert ats_score_band(70) == "mid"
    assert ats_score_band(71) == "high"


@pytest.mark.parametrize(
    ("score", "expected_title"),
    [
        (25, "Needs Work"),
        (45, "Fair Match"),
        (65, "Moderate Match"),
        (85, "Strong Match"),
    ],
)
def test_ats_score_title_labels(score: int, expected_title: str):
    assert ats_score_title(score) == expected_title


def test_ats_band_guidance_low_mid_high():
    assert "Significant gaps" in ats_band_guidance(25)
    assert "Several gaps" in ats_band_guidance(55)
    assert "great shape" in ats_band_guidance(80)


def test_ats_score_description_includes_counts():
    text = ats_score_description(55, matched_count=4, job_keyword_count=10)
    assert "4 of 10" in text


def test_normalize_job_description_rejects_prompt_paste():
    assert normalize_job_description_input("STRICT_SYSTEM_PROMPT\nfoo") == ""
    assert normalize_job_description_input("  Real JD text  ") == "Real JD text"


def test_phase_key_stable_for_same_inputs():
    assert phase_key("cv", "jd") == phase_key("cv", "jd")
    assert phase_key("cv", "jd") != phase_key("cv", "other")


def test_optimise_queue_signature_changes_with_email():
    a = optimise_queue_signature(
        email="a@b.com", company="", resume_text="x", job_description="y"
    )
    b = optimise_queue_signature(
        email="c@d.com", company="", resume_text="x", job_description="y"
    )
    assert a != b


def test_serialisable_cv_payload_strips_private_keys():
    out = serialisable_cv_payload({"name": "A", "_ats_debug": {}})
    assert "_ats_debug" not in out
    assert out["name"] == "A"


def test_sanitize_coach_display_strips_fences():
    raw = "# Title\n```python\nx=1\n```\nLine"
    assert "```" not in sanitize_coach_display(raw)
    assert "**Title**" in sanitize_coach_display(raw)


def test_input_signature_includes_company():
    assert input_signature(True, "jd", "Acme") != input_signature(True, "jd", "Other")


def test_optimised_hm_cache_key_includes_phase():
    key = optimised_hm_cache_key("cv", "jd", "optimised text")
    assert ":optimised:" in key


def test_evaluate_rq_job_worker_offline_without_redis():
    state = evaluate_rq_email_job(
        job_id="abc",
        queued_at=0.0,
        now=100.0,
        redis_available=False,
        fetch_job=lambda _: None,
    )
    assert state["worker_offline"] is True
    assert RQ_WORKER_OFFLINE_MSG in (state["failure_reason"] or "")


def test_evaluate_rq_job_timeout():
    job = MagicMock(is_finished=False, is_failed=False)
    state = evaluate_rq_email_job(
        job_id="abc",
        queued_at=0.0,
        now=float(RQ_EMAIL_TIMEOUT_SECONDS + 1),
        redis_available=True,
        fetch_job=lambda _: job,
    )
    assert state["timed_out"] is True


def test_evaluate_rq_job_failed_shows_exc_info():
    job = MagicMock()
    job.is_finished = False
    job.is_failed = True
    job.exc_info = "SMTP connection refused"
    job.result = None
    state = evaluate_rq_email_job(
        job_id="abc",
        queued_at=10.0,
        now=20.0,
        redis_available=True,
        fetch_job=lambda _: job,
    )
    assert state["failed"] is True
    assert "SMTP" in (state["failure_reason"] or "")
