"""Groq-powered resume coaching calls."""

from __future__ import annotations

import logging

from applylytics.config import settings
from applylytics.llm.client import get_groq_client, require_groq_api_key
from utils.cv_optimizer import get_gemini_client, get_gemini_model, safe_groq_chat_create
from applylytics.llm.context import prepare_resume_job_texts
from applylytics.llm import prompts

logger = logging.getLogger("applylytics")


def analyze_resume_with_job(resume_text: str, job_text: str) -> str | None:
    """Targeted resume vs job feedback. Returns None on missing API key; raises RateLimitError on quota."""
    if require_groq_api_key() is None:
        return None
    client = get_groq_client()
    resume, job = prepare_resume_job_texts(resume_text, job_text)
    gemini = get_gemini_client()
    gemini_model = get_gemini_model() if gemini else None
    response, err = safe_groq_chat_create(
        client,
        fallback_client=gemini,
        fallback_model=gemini_model,
        model=settings.groq_model,
        messages=[
            {"role": "system", "content": prompts.RESUME_COACH_SYSTEM},
            {"role": "user", "content": prompts.resume_analysis_user(resume, job)},
        ],
        temperature=0.4,
    )
    if err:
        return err
    return response.choices[0].message.content or ""


def get_hiring_manager_comment(
    resume_text: str,
    job_text: str,
    ats_score: int,
    optimised_text: str | None = None,
    *,
    fresh_review: bool = False,
) -> str | None:
    """Hiring manager brief. Returns None on missing API key; raises RateLimitError on quota."""
    if require_groq_api_key() is None:
        return None
    client = get_groq_client()
    resume, job = prepare_resume_job_texts(resume_text, job_text)
    if fresh_review:
        prompt = prompts.hiring_manager_fresh_review_prompt(resume, job, ats_score)
    else:
        prompt = prompts.hiring_manager_prompt(resume, job, ats_score, optimised_text)
    gemini = get_gemini_client()
    gemini_model = get_gemini_model() if gemini else None
    response, err = safe_groq_chat_create(
        client,
        fallback_client=gemini,
        fallback_model=gemini_model,
        model=settings.groq_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
    )
    if err:
        return err
    return (response.choices[0].message.content or "").strip()
