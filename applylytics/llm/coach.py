"""Groq-powered resume coaching calls."""

from __future__ import annotations

import logging

from applylytics.config import settings
from applylytics.llm.client import get_groq_client, require_groq_api_key
from utils.cv_optimizer import get_gemini_client, get_gemini_model, safe_groq_chat_create
from applylytics.llm.context import prepare_resume_job_texts
from applylytics.llm import prompts

logger = logging.getLogger("applylytics")

_FRESH_HM_BANNED = (
    "optimised version",
    "optimized version",
    "original cv",
    "if it were like",
    "in your original",
    "compared to your",
    "rewritten version",
)


def _sanitize_fresh_hiring_manager_comment(text: str) -> str:
    """Strip comparison/optimisation wording the model sometimes ignores."""
    import re

    s = (text or "").strip()
    if not s:
        return s
    s = re.sub(
        r"Based on the optimi[sz]ed version,?",
        "Based on this CV,",
        s,
        flags=re.IGNORECASE,
    )
    s = re.sub(r"optimi[sz]ed version", "this CV", s, flags=re.IGNORECASE)
    s = re.sub(r"original cv", "this CV", s, flags=re.IGNORECASE)
    if "based on this cv" not in s.lower():
        verdict = re.search(
            r"(I would (?:definitely|probably|maybe|not) invite you for an interview\.?)",
            s,
            flags=re.IGNORECASE,
        )
        if verdict:
            s = s.rstrip() + f"\n\nBased on this CV, {verdict.group(1)}"
    return s.strip()


def _fresh_hm_looks_like_comparison(text: str) -> bool:
    lo = (text or "").lower()
    return any(marker in lo for marker in _FRESH_HM_BANNED)


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
        messages = [
            {"role": "system", "content": prompts.HM_FRESH_REVIEW_SYSTEM},
            {"role": "user", "content": prompt},
        ]
        temperature = 0.35
    else:
        prompt = prompts.hiring_manager_prompt(resume, job, ats_score, optimised_text)
        messages = [{"role": "user", "content": prompt}]
        temperature = 0.6
    gemini = get_gemini_client()
    gemini_model = get_gemini_model() if gemini else None
    response, err = safe_groq_chat_create(
        client,
        fallback_client=gemini,
        fallback_model=gemini_model,
        model=settings.groq_model,
        messages=messages,
        temperature=temperature,
    )
    if err:
        return err
    out = (response.choices[0].message.content or "").strip()
    if fresh_review:
        out = _sanitize_fresh_hiring_manager_comment(out)
        if _fresh_hm_looks_like_comparison(out):
            logger.warning("Fresh HM comment still contains comparison phrasing after sanitize")
    return out
