"""CV optimisation pipeline (generic Groq optimiser with keyword preservation)."""

from __future__ import annotations

import logging
import re

from applylytics.llm.client import RateLimitError, is_groq_rate_limit_error
from applylytics.llm.context import prepare_resume_job_texts
from applylytics.llm import prompts
from applylytics.ats.scorer import calculate_ats_score_cached
from applylytics.cv.renderer import cv_plaintext_for_ats_scoring
from applylytics.cv.schema import normalize_cv_json

logger = logging.getLogger("applylytics")

SPACING_USER_APPENDIX = (
    "Ensure all institution names, project titles, and skill phrases have proper spaces. "
    "For example, write 'Gayatri Vidya Parishad College of Engineering' not "
    "'GayatriVidyaParishadCollegeOfEngineering'."
)


def _fix_concatenated_words(text: str) -> str:
    """Insert spaces between words that were accidentally concatenated."""
    if not text or not isinstance(text, str):
        return text
    if "@" in text or text.strip().lower().startswith("http"):
        return text
    s = text
    s = re.sub(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", " ", s)
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", s)
    s = re.sub(r"-([A-Z][a-z])", r"- \1", s)
    return re.sub(r"\s+", " ", s).strip()


def _fix_json_strings(obj: object) -> object:
    """Recursively fix concatenated words in all string values."""
    if isinstance(obj, dict):
        return {k: _fix_json_strings(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_fix_json_strings(item) for item in obj]
    if isinstance(obj, str):
        return _fix_concatenated_words(obj)
    return obj


def _fix_json_strings_with_log(obj: object) -> tuple[object, list[str]]:
    """Fix strings and return human-readable before/after samples for UI warnings."""
    examples: list[str] = []

    def _walk(value: object) -> object:
        if isinstance(value, dict):
            return {k: _walk(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_walk(item) for item in value]
        if isinstance(value, str):
            fixed = _fix_concatenated_words(value)
            if fixed != value and value.strip():
                preview_old = value if len(value) <= 60 else value[:57] + "…"
                preview_new = fixed if len(fixed) <= 60 else fixed[:57] + "…"
                examples.append(f"{preview_old} → {preview_new}")
            return fixed
        return value

    return _walk(obj), examples


def _finalize_parsed_cv(parsed: dict) -> dict:
    """Normalise schema and fix concatenated words after LLM JSON parse."""
    fixed, examples = _fix_json_strings_with_log(parsed)
    out = normalize_cv_json(fixed if isinstance(fixed, dict) else parsed)
    if examples:
        out["_spacing_auto_fixed"] = examples
        logger.warning("Auto-fixed %s concatenated word string(s) in CV JSON", len(examples))
    return out


def _keyword_in_plaintext(keyword: str, plaintext: str) -> bool:
    """True if keyword appears as a whole phrase in lowercased plaintext."""
    kw = keyword.strip().lower()
    if not kw:
        return False
    return bool(re.search(rf"(?<!\w){re.escape(kw)}(?!\w)", plaintext.lower()))


def _format_required_keywords_block(missing_keywords: list[str]) -> str:
    """Extra user-message block: REQUIRED verbatim phrases for the optimiser."""
    mk = [str(k).strip() for k in (missing_keywords or []) if k and str(k).strip()][:40]
    if not mk:
        return ""
    bullets = "\n".join(f'  • "{kw}"' for kw in mk)
    return (
        "\n\nREQUIRED KEYWORDS (mandatory where truthful — exact wording, do not paraphrase):\n"
        f"{bullets}\n"
        "Preserve every skill keyword already present in the source CV; only add missing ones.\n"
        f"{SPACING_USER_APPENDIX}\n"
    )


def _ensure_keywords_present(cv_json: dict, target_keywords: list[str]) -> None:
    """
    Post-optimisation safety net: append missing single/two-word skills to ``skills``
    (and ``key_skills`` when room) if not already present in CV plaintext.
    """
    plain = cv_plaintext_for_ats_scoring(cv_json)
    skills = cv_json.get("skills")
    if not isinstance(skills, list):
        skills = []
        cv_json["skills"] = skills
    key_skills = cv_json.get("key_skills")
    if not isinstance(key_skills, list):
        key_skills = []
        cv_json["key_skills"] = key_skills

    def _listed(kw: str) -> bool:
        kl = kw.lower()
        return any(isinstance(s, str) and s.strip().lower() == kl for s in skills + key_skills)

    for raw in target_keywords:
        kw = str(raw).strip()
        if not kw or len(kw.split()) > 2:
            continue
        if _keyword_in_plaintext(kw, plain) or _listed(kw):
            continue
        skills.append(kw)
        if len(key_skills) < 18 and not any(
            isinstance(s, str) and s.strip().lower() == kw.lower() for s in key_skills
        ):
            key_skills.append(kw)


def optimize_cv_generic(
    resume_text: str,
    job_text: str,
    missing_keywords: list[str],
    *,
    keyword_requirements_block: str = "",
) -> dict:
    """Generic CV optimiser via utils pipeline with keyword preservation."""
    from utils.cv_optimizer import run_generic_cv_optimisation_pipeline

    resume, job = prepare_resume_job_texts(resume_text, job_text)
    rt = _fix_concatenated_words(resume.strip())
    jt = (job or "").strip()
    mk = [str(k).strip() for k in (missing_keywords or []) if k and str(k).strip()]

    logger.debug(
        "optimize_cv_generic() → pipeline | resume_chars=%s | missing_keywords=%s",
        len(rt),
        len(mk),
    )

    try:
        before_result = calculate_ats_score_cached(rt, jt)
        score_before = int(before_result.get("score", 0))
    except Exception as exc:
        logger.warning("Operation failed: %s", exc)
        score_before = 0

    keyword_block = _format_required_keywords_block(mk)
    combined_keyword_block = keyword_block + (keyword_requirements_block or "")

    out = run_generic_cv_optimisation_pipeline(
        rt,
        job_text=jt,
        missing_keywords=mk,
        keyword_requirements_block=combined_keyword_block,
    )
    if is_groq_rate_limit_error(out):
        raise RateLimitError(str(out.get("error") or "Rate limit reached during CV optimisation."))
    if out.get("error"):
        logger.debug("optimize_cv_generic() pipeline error: %s", out["error"])
        return out

    out = _finalize_parsed_cv(out)
    _ensure_keywords_present(out, mk)

    try:
        cv_text = cv_plaintext_for_ats_scoring(out)
        ats_result = calculate_ats_score_cached(cv_text, jt)
        score_after = int(ats_result.get("score", 0))
        out["_ats_score_achieved"] = score_after
    except Exception as exc:
        logger.warning("Operation failed: %s", exc)
        score_after = 0
        out["_ats_score_achieved"] = 0
        ats_result = {"matched_keywords": []}

    plain_after = cv_plaintext_for_ats_scoring(out)
    matched_from_sent = [k for k in mk if _keyword_in_plaintext(k, plain_after)]
    still_missing = [k for k in mk if k not in matched_from_sent]

    out["_ats_debug"] = {
        "keywords_sent": mk,
        "matched_after": matched_from_sent,
        "still_missing": still_missing,
        "score_before": score_before,
        "score_after": score_after,
        "spacing_auto_fixed": out.pop("_spacing_auto_fixed", None) or [],
    }

    ps = str(out.get("personal_statement", "") or "")
    n_jobs = len(out.get("work_experience") or [])
    n_bullets = sum(
        len(exp.get("bullets") or [])
        for exp in (out.get("work_experience") or [])
        if isinstance(exp, dict)
    )
    logger.debug(
        "optimize_cv_generic() OK | jobs=%s | bullets=%s | ats_estimate=%s",
        n_jobs,
        n_bullets,
        out.get("_ats_score_achieved"),
    )
    logger.debug("personal_statement[:200]=%r", ps[:200])
    return out
