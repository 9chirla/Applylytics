"""Pure UI helpers (no Streamlit calls) extracted from panels."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from applylytics.constants import (
    APP_DIR,
    CV_TEMPLATE_CHOICES,
    DEFAULT_CV_TEMPLATE,
    RATE_LIMIT_WARNING,
    resolve_cv_template,
)

FEEDBACK_JSONL = APP_DIR / "data" / "feedback.jsonl"
from applylytics.llm.client import ApplylyticsLLMError, RateLimitError

_JOB_DESCRIPTION_BLOCKLIST = (
    "STRICT_SYSTEM_PROMPT",
    "cv_optimizer.py",
    "build_targeted_keyword_requirements",
    "def render_",
)

RQ_EMAIL_TIMEOUT_SECONDS = 60
RQ_WORKER_OFFLINE_MSG = (
    "Email worker is offline — start Redis and run "
    "`rq worker --with-scheduler`, or download your DOCX below."
)


def phase_key(resume_text: str, job_description: str) -> str:
    return (
        f"{hashlib.sha256(resume_text.encode()).hexdigest()[:16]}:"
        f"{hashlib.sha256(job_description.encode()).hexdigest()[:24]}"
    )


def optimise_queue_signature(
    *,
    email: str,
    company: str,
    resume_text: str,
    job_description: str,
) -> str:
    return f"{email}|{company}|{len(resume_text)}|{len(job_description)}"


def serialisable_cv_payload(data: dict) -> dict:
    """Strip private keys before sending optimised JSON to a background worker."""
    return {k: v for k, v in data.items() if not str(k).startswith("_")}


def sanitize_coach_display(text: str) -> str:
    """Strip markdown headings and fenced code blocks from coach output."""
    lines: list[str] = []
    for line in text.splitlines():
        if line.strip().startswith("```"):
            continue
        if re.match(r"^#{1,6}\s+", line):
            heading = re.sub(r"^#{1,6}\s+", "", line).strip()
            lines.append(f"**{heading}**")
        else:
            lines.append(line)
    return "\n".join(lines)


def optimised_hm_cache_key(
    resume_text: str,
    job_description: str,
    optimised_cv_text: str,
) -> str:
    base = phase_key(resume_text, job_description)
    fp = hashlib.sha256(optimised_cv_text.encode()).hexdigest()[:20]
    return f"{base}:optimised:{fp}:fresh_v3"


def input_signature(has_resume: bool, job_description: str, company: str = "") -> str:
    payload = f"{has_resume}:{job_description}:{company}"
    return hashlib.sha256(payload.encode()).hexdigest()[:20]


def normalize_job_description_input(jd: str) -> str:
    """Reject pasted code/prompt fragments masquerading as a job description."""
    if any(marker in jd for marker in _JOB_DESCRIPTION_BLOCKLIST):
        return ""
    return jd.strip()


def resolve_template_choice(choice: str | None) -> str:
    key = (choice or DEFAULT_CV_TEMPLATE).strip()
    if key in CV_TEMPLATE_CHOICES:
        return key
    return DEFAULT_CV_TEMPLATE


def resolve_template_path(choice: str | None = None) -> Path:
    return resolve_cv_template(resolve_template_choice(choice))


def format_llm_warning_message(exc: BaseException) -> str:
    """User-facing LLM warning text for st.warning()."""
    if isinstance(exc, ApplylyticsLLMError):
        return exc.user_message.strip()
    if isinstance(exc, RateLimitError):
        return str(exc).strip() or RATE_LIMIT_WARNING
    return str(exc).strip() or "The AI service is temporarily unavailable."


def ats_score_band(score: int) -> str:
    """Coarse ATS band used in tests and UI copy."""
    if score < 40:
        return "low"
    if score <= 70:
        return "mid"
    return "high"


def ats_score_title(score: int) -> str:
    if score >= 80:
        return "Strong Match"
    if score >= 60:
        return "Moderate Match"
    if score >= 40:
        return "Fair Match"
    return "Needs Work"


def ats_score_description(score: int, matched_count: int, job_keyword_count: int) -> str:
    total = max(job_keyword_count, 1)
    if score >= 80:
        tail = "You're in great shape for this role."
    elif score >= 60:
        tail = "You're close — a targeted rewrite could push this to 80%+."
    elif score >= 40:
        tail = "Several gaps remain — focus on missing phrases where truthful."
    else:
        tail = "Significant gaps — prioritise the missing keywords in your rewrite."
    return f"{matched_count} of {total} skill phrases found. {tail}"


def ats_band_guidance(score: int) -> str:
    """Short guidance line for score bands (<40, 40–70, >70)."""
    band = ats_score_band(score)
    if band == "low":
        return "Significant gaps — prioritise the missing keywords in your rewrite."
    if band == "mid":
        return "Several gaps remain — focus on missing phrases where truthful."
    return "You're in great shape for this role."


def evaluate_rq_email_job(
    *,
    job_id: str | None,
    queued_at: float | None,
    now: float,
    redis_available: bool,
    fetch_job: Any,
) -> dict[str, Any]:
    """
    Pure rq poll state for the email delivery UI.

    Returns dict with keys: in_progress, finished, failed, timed_out, worker_offline,
    failure_reason, result_payload.
    """
    idle = {
        "in_progress": False,
        "finished": False,
        "failed": False,
        "timed_out": False,
        "worker_offline": False,
        "failure_reason": None,
        "result_payload": None,
    }
    if not job_id:
        return idle
    if not redis_available:
        return {**idle, "worker_offline": True, "failure_reason": RQ_WORKER_OFFLINE_MSG}

    try:
        job = fetch_job(job_id)
    except Exception as exc:
        return {
            **idle,
            "worker_offline": True,
            "failure_reason": f"{RQ_WORKER_OFFLINE_MSG} ({exc})",
        }

    if getattr(job, "is_failed", False):
        reason = _format_rq_failure(job)
        return {
            **idle,
            "failed": True,
            "failure_reason": reason,
            "result_payload": getattr(job, "result", None),
        }

    if getattr(job, "is_finished", False):
        return {**idle, "finished": True, "result_payload": getattr(job, "result", None)}

    if queued_at is not None and (now - queued_at) > RQ_EMAIL_TIMEOUT_SECONDS:
        return {
            **idle,
            "timed_out": True,
            "failure_reason": (
                "Email is taking longer than expected (60s). "
                "The worker may be offline or busy."
            ),
        }

    return {**idle, "in_progress": True}


def feedback_file_path() -> Path:
    """Path to append-only export feedback log."""
    return FEEDBACK_JSONL


def record_export_feedback(phase: str, score: int, helpful: bool) -> None:
    """Append one JSON line of export feedback (creates data/ if needed)."""
    path = feedback_file_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        "ats_score": int(score),
        "helpful": bool(helpful),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _format_rq_failure(job: Any) -> str:
    if getattr(job, "exc_info", None):
        return str(job.exc_info)
    result = getattr(job, "result", None)
    if isinstance(result, dict) and result.get("reason"):
        return str(result["reason"])
    if result is not None:
        return str(result)
    return "Background email job failed."
