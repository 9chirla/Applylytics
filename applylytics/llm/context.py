"""Text truncation helpers for LLM context windows."""

from collections.abc import Callable

from applylytics.constants import MAX_COMBINED_CHARS, MAX_TEXT_CHARS, TRUNCATION_WARNING


def smart_truncate(text: str, max_chars: int) -> str:
    """Keep the first 70% and last 30% of characters when over the limit."""
    if len(text) <= max_chars:
        return text
    head = int(max_chars * 0.7)
    tail = max_chars - head
    return text[:head] + "\n…\n" + text[-tail:]


def prepare_resume_job_texts(
    resume_text: str,
    job_text: str,
    *,
    warn: Callable[[str], None] | None = None,
) -> tuple[str, str]:
    """
    Truncate resume and job text for LLM prompts.

    Single-field limit: 100k chars. Combined limit: 120k (job trimmed first, then resume).
    """
    resume = resume_text or ""
    job = job_text or ""

    if warn:
        if len(resume) > MAX_TEXT_CHARS:
            warn(TRUNCATION_WARNING)
        if len(job) > MAX_TEXT_CHARS:
            warn(TRUNCATION_WARNING)

    resume = smart_truncate(resume, MAX_TEXT_CHARS)
    job = smart_truncate(job, MAX_TEXT_CHARS)

    combined = len(resume) + len(job)
    if combined > MAX_COMBINED_CHARS:
        if warn:
            warn(TRUNCATION_WARNING)
        overflow = combined - MAX_COMBINED_CHARS
        if len(job) >= overflow:
            job = smart_truncate(job, max(len(job) - overflow, 1000))
        else:
            job = ""
            overflow -= len(job_text or "")
            resume = smart_truncate(resume, max(len(resume) - overflow, 1000))

    return resume, job
