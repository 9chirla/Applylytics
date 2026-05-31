"""Startup health checks for templates, LLM keys, and Redis."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from applylytics.config import resolve_groq_api_key
from applylytics.constants import CV_TEMPLATE_CHOICES, CV_TEMPLATE_DOCX

logger = logging.getLogger("applylytics")


@dataclass
class PreflightStatus:
    templates_ok: bool = False
    missing_templates: list[str] = field(default_factory=list)
    groq_ok: bool = False
    redis_ok: bool = False
    email_queue_ok: bool = False

    @property
    def llm_enabled(self) -> bool:
        return self.groq_ok

    @property
    def email_enabled(self) -> bool:
        return self.redis_ok and self.email_queue_ok


def _check_templates() -> tuple[bool, list[str]]:
    missing = [
        name for name, path in CV_TEMPLATE_CHOICES.items() if not path.is_file()
    ]
    if not CV_TEMPLATE_DOCX.is_file() and "MasterTemplate" not in missing:
        missing.append("MasterTemplate")
    return len(missing) == 0, missing


def _check_groq() -> bool:
    return bool(resolve_groq_api_key())


def _check_redis(timeout_seconds: float = 1.0) -> bool:
    try:
        from applylytics.queue import ping_redis

        return ping_redis(timeout_seconds=timeout_seconds)
    except Exception as exc:
        logger.warning("Redis preflight failed: %s", exc)
        return False


def run_preflight() -> PreflightStatus:
    """Verify templates, Groq key, and Redis; log warnings without raising."""
    templates_ok, missing = _check_templates()
    if not templates_ok:
        logger.warning("Missing CV templates: %s", ", ".join(missing))

    groq_ok = _check_groq()
    if not groq_ok:
        logger.warning("GROQ_API_KEY is not set — LLM features will be disabled.")

    redis_ok = _check_redis()
    if not redis_ok:
        logger.warning(
            "Redis is not reachable — background email queue is disabled. "
            "Start Redis and `rq worker --with-scheduler` for email delivery."
        )

    return PreflightStatus(
        templates_ok=templates_ok,
        missing_templates=missing,
        groq_ok=groq_ok,
        redis_ok=redis_ok,
        email_queue_ok=redis_ok,
    )


def service_indicator(ok: bool, *, degraded: bool = False) -> str:
    if ok:
        return "🟢"
    if degraded:
        return "🟡"
    return "🔴"
