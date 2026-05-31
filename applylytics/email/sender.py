"""
Email sending using Brevo API.
"""

from __future__ import annotations

import base64
import logging
import os
from pathlib import Path

import sib_api_v3_sdk
from sib_api_v3_sdk.rest import ApiException

from applylytics.email.content import (
    build_cv_attachment_filename,
    build_email_subject,
    build_optimised_cv_html,
    build_plain_text_fallback,
)
from applylytics.email.job_metadata import JobMetadata

logger = logging.getLogger("applylytics")


def _resolve_brevo_api_key() -> str:
    """Read Brevo key from .env / settings (reloads .env for Celery workers)."""
    from applylytics.config import resolve_brevo_api_key

    return resolve_brevo_api_key()


def _resolve_sender() -> tuple[str, str]:
    """Sender email and display name (env / secrets / defaults)."""
    email = (os.environ.get("BREVO_SENDER_EMAIL") or "9chirla@gmail.com").strip()
    name = (os.environ.get("BREVO_SENDER_NAME") or "Applylytics").strip()
    try:
        import streamlit as st

        email = str(st.secrets.get("BREVO_SENDER_EMAIL", email) or email).strip()
        name = str(st.secrets.get("BREVO_SENDER_NAME", name) or name).strip()
    except Exception:
        pass
    return email, name


def _resolve_portfolio_url() -> str:
    try:
        from applylytics.config import settings

        return (settings.portfolio_url or "").strip()
    except Exception:
        return (os.environ.get("PORTFOLIO_URL") or "").strip()


def send_optimised_cv_email(
    recipient_email: str,
    recipient_name: str,
    job: JobMetadata,
    cv_file_path: Path,
    *,
    candidate_name: str = "",
) -> tuple[bool, str | None]:
    """
    Send an email with the optimised CV as attachment.
    Returns (success, error_message). error_message is None on success.
    """
    api_key = _resolve_brevo_api_key()
    if not api_key:
        return False, (
            "BREVO_API_KEY missing or placeholder in .env. "
            "Set a valid xkeysib-… key and restart the Celery worker."
        )

    display_name = (candidate_name or recipient_name or "").strip() or "Candidate"
    attachment_name = build_cv_attachment_filename(
        candidate_name=display_name,
        job=job,
    )
    subject = build_email_subject(job)
    html_content = build_optimised_cv_html(
        recipient_name=recipient_name,
        job=job,
        portfolio_url=_resolve_portfolio_url(),
    )
    text_content = build_plain_text_fallback(
        recipient_name=recipient_name,
        job=job,
        attachment_name=attachment_name,
    )

    configuration = sib_api_v3_sdk.Configuration()
    configuration.api_key["api-key"] = api_key
    api_instance = sib_api_v3_sdk.TransactionalEmailsApi(
        sib_api_v3_sdk.ApiClient(configuration)
    )

    with open(cv_file_path, "rb") as f:
        file_content = base64.b64encode(f.read()).decode()

    attachment = sib_api_v3_sdk.SendSmtpEmailAttachment(
        content=file_content,
        name=attachment_name,
    )

    sender_email, sender_name = _resolve_sender()
    send_smtp_email = sib_api_v3_sdk.SendSmtpEmail(
        to=[{"email": recipient_email, "name": recipient_name}],
        sender={"email": sender_email, "name": sender_name},
        subject=subject,
        html_content=html_content,
        text_content=text_content,
        attachment=[attachment],
    )

    try:
        api_instance.send_transac_email(send_smtp_email)
        logger.info(
            "Brevo email sent to %s subject=%r attachment=%s",
            recipient_email,
            subject,
            attachment_name,
        )
        return True, None
    except ApiException as e:
        logger.error("Brevo API error: %s", e)
        body = getattr(e, "body", None) or str(e)
        if "Key not found" in str(body) or getattr(e, "status", None) == 401:
            return False, (
                "Brevo rejected the API key (401 Key not found). "
                "Use the same xkeysib key that works in curl, update .env, "
                "and restart the RQ worker: rq worker --with-scheduler. "
                "Also check Brevo → Security → authorized IPs is off for local dev."
            )
        return False, f"Brevo email failed: {body}"
