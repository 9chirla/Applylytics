"""Background tasks for Applylytics (rq workers)."""

from __future__ import annotations

import tempfile
from pathlib import Path

from applylytics.config import load_project_dotenv

load_project_dotenv()

from applylytics.constants import resolve_cv_template
from applylytics.cv.optimiser import optimize_cv_generic
from applylytics.cv.renderer import cv_context_from_structured, render_docx_to_bytes
from applylytics.email.job_metadata import build_job_metadata
from applylytics.email.sender import send_optimised_cv_email


def _docx_and_email(
    *,
    recipient_email: str,
    recipient_name: str,
    job_title: str,
    company_name: str,
    resume_text: str,
    job_description: str,
    optimised_cv: dict,
    target_company: str = "",
    template_name: str = "Classic",
) -> dict:
    """Render DOCX from pre-optimised JSON and send via Brevo."""
    template_path = resolve_cv_template(template_name)
    context = cv_context_from_structured(
        optimised_cv,
        source_resume_text=resume_text,
    )
    docx_bytes = render_docx_to_bytes(template_path, context)

    user_company = (target_company or company_name or "").strip()
    job_meta = build_job_metadata(
        job_description,
        company_override=user_company,
        job_title_override=job_title if job_title and job_title != "this role" else "",
    )
    cv_name = str(optimised_cv.get("name") or "").strip() or recipient_name

    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
        tmp.write(docx_bytes)
        tmp_path = Path(tmp.name)

    success, email_err = send_optimised_cv_email(
        recipient_email,
        recipient_name,
        job_meta,
        tmp_path,
        candidate_name=cv_name,
    )
    tmp_path.unlink(missing_ok=True)

    if success:
        return {"status": "success", "email": recipient_email}
    return {
        "status": "failed",
        "reason": email_err or "Email sending failed",
        "optimisation_ok": True,
    }


def deliver_optimised_cv_email_task(
    recipient_email: str,
    recipient_name: str,
    job_title: str,
    company_name: str,
    resume_text: str,
    job_description: str,
    optimised_cv: dict,
    target_company: str = "",
    template_name: str = "Classic",
) -> dict:
    """Email an already-optimised CV (no second Groq run)."""
    try:
        return _docx_and_email(
            recipient_email=recipient_email,
            recipient_name=recipient_name,
            job_title=job_title,
            company_name=company_name,
            resume_text=resume_text,
            job_description=job_description,
            optimised_cv=optimised_cv,
            target_company=target_company,
            template_name=template_name,
        )
    except Exception as e:
        return {"status": "failed", "reason": str(e)}


def optimise_and_email_task(
    recipient_email: str,
    recipient_name: str,
    job_title: str,
    company_name: str,
    resume_text: str,
    job_description: str,
    missing_keywords: list,
    target_company: str = "",
    template_name: str = "Classic",
) -> dict:
    """Optimise CV in the background and email the DOCX to the recipient."""
    try:
        optimised_cv = optimize_cv_generic(
            resume_text,
            job_description,
            missing_keywords,
            fast_mode=True,
        )
        if isinstance(optimised_cv, dict) and optimised_cv.get("error"):
            return {"status": "failed", "reason": optimised_cv["error"]}

        return _docx_and_email(
            recipient_email=recipient_email,
            recipient_name=recipient_name,
            job_title=job_title,
            company_name=company_name,
            resume_text=resume_text,
            job_description=job_description,
            optimised_cv=optimised_cv,
            target_company=target_company,
            template_name=template_name,
        )
    except Exception as e:
        from applylytics.llm.client import ApplylyticsLLMError, is_groq_rate_limit_error

        if isinstance(e, ApplylyticsLLMError) or is_groq_rate_limit_error(e):
            return {
                "status": "failed",
                "reason": getattr(e, "user_message", str(e)),
                "rate_limited": True,
            }
        return {"status": "failed", "reason": str(e)}
