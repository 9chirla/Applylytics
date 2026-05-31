"""PDF upload validation and text extraction."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pdfplumber

MAX_PDF_BYTES = 10 * 1024 * 1024
PDF_MAGIC = b"%PDF"

_JOB_DESCRIPTION_PHRASES = (
    "we are looking for",
    "responsibilities include",
    "requirements:",
    "qualifications:",
    "about the role",
    "you will",
    "must have",
)


def detect_is_job_description(text: str) -> bool:
    """
    Heuristic: True when extracted PDF text looks like a job posting, not a CV.

    Requires 3+ known JD phrases (case-insensitive substring match).
    """
    if not (text or "").strip():
        return False
    lower = text.lower()
    hits = sum(1 for phrase in _JOB_DESCRIPTION_PHRASES if phrase in lower)
    return hits >= 3


def validate_pdf(file_bytes: bytes, *, mime_type: str | None = None) -> tuple[bool, str]:
    """
    Validate an uploaded PDF before extraction.

    Checks: MIME type (if provided), size ≤ 10 MB, magic bytes %PDF.
    Returns (ok, error_message).
    """
    if not file_bytes:
        return False, "No file was uploaded."
    if mime_type is not None and mime_type.strip().lower() not in (
        "application/pdf",
        "application/x-pdf",
    ):
        return False, "Please upload a PDF file (application/pdf)."
    if len(file_bytes) > MAX_PDF_BYTES:
        return False, "PDF must be 10 MB or smaller."
    if not file_bytes[:4].startswith(PDF_MAGIC):
        return False, "File does not look like a valid PDF (missing %PDF header)."
    return True, ""


def extract_text_from_pdf(file_path: str | Path) -> str:
    """Return all page text from a PDF, joined with newlines."""
    path = Path(file_path)
    parts: list[str] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            layout_text = ""
            try:
                layout_text = (page.extract_text(layout=True) or "").strip()
            except (TypeError, ValueError, AttributeError):
                layout_text = ""
            default_text = (page.extract_text() or "").strip()
            text = layout_text if len(layout_text) >= len(default_text) else default_text
            if not text:
                text = layout_text or default_text
            if text:
                parts.append(text)
    return "\n\n".join(parts)


def extract_text_from_pdf_bytes(file_bytes: bytes) -> str:
    """Validate PDF bytes, write a temp file, and extract text."""
    ok, err = validate_pdf(file_bytes)
    if not ok:
        raise ValueError(err)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        path = tmp.name
    try:
        return extract_text_from_pdf(path)
    finally:
        Path(path).unlink(missing_ok=True)
