"""Extract plain text from PDF resumes using pdfplumber."""

from pathlib import Path

import pdfplumber


def extract_text_from_pdf(file_path: str | Path) -> str:
    """Return all page text from a PDF, joined with newlines.

    Word/LibreOffice “Save as PDF” and multi-column layouts often read poorly with the
    default reading order. We prefer ``layout=True`` when it yields more text (usually
    closer to visual reading order); otherwise we fall back to the default extractor.
    """
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
            # Prefer the extraction that recovered more characters (better for CVs).
            text = layout_text if len(layout_text) >= len(default_text) else default_text
            if not text:
                text = layout_text or default_text
            if text:
                parts.append(text)
    return "\n\n".join(parts)
