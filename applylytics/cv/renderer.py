"""DOCX rendering and CV text helpers."""

from __future__ import annotations

import tempfile
from pathlib import Path

from utils.cv_generator import cv_context_from_structured, render_cv

__all__ = ["cv_context_from_structured", "render_cv_to_text", "render_docx_to_bytes"]


def render_docx_to_bytes(template_path: Path, context: dict) -> bytes:
    """Render a DOCX template to bytes via a temporary file."""
    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
        out_path = tmp.name
    try:
        render_cv(str(template_path), out_path, context)
        return Path(out_path).read_bytes()
    finally:
        Path(out_path).unlink(missing_ok=True)


def render_cv_to_text(cv_json: dict) -> str:
    """Convert optimised CV JSON to a plain text summary for prompts."""
    text = f"Personal Statement: {cv_json.get('personal_statement', '')}\n\n"

    work = cv_json.get("work_experience") or []
    if work:
        text += "Work Experience:\n"
        for exp in work:
            if not isinstance(exp, dict):
                continue
            text += f"- {exp.get('job_title', '')} at {exp.get('employer', '')} ({exp.get('dates', '')})\n"
            for bullet in exp.get("bullets") or []:
                text += f"  • {bullet}\n"

    projects = cv_json.get("projects") or []
    if projects:
        text += "\nProjects:\n"
        for proj in projects:
            if not isinstance(proj, dict):
                continue
            text += f"- {proj.get('name', '')} ({proj.get('date', '')})\n"
            if proj.get("description"):
                text += f"  {proj['description']}\n"
            techs = proj.get("technologies") or []
            if techs:
                text += f"  Technologies: {', '.join(str(t) for t in techs)}\n"

    text += f"\nKey Skills: {', '.join(cv_json.get('key_skills') or [])}\n"
    text += f"Skills: {', '.join(cv_json.get('skills') or [])}\n"
    return text


def cv_plaintext_for_ats_scoring(data: dict) -> str:
    """Flatten structured CV JSON for whitelist keyword overlap scoring."""
    parts: list[str] = []
    if data.get("personal_statement"):
        parts.append(str(data["personal_statement"]))
    for exp in data.get("work_experience") or []:
        if not isinstance(exp, dict):
            continue
        parts.append(str(exp.get("job_title", "") or ""))
        parts.append(str(exp.get("employer", "") or ""))
        parts.extend(str(b) for b in (exp.get("bullets") or []) if b)
    for proj in data.get("projects") or []:
        if not isinstance(proj, dict):
            continue
        parts.append(str(proj.get("name", "") or ""))
        parts.append(str(proj.get("description", "") or ""))
        parts.extend(str(t) for t in (proj.get("technologies") or []) if t)
        if proj.get("link"):
            parts.append(str(proj["link"]))
    parts.extend(str(s) for s in (data.get("key_skills") or []) if s)
    parts.extend(str(s) for s in (data.get("skills") or []) if s)
    for edu in data.get("education") or []:
        if not isinstance(edu, dict):
            continue
        for key in ("degree", "institution", "dates", "grade"):
            val = edu.get(key)
            if val:
                parts.append(str(val))
    parts.extend(str(c) for c in (data.get("certifications") or []) if c)
    interests = data.get("interests")
    if isinstance(interests, list):
        parts.extend(str(i) for i in interests if i)
    elif isinstance(interests, str) and interests.strip():
        parts.append(interests)
    return "\n".join(parts)
