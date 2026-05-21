"""Derive which CV sections to show from source resume text and structured JSON."""

from __future__ import annotations

import re
from typing import Any, Mapping

# Coach / LLM placeholder text — must never become a DOCX "project".
_PLACEHOLDER_PROJECT_RE = re.compile(
    r"no\s+projects?\s+listed|candidate\s+should\s+add|strengthen\s+technical\s+credibility|"
    r"e\.g\.\s+a\s+power\s+bi\s+dashboard\s+on\s+public\s+data",
    re.IGNORECASE,
)

_SECTION_HEADER_PATTERNS: dict[str, re.Pattern[str]] = {
    "work_experience": re.compile(
        r"\b(work\s+experience|employment\s+history|professional\s+experience|experience)\b",
        re.IGNORECASE,
    ),
    "projects": re.compile(
        r"\b(projects?|personal\s+projects|academic\s+projects|portfolio)\b",
        re.IGNORECASE,
    ),
    "education": re.compile(r"\b(education|academic\s+background|qualifications)\b", re.IGNORECASE),
    "certifications": re.compile(
        r"\b(certifications?|certificates|licences|licenses|professional\s+development)\b",
        re.IGNORECASE,
    ),
    "skills": re.compile(r"\b(skills|technical\s+skills|core\s+competencies|key\s+skills)\b", re.IGNORECASE),
    "interests": re.compile(r"\b(interests|hobbies)\b", re.IGNORECASE),
}


def is_placeholder_project_text(text: str) -> bool:
    """True if text is optimiser coaching, not a real project."""
    t = (text or "").strip()
    if not t:
        return False
    if _PLACEHOLDER_PROJECT_RE.search(t):
        return True
    if len(t) > 120 and "project" in t.lower() and "add a personal" in t.lower():
        return True
    return False


def sanitize_projects(projects: Any) -> list[dict[str, Any]]:
    """Drop placeholder / empty project entries."""
    if not isinstance(projects, list):
        return []
    out: list[dict[str, Any]] = []
    for item in projects:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "") or "").strip()
        description = str(item.get("description", "") or "").strip()
        combined = f"{name} {description}".strip()
        if is_placeholder_project_text(combined):
            continue
        if name.lower() in ("projects", "project", "n/a", "none"):
            if is_placeholder_project_text(description) or not description:
                continue
        if not name and not description:
            continue
        techs = [str(t).strip() for t in (item.get("technologies") or []) if str(t).strip()]
        out.append(
            {
                "name": name or "Project",
                "description": description,
                "technologies": techs,
                "date": str(item.get("date", "") or "").strip(),
                "link": str(item.get("link", "") or "").strip(),
            }
        )
    return out


def detect_sections_in_resume_text(resume_text: str) -> dict[str, bool]:
    """Which section headers / topics appear in the extracted source CV."""
    text = resume_text or ""
    return {key: bool(pat.search(text)) for key, pat in _SECTION_HEADER_PATTERNS.items()}


def compute_section_visibility(
    cv_json: Mapping[str, Any],
    source_resume_text: str | None = None,
) -> dict[str, bool]:
    """
    Decide which DOCX sections to render.

    Uses structured JSON first; source PDF text informs projects/work when ambiguous.
    Never show PROJECTS for coach placeholders or when there is no real project content.
    """
    source_hints = detect_sections_in_resume_text(source_resume_text or "")

    work = cv_json.get("work_experience") or []
    has_work = bool(
        isinstance(work, list)
        and any(
            isinstance(j, dict)
            and (str(j.get("job_title", "")).strip() or str(j.get("employer", "")).strip())
            for j in work
        )
    )

    projects = sanitize_projects(cv_json.get("projects"))
    has_projects = bool(projects)

    education = cv_json.get("education") or []
    has_education = bool(
        isinstance(education, list)
        and any(
            isinstance(e, dict)
            and (str(e.get("degree", "")).strip() or str(e.get("institution", "")).strip())
            for e in education
        )
    )

    certs = cv_json.get("certifications") or []
    has_certs = bool(
        isinstance(certs, list) and any(str(c).strip() for c in certs)
    )

    skills = cv_json.get("skills") or []
    has_skills = bool(isinstance(skills, list) and any(str(s).strip() for s in skills))

    interests = cv_json.get("interests") or []
    if isinstance(interests, str):
        has_interests = bool(interests.strip())
    else:
        has_interests = bool(
            isinstance(interests, list) and any(str(i).strip() for i in interests)
        )

    # If source clearly had no projects section and JSON only has placeholders, hide.
    if not has_projects and not source_hints.get("projects"):
        has_projects = False

    return {
        "show_work_experience": has_work,
        "show_projects": has_projects,
        "show_education": has_education,
        "show_certifications": has_certs,
        "show_skills": has_skills,
        "show_interests": has_interests,
    }
