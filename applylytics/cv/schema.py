"""Flexible CV JSON schema, documentation for prompts, and normalisation."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("applylytics")

CV_SCHEMA_JSON = """
{
  "name": "string",
  "contact_phone": "string",
  "contact_email": "string",
  "contact_location": "string",
  "linkedin": "string",
  "personal_statement": "string",
  "key_skills": ["string"],
  "skills": ["string"],
  "work_experience": [
    {"job_title": "string", "employer": "string", "dates": "string", "bullets": ["string"]}
  ],
  "projects": [
    {
      "name": "string",
      "description": "string",
      "technologies": ["string"],
      "date": "string",
      "link": "string (optional, may be empty)"
    }
  ],
  "education": [
    {"degree": "string", "institution": "string", "dates": "string", "grade": "string"}
  ],
  "certifications": ["string"],
  "interests": ["string"],
  "references": "string"
}
"""

CV_SCHEMA_INSTRUCTIONS = """
SCHEMA RULES:
- Output ONLY fields relevant to the candidate. Do not invent jobs, projects, degrees, or certifications.
- Always include every top-level key listed in the schema. Use [] for empty arrays (never null).
- work_experience: employment history. Use [] if the CV has no jobs (e.g. student or project-only CV).
- projects: personal, academic, or portfolio projects from the SOURCE CV only. Use [] if the source has no projects — never output coaching or placeholder text in this field.
- If both work experience and projects exist in the source CV, include BOTH.
- key_skills: 8–12 headline skills; skills: fuller list (12–18 where possible).
- certifications, interests: use [] when absent. references: "" or "Available on request" when appropriate.
- For ATS, include exact skill phrases in profile, work bullets, project descriptions, and skills where truthful.
"""


def _normalize_project(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    techs = item.get("technologies")
    if not isinstance(techs, list):
        techs = []
    technologies = [str(t).strip() for t in techs if str(t).strip()]
    name = str(item.get("name", "") or "").strip()
    description = str(item.get("description", "") or "").strip()
    if not name and not description:
        return None
    return {
        "name": name or "Project",
        "description": description,
        "technologies": technologies,
        "date": str(item.get("date", "") or "").strip(),
        "link": str(item.get("link", "") or "").strip(),
    }


def _normalize_work(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    bullets = item.get("bullets")
    if not isinstance(bullets, list):
        bullets = []
    bullet_list = [str(b).strip() for b in bullets if str(b).strip()]
    title = str(item.get("job_title", "") or "").strip()
    employer = str(item.get("employer", "") or "").strip()
    if not title and not employer and not bullet_list:
        return None
    return {
        "job_title": title,
        "employer": employer,
        "dates": str(item.get("dates", "") or "").strip(),
        "bullets": bullet_list,
    }


def _normalize_education(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    degree = str(item.get("degree", "") or "").strip()
    institution = str(item.get("institution", "") or "").strip()
    if not degree and not institution:
        return None
    return {
        "degree": degree,
        "institution": institution,
        "dates": str(item.get("dates", "") or "").strip(),
        "grade": str(item.get("grade", "") or "").strip(),
    }


def normalize_cv_json(data: dict[str, Any]) -> dict[str, Any]:
    """
    Ensure CV JSON matches the flexible schema with safe defaults.
    Logs warnings for missing or malformed keys; never raises.
    """
    if not isinstance(data, dict):
        logger.warning("normalize_cv_json: expected dict, got %s", type(data).__name__)
        data = {}

    p: dict[str, Any] = {k: v for k, v in data.items() if not str(k).startswith("_")}

    string_fields = (
        "name",
        "contact_phone",
        "contact_email",
        "contact_location",
        "linkedin",
        "personal_statement",
        "references",
    )
    for key in string_fields:
        if key not in p:
            logger.warning("normalize_cv_json: missing key %r — using default", key)
        if key not in p or not isinstance(p.get(key), str):
            p[key] = str(p.get(key, "") or "")

    for key in ("key_skills", "skills", "certifications"):
        val = p.get(key)
        if val is None:
            p[key] = []
        elif not isinstance(val, list):
            logger.warning("normalize_cv_json: %r should be a list — coercing", key)
            p[key] = [str(val).strip()] if str(val).strip() else []
        else:
            p[key] = [str(x).strip() for x in val if str(x).strip()]

    p["certifications"] = [
        str(c).lstrip("-• ").strip() for c in p["certifications"] if str(c).lstrip("-• ").strip()
    ]

    raw_interests = p.get("interests")
    if raw_interests is None:
        p["interests"] = []
    elif isinstance(raw_interests, str):
        p["interests"] = [s.strip() for s in raw_interests.split(",") if s.strip()] if raw_interests.strip() else []
    elif isinstance(raw_interests, list):
        p["interests"] = [str(x).strip() for x in raw_interests if str(x).strip()]
    else:
        logger.warning("normalize_cv_json: interests has unexpected type — using []")
        p["interests"] = []

    from applylytics.cv.sections import is_placeholder_project_text, sanitize_projects

    raw_projects = p.get("projects")
    if isinstance(raw_projects, str) and raw_projects.strip():
        if is_placeholder_project_text(raw_projects.strip()):
            logger.warning("normalize_cv_json: dropped placeholder projects string")
            p["projects"] = []
        else:
            logger.warning("normalize_cv_json: projects was a string — converted to one project entry")
            p["projects"] = sanitize_projects(
                [
                    {
                        "name": "Project",
                        "description": raw_projects.strip(),
                        "technologies": [],
                        "date": "",
                        "link": "",
                    }
                ]
            )
    else:
        p["projects"] = sanitize_projects(raw_projects)

    raw_work = p.get("work_experience")
    work: list[dict[str, Any]] = []
    if raw_work is None:
        p["work_experience"] = []
    elif isinstance(raw_work, list):
        for item in raw_work:
            norm = _normalize_work(item)
            if norm:
                work.append(norm)
        p["work_experience"] = work
    else:
        logger.warning("normalize_cv_json: work_experience should be a list — using []")
        p["work_experience"] = []

    raw_edu = p.get("education")
    education: list[dict[str, Any]] = []
    if raw_edu is None:
        p["education"] = []
    elif isinstance(raw_edu, list):
        for item in raw_edu:
            norm = _normalize_education(item)
            if norm:
                education.append(norm)
        p["education"] = education
    else:
        logger.warning("normalize_cv_json: education should be a list — using []")
        p["education"] = []

    if not isinstance(p.get("key_skills"), list) or not p["key_skills"]:
        derived: list[str] = []
        for item in p.get("skills") or []:
            if isinstance(item, str) and item.strip():
                s = item.strip()
                for prefix in ("Technical:", "Analytical:", "Business:"):
                    if s.lower().startswith(prefix.lower()):
                        s = s.split(":", 1)[-1].strip()
                        break
                derived.append(s)
        p["key_skills"] = derived[:12]
    else:
        p["key_skills"] = [str(x).strip() for x in p["key_skills"] if str(x).strip()][:12]

    return p
