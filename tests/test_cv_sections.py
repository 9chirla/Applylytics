"""Section visibility and project placeholder filtering."""

from applylytics.cv.schema import normalize_cv_json
from applylytics.cv.sections import (
    compute_section_visibility,
    is_placeholder_project_text,
    sanitize_projects,
)
from utils.cv_generator import cv_context_from_structured

PLACEHOLDER = (
    "No projects listed. Candidate should add a personal project "
    "(e.g. a Power BI dashboard on public data, a Python analysis published to GitHub) "
    "to strengthen technical credibility."
)


def test_is_placeholder_project_text():
    assert is_placeholder_project_text(PLACEHOLDER)
    assert not is_placeholder_project_text("Built a forecasting dashboard in Power BI")


def test_sanitize_projects_drops_coaching():
    assert sanitize_projects([{"name": "Projects", "description": PLACEHOLDER}]) == []
    assert sanitize_projects(PLACEHOLDER) == []


def test_normalize_cv_json_drops_placeholder_string():
    data = normalize_cv_json({"projects": PLACEHOLDER})
    assert data["projects"] == []


def test_compute_section_visibility_hides_empty_projects():
    vis = compute_section_visibility(
        {"projects": [], "work_experience": [{"job_title": "Dev", "employer": "Co", "dates": "2024", "bullets": []}]},
        source_resume_text="Work Experience\nDeveloper at Co",
    )
    assert vis["show_projects"] is False
    assert vis["show_work_experience"] is True


def test_cv_context_hides_projects_for_placeholder():
    ctx = cv_context_from_structured(
        normalize_cv_json({"projects": PLACEHOLDER, "skills": ["Python"]}),
        source_resume_text="Skills\nPython",
    )
    assert ctx["show_projects"] is False
    assert ctx["projects"] == []
