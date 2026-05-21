"""Tests for flexible CV JSON normalisation."""

from applylytics.cv.schema import normalize_cv_json
from applylytics.cv.renderer import cv_plaintext_for_ats_scoring


def test_normalize_empty_work_and_projects():
    data = normalize_cv_json(
        {
            "name": "Test User",
            "work_experience": [],
            "projects": [
                {
                    "name": "Portfolio App",
                    "description": "Built with Python and SQL",
                    "technologies": ["Python", "SQL"],
                    "date": "2024",
                }
            ],
        }
    )
    assert data["work_experience"] == []
    assert len(data["projects"]) == 1
    plain = cv_plaintext_for_ats_scoring(data)
    assert "python" in plain.lower()
    assert "portfolio" in plain.lower()


def test_normalize_legacy_projects_string():
    data = normalize_cv_json({"projects": "Legacy project description"})
    assert len(data["projects"]) == 1
    assert "Legacy" in data["projects"][0]["description"]
