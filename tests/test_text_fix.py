"""Tests for concatenated-word post-processing."""

from applylytics.cv.optimiser import (
    _fix_concatenated_words,
    _fix_json_strings_with_log,
    _finalize_parsed_cv,
)


def test_fix_college_name():
    raw = "GayatriVidyaParishadCollegeOfEngineering"
    fixed = _fix_concatenated_words(raw)
    assert " " in fixed
    assert "College" in fixed and "Engineering" in fixed
    assert raw not in fixed


def test_fix_netflix_project():
    raw = "NetflixAssistGPT-AMovieCompanion"
    fixed = _fix_concatenated_words(raw)
    assert "Netflix" in fixed
    assert " " in fixed
    assert "GPT" in fixed


def test_fix_json_nested():
    data = {
        "education": [{"institution": "GayatriVidyaParishadCollegeOfEngineering"}],
        "projects": [{"name": "NetflixAssistGPT-AMovieCompanion", "description": "Built API"}],
    }
    fixed, examples = _fix_json_strings_with_log(data)
    assert " " in fixed["education"][0]["institution"]
    assert len(examples) >= 1


def test_finalize_no_concatenation_in_output():
    parsed = {
        "name": "Test",
        "contact_phone": "",
        "contact_email": "",
        "contact_location": "",
        "linkedin": "",
        "personal_statement": "Profile",
        "key_skills": [],
        "skills": [],
        "work_experience": [],
        "projects": [
            {
                "name": "NetflixAssistGPT-AMovieCompanion",
                "description": "desc",
                "technologies": [],
                "date": "",
                "link": "",
            }
        ],
        "education": [{"degree": "BSc", "institution": "GayatriVidyaParishadCollegeOfEngineering", "dates": "", "grade": ""}],
        "certifications": [],
        "interests": [],
        "references": "",
    }
    out = _finalize_parsed_cv(parsed)
    inst = out["education"][0]["institution"]
    proj = out["projects"][0]["name"]
    assert "Gayatri Vidya" in inst
    assert "College Of Engineering" in inst or "College of Engineering" in inst
    assert "Netflix Assist" in proj
    assert out.get("_spacing_auto_fixed")
