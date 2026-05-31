"""Tests for CV vs job-description detection on extracted PDF text."""

from applylytics.cv.pdf import detect_is_job_description

SAMPLE_CV = """
Jane Doe
Software Engineer with five years of Python API development.
Experience at TechCo building REST services, unit tests, and CI pipelines.
Skills: Python, FastAPI, PostgreSQL, Docker, Git.
Education: BSc Computer Science, University of Example, 2018.
"""

SAMPLE_JD = """
We are looking for a Senior Python Engineer to join our platform team.
Responsibilities include designing APIs and mentoring engineers.
Requirements: Python, FastAPI, PostgreSQL, Docker, AWS.
Qualifications: 3+ years backend experience.
About the role: you will own services end-to-end.
You will collaborate with product and must have strong communication skills.
"""


def test_clear_job_description_detected():
    assert detect_is_job_description(SAMPLE_JD) is True


def test_clear_cv_not_detected():
    assert detect_is_job_description(SAMPLE_CV) is False


def test_one_phrase_not_detected():
    text = "Jane Doe\nWe are looking for new challenges in data engineering."
    assert detect_is_job_description(text) is False


def test_three_phrases_detected():
    text = (
        "Header\n"
        "We are looking for a candidate.\n"
        "Requirements: Python skills.\n"
        "You will work remotely.\n"
    )
    assert detect_is_job_description(text) is True


def test_case_insensitive_detection():
    text = (
        "WE ARE LOOKING FOR talent.\n"
        "RESPONSIBILITIES INCLUDE delivery.\n"
        "QUALIFICATIONS: degree.\n"
    )
    assert detect_is_job_description(text) is True


def test_empty_text_not_detected():
    assert detect_is_job_description("") is False
    assert detect_is_job_description("   ") is False
