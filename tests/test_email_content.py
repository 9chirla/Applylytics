"""Tests for email metadata and content helpers."""

from applylytics.email.content import build_cv_attachment_filename, build_email_subject
from applylytics.email.job_metadata import extract_job_metadata


def test_extract_labeled_fields():
    jd = """
    Job Title: Junior Data Analyst
    Company: Buitelaar Analytics Ltd
    Location: London (Hybrid)
    """
    meta = extract_job_metadata(jd)
    assert meta.job_title == "Junior Data Analyst"
    assert meta.company_name == "Buitelaar Analytics Ltd"
    assert "London" in meta.location


def test_extract_title_at_company():
    meta = extract_job_metadata("Graduate Engineer at Acme Corp\nLondon\nFull-time")
    assert meta.job_title == "Graduate Engineer"
    assert meta.company_name == "Acme Corp"


def test_attachment_filename():
    meta = extract_job_metadata("Data Analyst at BigCo")
    name = build_cv_attachment_filename(candidate_name="Jane Doe", job=meta)
    assert name.endswith(".docx")
    assert "Jane_Doe" in name
    assert "CV" in name
    assert "BigCo" in name or "Data_Analyst" in name


def test_subject_with_company():
    meta = extract_job_metadata("Role at Acme")
    assert "Acme" in build_email_subject(meta)


def test_reject_about_the_job_heading():
    meta = extract_job_metadata("About the job\n\nResponsibilities:\n- Do things")
    assert meta.job_title == "this role"
    assert not meta.has_company
    assert "About" not in build_email_subject(meta)


def test_company_override_wins():
    from applylytics.email.job_metadata import build_job_metadata

    meta = build_job_metadata(
        "About the job\nResponsibilities",
        company_override="Acme Ltd",
    )
    assert meta.company_name == "Acme Ltd"
    assert "Acme" in build_email_subject(meta)


def test_certification_dict_normalized():
    from applylytics.cv.schema import normalize_cv_json

    out = normalize_cv_json(
        {"certifications": [{"title": "Python for Data Science", "issuer": "IBM"}]}
    )
    assert out["certifications"] == ["Python for Data Science (IBM)"]
