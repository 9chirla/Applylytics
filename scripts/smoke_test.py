#!/usr/bin/env python3
"""
Quick environment smoke test for Applylytics (no pytest).

Run from repo root:
  PYTHONPATH=. python scripts/smoke_test.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _pass(msg: str) -> None:
    print(f"PASS: {msg}")


def _fail(msg: str) -> None:
    print(f"FAIL: {msg}")


def _mock_streamlit() -> None:
    st = MagicMock()
    st.session_state = {}
    st.secrets = {}
    st.cache_resource = lambda **kwargs: (lambda fn: fn)
    st.cache_data = lambda **kwargs: (lambda fn: fn)
    st.fragment = lambda **kwargs: (lambda fn: fn)
    sys.modules["streamlit"] = st


def step_import_app() -> bool:
    try:
        _mock_streamlit()
        import app  # noqa: F401

        _pass("import app.py (Streamlit mocked)")
        return True
    except Exception as exc:
        _fail(f"import app.py — {exc}")
        return False


def step_preflight() -> bool:
    try:
        from applylytics.preflight import run_preflight

        status = run_preflight()
        print(f"  templates_ok={status.templates_ok} missing={status.missing_templates}")
        print(f"  groq_ok={status.groq_ok}")
        print(f"  redis_ok={status.redis_ok}")
        _pass("run_preflight()")
        return True
    except Exception as exc:
        _fail(f"run_preflight() — {exc}")
        return False


def step_ats_scorer() -> bool:
    try:
        from applylytics.ats.scorer import (
            calculate_ats_score,
            semantic_similarity_score,
        )

        cv = "Jane Doe. Python engineer. Skills: Python, SQL, Docker."
        jd = (
            "We are looking for a Python engineer. "
            "Requirements: Python, SQL. Responsibilities include API work."
        )
        kw = calculate_ats_score(cv, jd, include_semantic=False)
        keyword_score = int(kw.get("keyword_score", kw.get("score", 0)))
        if keyword_score <= 0:
            _fail(f"keyword score expected > 0, got {keyword_score}")
            return False
        sem = semantic_similarity_score(cv, jd)
        if sem is None:
            _fail("semantic score returned None")
            return False
        print(f"  keyword_score={keyword_score} semantic_score={sem}")
        _pass("ATS keyword + semantic scores")
        return True
    except Exception as exc:
        _fail(f"ATS scorer — {exc}")
        return False


def step_docx_render() -> bool:
    try:
        from applylytics.constants import CV_TEMPLATE_CHOICES
        from applylytics.cv.docx_generator import cv_context_from_structured, render_cv

        template = CV_TEMPLATE_CHOICES["Classic"]
        if not template.is_file():
            _fail(f"template missing: {template}")
            return False
        sample = {
            "name": "Smoke Test",
            "contact_email": "smoke@example.com",
            "personal_statement": "Engineer.",
            "work_experience": [
                {
                    "company": "Co",
                    "role": "Dev",
                    "dates": "2020 – Present",
                    "bullets": ["Built APIs."],
                }
            ],
            "projects": [],
            "education": [],
            "skills": ["Python"],
            "certifications": [],
        }
        context = cv_context_from_structured(sample)
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
            out = Path(tmp.name)
        try:
            render_cv(str(template), str(out), context)
            size = out.stat().st_size
            if size <= 0:
                _fail(f"DOCX size expected > 0, got {size}")
                return False
            print(f"  classic.docx bytes={size}")
            _pass("render classic.docx template")
            return True
        finally:
            out.unlink(missing_ok=True)
    except Exception as exc:
        _fail(f"DOCX render — {exc}")
        return False


def main() -> int:
    print("Applylytics smoke test\n")
    results = [
        step_import_app(),
        step_preflight(),
        step_ats_scorer(),
        step_docx_render(),
    ]
    print()
    if all(results):
        print("All smoke checks passed.")
        return 0
    print("One or more smoke checks failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
