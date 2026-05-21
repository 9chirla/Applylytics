"""Shared constants and session-state keys."""

from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
TEMPLATE_DIR = APP_DIR / "templates"
CV_TEMPLATE_DOCX = TEMPLATE_DIR / "MasterTemplate.docx"
ASSETS_DIR = APP_DIR / "assets"

RATE_LIMIT_WARNING = (
    "⏳ Groq and/or Gemini rate limits were hit. Wait about a minute and try again. "
    "If Gemini is configured, the app already retried with your fallback model "
    "(default gemini-2.5-flash via GEMINI_MODEL in .env)."
)

TRUNCATION_WARNING = (
    "⚠️ Input text was trimmed to fit the model context window. "
    "Check your PDF export settings if content looks incomplete."
)

# Groq chat completions reject very large HTTP bodies (413). Keep inputs conservative.
MAX_TEXT_CHARS = 24_000
MAX_COMBINED_CHARS = 32_000
MAX_CV_OPTIMIZER_CHARS = 18_000


class SessionKey:
    """Streamlit session_state keys."""

    OPTIMIZED_DATA = "optimized_data"
    MANAGER_COMMENT = "manager_comment"
    HM_FEEDBACK_CACHE_KEY = "_hm_feedback_cache_key"
    PHASE3_KEY = "_phase3_key"
    AUTHED = "_authed"
    OPTIMISED_CV = "optimised_cv"
    OPTIMISED_DOCX = "optimised_docx"
    ATS_RESULT = "ats_result"
    ANALYSE_RUN_ID = "analyse_run_id"
    ANALYSE_DONE_ID = "analyse_done_id"
    INPUT_SIGNATURE = "_input_signature"
    RESUME_BYTES = "resume_pdf_bytes"
    RESUME_TEXT = "resume_text_cache"
