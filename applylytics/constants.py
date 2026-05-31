"""Shared constants and session-state keys."""

from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
TEMPLATE_DIR = APP_DIR / "templates"
CV_TEMPLATE_DOCX = TEMPLATE_DIR / "MasterTemplate.docx"
ASSETS_DIR = APP_DIR / "assets"

CV_TEMPLATE_CHOICES: dict[str, Path] = {
    "Classic": TEMPLATE_DIR / "classic.docx",
    "Modern": TEMPLATE_DIR / "Modern.docx",
    "ATS-safe": TEMPLATE_DIR / "ats_safe.docx",
}
DEFAULT_CV_TEMPLATE = "Classic"


def resolve_cv_template(name: str | None = None) -> Path:
    """Return path to selected DOCX template; falls back to classic then MasterTemplate."""
    key = (name or DEFAULT_CV_TEMPLATE).strip()
    path = CV_TEMPLATE_CHOICES.get(key)
    if path and path.is_file():
        return path
    if CV_TEMPLATE_CHOICES["Classic"].is_file():
        return CV_TEMPLATE_CHOICES["Classic"]
    return CV_TEMPLATE_DOCX

RATE_LIMIT_WARNING = (
    "⏳ Groq rate limit (TPM). Wait ~10 seconds and try again. "
    "The app retries with llama-3.1-8b-instant automatically; for fewer 429s set "
    "GROQ_MODEL=llama-3.1-8b-instant in .env, or add a valid GEMINI_API_KEY."
)

TRUNCATION_WARNING = (
    "⚠️ Input text was trimmed to fit the model context window. "
    "Check your PDF export settings if content looks incomplete."
)

# Groq chat completions reject very large HTTP bodies (413). Keep inputs conservative.
MAX_TEXT_CHARS = 24_000
MAX_COMBINED_CHARS = 32_000
MAX_CV_OPTIMIZER_CHARS = 18_000
# Background/email optimise path — smaller payloads, fewer Groq tokens.
MAX_CV_OPTIMIZER_CHARS_FAST = 12_000
MAX_JOB_CONTEXT_CHARS_FAST = 6_000


class SessionKey:
    """Streamlit session_state keys."""

    PREFLIGHT = "preflight_status"
    OPTIMIZED_DATA = "optimized_data"
    OPTIMISED_ATS_RESULT = "optimised_ats_result"
    MANAGER_COMMENT = "manager_comment"
    OPTIMISED_MANAGER_COMMENT = "optimised_manager_comment"
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
    CV_TEMPLATE = "cv_template_choice"
