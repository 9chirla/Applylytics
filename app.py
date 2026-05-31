"""Applylytics — AI Resume Analyzer (Streamlit entrypoint)."""

from __future__ import annotations

import logging
import secrets
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Local only: .env is gitignored and absent on Streamlit Cloud (use Secrets there).
_env_file = Path(__file__).resolve().parent / ".env"
if _env_file.is_file():
    load_dotenv(_env_file, override=True)

from applylytics.config import bootstrap_settings, settings

# Streamlit Cloud: load secrets before panels/LLM modules import settings.
bootstrap_settings()
from applylytics.constants import ASSETS_DIR, SessionKey  # noqa: E402
from applylytics.preflight import PreflightStatus, run_preflight, service_indicator
from applylytics.ui.components import render_hero_section, render_top_bar, scroll_to_snap
from applylytics.ui.panels import (
    build_step_status,
    render_results_container,
    render_upload_panel,
)

logger = logging.getLogger("applylytics")


def inject_design_system() -> None:
    """Inject global CSS."""
    css = (ASSETS_DIR / "design.css").read_text()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def _ensure_preflight() -> PreflightStatus:
    if SessionKey.PREFLIGHT not in st.session_state:
        st.session_state[SessionKey.PREFLIGHT] = run_preflight()
    return st.session_state[SessionKey.PREFLIGHT]


def _render_preflight_sidebar(status: PreflightStatus) -> None:
    with st.sidebar:
        st.markdown("### Service status")
        st.markdown(
            f"{service_indicator(status.templates_ok)} **CV templates**"
            + (" — missing: " + ", ".join(status.missing_templates) if status.missing_templates else "")
        )
        st.markdown(
            f"{service_indicator(status.groq_ok)} **Groq LLM**"
            + ("" if status.groq_ok else " — set `GROQ_API_KEY` in `.env` or Secrets")
        )
        redis_label = "Redis / email queue"
        if status.redis_ok:
            st.markdown(f"{service_indicator(True)} **{redis_label}**")
        else:
            st.markdown(
                f"{service_indicator(False, degraded=True)} **{redis_label}** — "
                "start Redis and `rq worker --with-scheduler`"
            )
        if not status.llm_enabled:
            st.caption("AI scoring and optimisation are disabled until Groq is configured.")
        if not status.email_enabled:
            st.caption("Background email uses in-app send when Redis is unavailable.")


def _check_auth() -> None:
    pwd = (settings.app_password or "").strip()
    if not pwd:
        return
    if st.session_state.get(SessionKey.AUTHED):
        return
    entered = st.text_input("Password", type="password")
    if st.button("Login"):
        if secrets.compare_digest(entered, pwd):
            st.session_state[SessionKey.AUTHED] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    try:
        from applylytics.llm.groq_optimizer import reset_gemini_client, reset_groq_client

        reset_groq_client()
        reset_gemini_client()
    except ImportError:
        pass
    st.set_page_config(
        page_title="Applylytics",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    preflight = _ensure_preflight()
    _render_preflight_sidebar(preflight)

    if not preflight.templates_ok:
        st.warning(
            "Some CV templates are missing: "
            + ", ".join(preflight.missing_templates)
            + ". DOCX export may fall back to MasterTemplate."
        )

    _check_auth()
    portfolio_url = (
        (settings.portfolio_url or "https://github.com/9chirla").strip()
        or "https://github.com/9chirla"
    )
    inject_design_system()
    render_top_bar(portfolio_url, portfolio_url)

    # Step status
    _rb = st.session_state.get(SessionKey.RESUME_BYTES)
    has_resume_pre = isinstance(_rb, (bytes, bytearray)) and len(_rb) > 0
    jd_pre = (st.session_state.get("job_description_input", "") or "").strip()
    company_pre = (st.session_state.get("target_company_input", "") or "").strip()
    step_status = build_step_status(has_resume_pre, jd_pre, company_pre)

    # Screen 1: Hero
    st.markdown(render_hero_section(step_status), unsafe_allow_html=True)

    # Screen 2: Input (st.container groups widgets so snap CSS applies)
    with st.container():
        st.markdown('<div class="al-snap-input" aria-hidden="true"></div>', unsafe_allow_html=True)
        job_description, has_resume, _company = render_upload_panel()

    # Screen 3: Results
    with st.container():
        st.markdown('<div class="al-snap-results" aria-hidden="true"></div>', unsafe_allow_html=True)
        render_results_container(job_description, has_resume)

    scroll_target = st.session_state.pop("_al_scroll_to", None)
    if scroll_target:
        scroll_to_snap(scroll_target)


if __name__ == "__main__":
    main()
