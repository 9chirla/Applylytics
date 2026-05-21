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

from applylytics.config import settings
from applylytics.constants import ASSETS_DIR, CV_TEMPLATE_DOCX, SessionKey
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
        from utils.cv_optimizer import reset_groq_client

        reset_groq_client()
    except ImportError:
        pass
    st.set_page_config(
        page_title="Applylytics",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    if not CV_TEMPLATE_DOCX.is_file():
        st.error(f"Template missing: {CV_TEMPLATE_DOCX}")
        st.stop()
    _check_auth()
    portfolio_url = (settings.portfolio_url or "https://github.com").strip() or "https://github.com"
    inject_design_system()
    render_top_bar(portfolio_url, portfolio_url)

    # Step status
    _rb = st.session_state.get(SessionKey.RESUME_BYTES)
    has_resume_pre = isinstance(_rb, (bytes, bytearray)) and len(_rb) > 0
    jd_pre = (st.session_state.get("job_description_input", "") or "").strip()
    step_status = build_step_status(has_resume_pre, jd_pre)

    # Screen 1: Hero (snap section via CSS class on .al-hero itself)
    st.markdown(render_hero_section(step_status), unsafe_allow_html=True)

    # Screen 2: Input (st.container groups widgets so snap CSS applies)
    with st.container():
        st.markdown('<div class="al-snap-input" aria-hidden="true"></div>', unsafe_allow_html=True)
        job_description, has_resume = render_upload_panel()

    # Screen 3: Results
    with st.container():
        st.markdown('<div class="al-snap-results" aria-hidden="true"></div>', unsafe_allow_html=True)
        render_results_container(job_description, has_resume)

    scroll_target = st.session_state.pop("_al_scroll_to", None)
    if scroll_target:
        scroll_to_snap(scroll_target)


if __name__ == "__main__":
    main()
