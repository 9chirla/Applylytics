"""Cached ATS scoring (re-exports from utils)."""

from __future__ import annotations

import hashlib

import streamlit as st

from utils.ats_analyzer import calculate_ats_score as _calculate_ats_score


@st.cache_data(show_spinner=False, ttl=3600)
def _cached_ats_score(resume_hash: str, job_hash: str, resume_text: str, job_text: str) -> dict:
    return _calculate_ats_score(resume_text, job_text)


def calculate_ats_score_cached(resume_text: str, job_text: str) -> dict:
    """Whitelist ATS overlap score with Streamlit cache keyed by content hashes."""
    rh = hashlib.sha256(resume_text.encode()).hexdigest()
    jh = hashlib.sha256(job_text.encode()).hexdigest()
    return _cached_ats_score(rh, jh, resume_text, job_text)
