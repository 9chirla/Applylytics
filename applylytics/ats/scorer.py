"""ATS scoring: keyword overlap (whitelist) + optional semantic similarity."""

from __future__ import annotations

import hashlib
import logging
from typing import Any

import streamlit as st

from applylytics.ats.keyword_scorer import calculate_ats_score as _keyword_ats_score

logger = logging.getLogger("applylytics")

_SEMANTIC_MODEL_FALLBACK = None


@st.cache_resource(show_spinner=False)
def _get_semantic_model():
    """Load the embedding model once per Streamlit server process."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("all-MiniLM-L6-v2")


def _get_semantic_model_for_tests():
    """Non-Streamlit fallback (pytest) when cache_resource is unavailable."""
    global _SEMANTIC_MODEL_FALLBACK
    if _SEMANTIC_MODEL_FALLBACK is None:
        from sentence_transformers import SentenceTransformer

        _SEMANTIC_MODEL_FALLBACK = SentenceTransformer("all-MiniLM-L6-v2")
    return _SEMANTIC_MODEL_FALLBACK


def load_semantic_model():
    """Return the cached SentenceTransformer instance."""
    try:
        return _get_semantic_model()
    except Exception:
        return _get_semantic_model_for_tests()


def semantic_similarity_score(resume_text: str, job_text: str) -> int | None:
    """
    Cosine similarity between CV and job description embeddings (0–100).
    Returns None if sentence-transformers is unavailable.
    """
    if not (resume_text or "").strip() or not (job_text or "").strip():
        return 0
    try:
        from sklearn.metrics.pairwise import cosine_similarity

        model = load_semantic_model()
        cv_snip = (resume_text or "")[:12_000]
        jd_snip = (job_text or "")[:12_000]
        embeddings = model.encode([cv_snip, jd_snip])
        sim = float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0])
        return max(0, min(100, round(sim * 100)))
    except Exception as exc:
        logger.warning("Semantic ATS score skipped: %s", exc)
        return None


def calculate_ats_score(resume_text: str, job_text: str, *, include_semantic: bool = True) -> dict[str, Any]:
    """Keyword whitelist score plus optional embedding-based semantic score."""
    result = _keyword_ats_score(resume_text, job_text)
    result["keyword_score"] = result.get("score", 0)
    if include_semantic:
        sem = semantic_similarity_score(resume_text, job_text)
        result["semantic_score"] = sem
        if sem is not None:
            result["semantic_insight"] = (
                "How closely your CV and the job description read as similar topics overall "
                "(meaning), not just shared skill words."
            )
    result["keyword_insight"] = result.get("insight") or (
        "Overlap between whitelisted skill phrases in the job posting and your CV."
    )
    return result


@st.cache_data(show_spinner=False, ttl=3600)
def _cached_ats_score(resume_hash: str, job_hash: str, resume_text: str, job_text: str) -> dict:
    return calculate_ats_score(resume_text, job_text)


def calculate_ats_score_cached(resume_text: str, job_text: str) -> dict:
    """Streamlit-cached ATS scores keyed by content hashes."""
    rh = hashlib.sha256(resume_text.encode()).hexdigest()
    jh = hashlib.sha256(job_text.encode()).hexdigest()
    return _cached_ats_score(rh, jh, resume_text, job_text)
