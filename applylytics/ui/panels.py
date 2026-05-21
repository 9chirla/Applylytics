"""Streamlit panel renderers for upload, ATS, AI coach, and CV optimiser."""

from __future__ import annotations

import hashlib
import logging
import re
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import streamlit as st

from applylytics.ats.scorer import calculate_ats_score_cached
from applylytics.config import settings
from applylytics.constants import CV_TEMPLATE_DOCX, RATE_LIMIT_WARNING, SessionKey
from applylytics.cv.optimiser import _fix_concatenated_words, optimize_cv_generic
from applylytics.cv.renderer import (
    cv_context_from_structured,
    cv_plaintext_for_ats_scoring,
    render_cv_to_text,
    render_docx_to_bytes,
)
from applylytics.llm.client import RateLimitError
from applylytics.llm.coach import analyze_resume_with_job, get_hiring_manager_comment
from applylytics.ui.components import (
    EMPTY_STATE_HTML,
    SKELETON_HTML,
    emit_html,
    render_field_label,
    render_hiring_manager_comment,
    render_insight,
    render_action_buttons_marker,
    render_keyword_section,
    render_results_label,
    render_score_panel,
    render_section_header,
)
from utils.pdf_extractor import extract_text_from_pdf

logger = logging.getLogger("applylytics")


@contextmanager
def _skeleton() -> Iterator[None]:
    ph = st.empty()
    ph.markdown(SKELETON_HTML, unsafe_allow_html=True)
    try:
        yield
    finally:
        ph.empty()


def _phase_key(resume_text: str, job_description: str) -> str:
    return (
        f"{hashlib.sha256(resume_text.encode()).hexdigest()[:16]}:"
        f"{hashlib.sha256(job_description.encode()).hexdigest()[:24]}"
    )


@st.cache_data(show_spinner=False)
def _cached_extract_pdf(file_hash: str, file_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        path = tmp.name
    try:
        return extract_text_from_pdf(path)
    finally:
        Path(path).unlink(missing_ok=True)


def _extract_resume_text(file_bytes: bytes) -> str:
    file_hash = hashlib.sha256(file_bytes).hexdigest()
    return _cached_extract_pdf(file_hash, file_bytes)


def _has_resume_bytes() -> bool:
    raw = st.session_state.get(SessionKey.RESUME_BYTES)
    return isinstance(raw, (bytes, bytearray)) and len(raw) > 0


def _get_job_description() -> str:
    """Job description from widget session state (with code/prompt guard)."""
    jd = st.session_state.get("job_description_input", "") or ""
    if any(
        marker in jd
        for marker in (
            "STRICT_SYSTEM_PROMPT",
            "cv_optimizer.py",
            "build_targeted_keyword_requirements",
            "def render_",
        )
    ):
        return ""
    return jd.strip()


def _input_signature(has_resume: bool, job_description: str) -> str:
    return hashlib.sha256(f"{has_resume}:{job_description}".encode()).hexdigest()[:20]


def _sync_input_signature(has_resume: bool, job_description: str) -> None:
    """Reset cached results when resume or JD changes."""
    sig = _input_signature(has_resume, job_description)
    if st.session_state.get(SessionKey.INPUT_SIGNATURE) == sig:
        return
    st.session_state[SessionKey.INPUT_SIGNATURE] = sig
    st.session_state[SessionKey.ANALYSE_DONE_ID] = st.session_state.get(
        SessionKey.ANALYSE_RUN_ID, 0
    )
    for key in (
        SessionKey.ATS_RESULT,
        SessionKey.RESUME_TEXT,
        SessionKey.OPTIMIZED_DATA,
        SessionKey.MANAGER_COMMENT,
        SessionKey.HM_FEEDBACK_CACHE_KEY,
    ):
        st.session_state.pop(key, None)


def _request_analyse() -> None:
    st.session_state[SessionKey.ANALYSE_RUN_ID] = (
        int(st.session_state.get(SessionKey.ANALYSE_RUN_ID, 0)) + 1
    )


def build_step_status(has_resume: bool, job_description: str) -> dict[str, bool]:
    """Step indicator state for hero (upload → JD → analyse → optimise)."""
    od = st.session_state.get(SessionKey.OPTIMIZED_DATA)
    optimised = isinstance(od, dict) and not od.get("error")
    return {
        "upload": has_resume,
        "jd": bool((job_description or "").strip()),
        "analyse": int(st.session_state.get(SessionKey.ANALYSE_DONE_ID, 0)) > 0,
        "optimise": optimised,
    }


def render_upload_panel() -> tuple[str, bool]:
    """Left column: file uploader + job description."""
    render_section_header(
        "Step 2 · Upload",
        "Your resume +",
        "the role.",
        "Upload your CV and paste the job description.<br>"
        "We'll score the match and tell you exactly what's missing.",
    )
    render_field_label("01 — Resume")
    uploaded = st.file_uploader(
        "Resume (PDF)",
        type=["pdf"],
        help="Upload the CV you want scored against the role.",
        label_visibility="collapsed",
        key="resume_pdf_upload",
    )
    render_field_label("02 — Job description")
    job_description = st.text_area(
        "Job description",
        height=130,
        placeholder="Paste the full job posting — requirements, responsibilities, tools...",
        help="Used for ATS keyword overlap and all AI prompts.",
        label_visibility="collapsed",
        key="job_description_input",
    )
    if uploaded is not None:
        st.session_state[SessionKey.RESUME_BYTES] = uploaded.getvalue()

    has_resume = _has_resume_bytes()

    job_description = _get_job_description()
    if (st.session_state.get("job_description_input", "") or "").strip() and not job_description:
        st.warning(
            "Job description looks like code or a prompt — please "
            "paste the actual job posting text."
        )

    _sync_input_signature(has_resume, job_description)

    can_analyse = has_resume and bool(job_description)
    if st.button(
        "Analyse match →",
        type="primary",
        use_container_width=True,
        disabled=not can_analyse,
        key="btn_analyse_match",
        help="Run ATS scoring and AI insights for this resume and job description.",
    ):
        _request_analyse()
        st.session_state["_al_scroll_to"] = "al-snap-results"

    run_id = int(st.session_state.get(SessionKey.ANALYSE_RUN_ID, 0))
    done_id = int(st.session_state.get(SessionKey.ANALYSE_DONE_ID, 0))
    if can_analyse and run_id <= done_id:
        st.caption("Click **Analyse match →** to run scoring.")

    return job_description, has_resume


def _render_spacing_fix_warning(spacing_fixes: list | None) -> None:
    if spacing_fixes:
        st.warning(
            "Some institution or project names were missing spaces and were auto-corrected: "
            + "; ".join(str(x) for x in spacing_fixes[:5])
            + (" …" if len(spacing_fixes) > 5 else "")
        )


def _run_ats_scoring(resume_text: str, job_description: str) -> dict:
    """Score resume against JD (no UI — safe inside st.status)."""
    resume_text = _fix_concatenated_words(resume_text)
    phase_key = _phase_key(resume_text, job_description)
    if st.session_state.get(SessionKey.PHASE3_KEY) != phase_key:
        st.session_state[SessionKey.PHASE3_KEY] = phase_key
        for key in (
            SessionKey.MANAGER_COMMENT,
            SessionKey.HM_FEEDBACK_CACHE_KEY,
            SessionKey.OPTIMIZED_DATA,
        ):
            st.session_state.pop(key, None)
    return calculate_ats_score_cached(resume_text, job_description)


def _render_resume_preview(resume_text: str) -> None:
    with st.expander(":material/description: Extracted resume text (preview)", expanded=False):
        st.text_area(
            "Resume text",
            value=resume_text[:8000] + ("…" if len(resume_text) > 8000 else ""),
            height=220,
            label_visibility="collapsed",
        )


def render_ats_panel(resume_text: str, job_description: str) -> dict:
    """ATS scoring and results display (must not run inside st.status)."""
    with st.spinner("Scoring whitelisted skill overlap against this role…"):
        ats_result = _run_ats_scoring(resume_text, job_description)
    st.session_state[SessionKey.ATS_RESULT] = ats_result
    _render_resume_preview(resume_text)
    _display_ats_results(ats_result)
    return ats_result


def _display_ats_results(ats_result: dict) -> None:
    """Render ATS score UI from a cached result (no re-scoring)."""
    render_score_panel(ats_result)
    render_keyword_section(
        "Matched keywords",
        ats_result["matched_keywords"],
        "matched",
        len(ats_result["matched_keywords"]),
    )
    render_keyword_section(
        "Missing keywords",
        ats_result["missing_keywords"],
        "missing",
        len(ats_result["missing_keywords"]),
    )
    insight = ats_result.get("insight")
    if insight:
        render_insight(str(insight))


def _sanitize_coach_display(text: str) -> str:
    """Strip markdown headings and fenced code blocks from coach output."""
    lines: list[str] = []
    for line in text.splitlines():
        if line.strip().startswith("```"):
            continue
        if re.match(r"^#{1,6}\s+", line):
            heading = re.sub(r"^#{1,6}\s+", "", line).strip()
            lines.append(f"**{heading}**")
        else:
            lines.append(line)
    return "\n".join(lines)


def render_ai_coach_panel(
    resume_text: str,
    job_description: str,
    ats_result: dict,
) -> None:
    """Hiring manager comment and general AI feedback."""
    phase_key = _phase_key(resume_text, job_description)
    od_hm = st.session_state.get(SessionKey.OPTIMIZED_DATA)
    opt_plain = ""
    if isinstance(od_hm, dict) and not od_hm.get("error"):
        opt_plain = render_cv_to_text(od_hm).strip()
    fp = hashlib.sha256(opt_plain.encode()).hexdigest()[:20] if opt_plain else "none"
    hm_fb_key = f"{phase_key}:{fp}"

    if st.session_state.get(SessionKey.HM_FEEDBACK_CACHE_KEY) != hm_fb_key:
        try:
            with st.spinner("Getting honest feedback from a hiring manager…"):
                comment = get_hiring_manager_comment(
                    resume_text,
                    job_description,
                    ats_result["score"],
                    optimised_text=opt_plain if opt_plain else None,
                )
        except RateLimitError:
            st.warning(RATE_LIMIT_WARNING)
            return
        if comment is None:
            return
        if comment.startswith("Groq API error"):
            st.error(comment)
            return
        st.session_state[SessionKey.MANAGER_COMMENT] = comment
        st.session_state[SessionKey.HM_FEEDBACK_CACHE_KEY] = hm_fb_key
    else:
        comment = st.session_state.get(SessionKey.MANAGER_COMMENT)

    render_hiring_manager_comment(comment)

    with st.expander("More coaching tools", expanded=False):
        if st.button("Get general AI feedback", use_container_width=True):
            try:
                with _skeleton():
                    with st.spinner("Analyzing with Groq Llama 3.3…"):
                        feedback = analyze_resume_with_job(resume_text, job_description)
            except RateLimitError:
                st.warning(RATE_LIMIT_WARNING)
                return
            if feedback is None:
                return
            if isinstance(feedback, str) and feedback.startswith("Groq API error"):
                st.error(feedback)
                return
            feedback = _sanitize_coach_display(feedback)
            st.markdown(feedback)


def _render_ats_debug_expander(debug: dict) -> None:
    with st.expander("ATS optimisation debug", expanded=False):
        st.markdown("**Keywords sent to the model**")
        sent = debug.get("keywords_sent") or []
        st.write(", ".join(sent) if sent else "_(none)_")
        st.markdown("**Now matched after optimisation**")
        matched_after = debug.get("matched_after") or []
        st.write(", ".join(matched_after) if matched_after else "_(none from the sent list)_")
        still = debug.get("still_missing") or []
        if still:
            st.markdown("**Still missing from sent list**")
            st.write(", ".join(still))
        c1, c2 = st.columns(2)
        with c1:
            st.metric("ATS score (before)", f"{debug.get('score_before', '—')}%")
        with c2:
            st.metric("ATS score (after)", f"{debug.get('score_after', '—')}%")
        _render_spacing_fix_warning(debug.get("spacing_auto_fixed"))


def render_optimiser_panel(
    resume_text: str,
    job_description: str,
    ats_result: dict,
) -> None:
    """Optimise CV + DOCX download action row."""
    from utils.cv_optimizer import build_targeted_keyword_requirements, get_gemini_client

    missing_kw = ats_result["missing_keywords"]
    od = st.session_state.get(SessionKey.OPTIMIZED_DATA)
    optimised_ready = isinstance(od, dict) and not od.get("error")

    render_action_buttons_marker()
    act1, act2 = st.columns(2)
    with act1:
        if st.button(
            "✦ Optimise CV →",
            type="primary",
            use_container_width=True,
            key="btn_optimise_cv",
        ):
            try:
                ats = st.session_state.get(SessionKey.ATS_RESULT, {})
                targeted_block = build_targeted_keyword_requirements(
                    missing_keywords=ats.get("missing_keywords", []),
                    matched_keywords=ats.get("matched_keywords", []),
                    job_description=job_description,
                )
                with _skeleton():
                    _opt_label = (
                        "Running CV optimiser (Groq, Gemini fallback if needed)…"
                        if get_gemini_client()
                        else "Running generic CV optimiser (Groq)…"
                    )
                    with st.spinner(_opt_label):
                        result = optimize_cv_generic(
                            resume_text,
                            job_description,
                            missing_kw,
                            keyword_requirements_block=targeted_block,
                        )
            except RateLimitError as exc:
                st.warning(str(exc).strip() or RATE_LIMIT_WARNING)
                return
            if result.get("error"):
                st.error(result["error"])
                return
            st.session_state[SessionKey.OPTIMIZED_DATA] = result
            try:
                st.session_state[SessionKey.OPTIMISED_CV] = render_cv_to_text(result)
            except Exception:
                st.session_state[SessionKey.OPTIMISED_CV] = str(result)
                st.warning("Could not format optimised CV as text; showing raw summary.")
            st.session_state.pop(SessionKey.HM_FEEDBACK_CACHE_KEY, None)
            done_msg = "Optimisation complete. Download your DOCX below."
            ach = result.get("_ats_score_achieved")
            if isinstance(ach, int):
                done_msg += f" Estimated whitelist overlap after this pass: {ach}%."
            st.success(done_msg)
            if settings.debug_mode:
                debug = result.get("_ats_debug")
                if isinstance(debug, dict):
                    _render_ats_debug_expander(debug)
            st.rerun()
    with act2:
        _render_download_button(resume_text, od if optimised_ready else None)

    with st.expander("Advanced · reset optimisation", expanded=False):
        if st.button("Reset optimised CV", use_container_width=True):
            for key in (
                SessionKey.OPTIMIZED_DATA,
                SessionKey.HM_FEEDBACK_CACHE_KEY,
                SessionKey.MANAGER_COMMENT,
                SessionKey.ATS_RESULT,
            ):
                st.session_state.pop(key, None)
            st.rerun()

    od = st.session_state.get(SessionKey.OPTIMIZED_DATA)
    if isinstance(od, dict) and not od.get("error"):
        if settings.debug_mode:
            debug = od.get("_ats_debug")
            if isinstance(debug, dict):
                _render_ats_debug_expander(debug)
            spacing_on_cv = od.get("_spacing_auto_fixed")
            if spacing_on_cv:
                _render_spacing_fix_warning(spacing_on_cv)
            with st.expander("Preview optimised data (JSON)", expanded=False):
                projects = od.get("projects") or []
                if projects:
                    st.markdown("**Projects**")
                    for proj in projects:
                        if isinstance(proj, dict):
                            st.markdown(
                                f"- **{proj.get('name', 'Project')}** ({proj.get('date', '')})"
                            )
                            if proj.get("description"):
                                st.caption(str(proj["description"])[:500])
                st.json(od)
        if st.session_state.get(SessionKey.OPTIMISED_CV):
            with st.expander("What was changed", expanded=False):
                st.markdown(
                    """
**This optimised CV has been rewritten to:**
- Tailor the profile to the target industry in the job description
- Add work bullet evidence for skills that were listed but unsupported
- Consolidate the skills section to 8–10 relevant items
- Preserve all your original dates, employers, and qualifications
"""
                )


def _render_download_button(resume_text: str, od: dict | None) -> None:
    """Download DOCX in the action row (disabled until optimisation completes)."""
    if not isinstance(od, dict) or od.get("error"):
        st.download_button(
            label="↓ Download DOCX",
            data=b"",
            file_name="CV.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True,
            key="dl_docx_optimised",
            disabled=True,
        )
        return
    if not CV_TEMPLATE_DOCX.is_file():
        st.error(f"Template missing: {CV_TEMPLATE_DOCX}")
        return
    try:
        docx_bytes = render_docx_to_bytes(
            CV_TEMPLATE_DOCX,
            cv_context_from_structured(od, source_resume_text=resume_text),
        )
        safe_name = re.sub(
            r"[^\w\-]+",
            "_",
            (od.get("name") or "CV").strip(),
        ).strip("_")[:80] or "CV"
        st.download_button(
            label="↓ Download DOCX",
            data=docx_bytes,
            file_name=f"{safe_name}_CV.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True,
            key="dl_docx_optimised",
        )
    except Exception as exc:
        st.error(f"DOCX render failed: {exc}")


def render_results_container(
    job_description: str,
    has_resume: bool,
) -> None:
    """Right column: extract PDF, run ATS + optimiser + coach panels."""
    render_section_header(
        "Step 3 · Results",
        "Your ATS",
        "results.",
        "Keyword scores, coaching, and your optimised CV export appear here after you analyse.",
    )

    job_description = _get_job_description()
    has_resume = _has_resume_bytes()
    ready = has_resume and bool(job_description)
    run_id = int(st.session_state.get(SessionKey.ANALYSE_RUN_ID, 0))
    done_id = int(st.session_state.get(SessionKey.ANALYSE_DONE_ID, 0))
    pending_run = ready and run_id > done_id

    if pending_run:
        resume_bytes = st.session_state.get(SessionKey.RESUME_BYTES)
        if not isinstance(resume_bytes, (bytes, bytearray)) or not resume_bytes:
            st.error(
                "Resume data missing — upload your PDF again, "
                "then click **Analyse match →**."
            )
            return
        with st.spinner("Running ATS analysis…"):
            try:
                resume_text = _extract_resume_text(bytes(resume_bytes))
            except Exception as exc:
                st.error(f"Could not read PDF: {exc}")
                return
            if not resume_text or len(resume_text.strip()) < 80:
                st.error(
                    "Could not extract text from this PDF — it may be "
                    "scanned or image-only. Re-export as a text-based PDF "
                    "and try again."
                )
                return
            ats_result = _run_ats_scoring(resume_text, job_description)
            st.session_state[SessionKey.RESUME_TEXT] = resume_text
            st.session_state[SessionKey.ATS_RESULT] = ats_result
            st.session_state[SessionKey.ANALYSE_DONE_ID] = run_id

        _render_resume_preview(resume_text)
        _display_ats_results(ats_result)
        render_optimiser_panel(resume_text, job_description, ats_result)
        render_ai_coach_panel(resume_text, job_description, ats_result)
        return

    cached_ats = st.session_state.get(SessionKey.ATS_RESULT)
    cached_resume = st.session_state.get(SessionKey.RESUME_TEXT)
    if ready and run_id > 0 and isinstance(cached_ats, dict) and cached_resume:
        _render_resume_preview(str(cached_resume))
        _display_ats_results(cached_ats)
        render_optimiser_panel(str(cached_resume), job_description, cached_ats)
        render_ai_coach_panel(str(cached_resume), job_description, cached_ats)
        return

    if ready:
        st.info("Click **Analyse match →** on the left to run ATS scoring and AI insights.")
    elif has_resume and not job_description:
        st.info("Add a job description on the left, then click **Analyse match →**.")
    elif job_description and not has_resume:
        st.info("Upload your resume PDF on the left to continue.")
    else:
        emit_html(EMPTY_STATE_HTML)
