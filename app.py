"""Applylytics — AI Resume Analyzer (Streamlit)."""
import hashlib
import html
import json
import os
import re
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from utils import calculate_ats_score, calculate_recruiter_penalties, extract_text_from_pdf
from utils.cv_generator import cv_context_from_structured, render_cv
from utils.cv_optimizer import is_groq_rate_limit_error, safe_groq_chat_create

load_dotenv()



class SessionKey:
    OPTIMIZED_DATA = "optimized_data"
    DOCX_BYTES = "_docx_bytes"
    MANAGER_COMMENT = "manager_comment"
    HM_FEEDBACK_CACHE_KEY = "_hm_feedback_cache_key"
    CV_FROM_REVIEW_PLAN = "_cv_from_review_plan"
    INTERVIEW_QUESTIONS = "interview_questions_md"
    IMPROVEMENT_PLAN = "improvement_plan_md"
    PHASE3_KEY = "_phase3_key"
    UI_WORD_COUNT = "_ui_word_count"
    UI_MATCH_PCT = "_ui_match_pct"
    UI_JOB_KW_COUNT = "_ui_job_kw_count"


_RATE_LIMIT_WARNING = "⏳ Groq rate limit hit — wait 30 seconds and try again."

_TRUNCATION_WARNING = (
    "⚠️ Resume text exceeds 60,000 characters and has been trimmed. "
    "Only the first portion was sent to the AI. Check your PDF export settings."
)


def _groq_api_key() -> str:
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        raise EnvironmentError(
            "GROQ_API_KEY is not set. Add it to your .env file. "
            "Do not reuse OPENAI_API_KEY — they go to different servers."
        )
    return key


def _require_groq_api_key() -> str:
    try:
        return _groq_api_key()
    except EnvironmentError as e:
        st.error(str(e))
        st.stop()


def _render_docx_to_bytes(template_path: Path, context: dict) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
        out_path = tmp.name
    try:
        render_cv(str(template_path), out_path, context)
        return Path(out_path).read_bytes()
    finally:
        Path(out_path).unlink(missing_ok=True)


_APP_DIR = Path(__file__).resolve().parent
_TEMPLATE_DIR = _APP_DIR / "templates"
CV_TEMPLATE_DOCX = _TEMPLATE_DIR / "MasterTemplate.docx"

_MODEL = (os.getenv("GROQ_MODEL") or "llama-3.3-70b-versatile").strip() or "llama-3.3-70b-versatile"

# Design system — single injectable block (dark theme, accessible palette)

def inject_design_system() -> None:
    css = (Path(__file__).parent / "assets" / "design.css").read_text()
    st.markdown(
        '<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css" crossorigin="anonymous" />',
        unsafe_allow_html=True,
    )
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)



def analyze_resume_with_job(resume_text: str, job_text: str) -> str | None:
    """Send both resume and job description to Groq for targeted feedback. Returns None on rate limit (UI silent)."""
    api_key = _require_groq_api_key()
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )

    if len(resume_text) > 60000:
        st.warning(_TRUNCATION_WARNING)
    if len(job_text) > 60000:
        st.warning(_TRUNCATION_WARNING)

    prompt = f"""
Resume:
{resume_text[:60000]}

Job Description:
{job_text[:60000]}
"""

    response, err = safe_groq_chat_create(
        client,
        model=_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are an expert resume coach and ATS specialist. Analyze the resume against the job description. Provide concise feedback: strengths (what matches well), gaps (what's missing from the job description), and 3-5 actionable improvements specifically tailored to this job.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
    )
    if err:
        return None if is_groq_rate_limit_error(err) else err
    return response.choices[0].message.content or ""


def get_hiring_manager_comment(
    resume_text: str,
    job_text: str,
    ats_score: int,
    optimised_text: str | None = None,
) -> str | None:
    """
    Return a detailed, honest, actionable hiring manager brief (roughly 100–200 words).
    Returns None on Groq rate limit (UI silent).
    """
    api_key = _require_groq_api_key()
    client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")

    if optimised_text and optimised_text.strip():
        prompt = f"""You are a senior UK hiring manager with 15 years of experience. You've just reviewed the original CV (ATS score {ats_score}%) and an optimised version. Write a **detailed feedback brief** (150-200 words) that:

1. Picks **two or three specific sections** (e.g., Personal Statement, Work Experience bullet points, Skills list).
2. For each section, say: "In your original CV, [problem]. If it were like [exact quote or example from the optimised version], I would have been impressed."
3. Add one **specific, actionable piece of advice** that would make the candidate shortlisted.
4. End with a shortlist verdict: "Based on the optimised version, I would [definitely / probably / maybe] invite you for an interview."

Use a professional but direct tone – not sarcastic, but honest. Write in paragraphs.

Job description snippet:
{job_text[:1200]}

Original CV (first 1500 chars):
{resume_text[:1500]}

Optimised CV (first 1500 chars):
{optimised_text[:1500]}

Output ONLY the feedback brief – no extra labels."""
    else:
        prompt = f"""You are a senior UK hiring manager. Review this CV (ATS score {ats_score}%) and write a **detailed feedback brief** (120-180 words) that:

1. Highlights one section that needs improvement.
2. Gives a concrete example of what would impress you: "If your [section] were like [rewritten example], I would have been impressed."
3. Provides two specific, actionable recommendations.
4. Ends with an honest shortlist verdict.

Job description snippet:
{job_text[:1200]}

CV (first 1500 chars):
{resume_text[:1500]}

Output ONLY the feedback brief."""

    response, err = safe_groq_chat_create(
        client,
        model=_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
    )
    if err:
        return None if is_groq_rate_limit_error(err) else err
    return (response.choices[0].message.content or "").strip()


def _cv_json_for_refinement(data: dict) -> dict:
    """Strip internal keys before sending CV JSON to the model."""
    if not isinstance(data, dict):
        return {}
    return {k: v for k, v in data.items() if not str(k).startswith("_")}


def refine_cv_with_feedback(current_cv_json: dict, job_text: str, feedback_text: str) -> dict:
    """Rewrite CV JSON to address every point in the hiring manager's feedback without inventing facts."""
    api_key = _require_groq_api_key()
    client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")

    system_prompt = """You are a CV expert. You will receive a CV (JSON) and honest feedback from a hiring manager. Rewrite the CV to address EVERY point in the feedback. Keep all factual information true. You may rephrase bullet points, add missing keywords that the candidate genuinely possesses, reorder sections, or adjust the personal statement. Do NOT invent false jobs, dates, or qualifications. Return a new JSON object with exactly these keys: name, contact_phone, contact_email, contact_location, linkedin, personal_statement, key_skills (array of strings), work_experience, education, skills, certifications, interests, references. Output ONLY valid JSON — no markdown fences or commentary."""

    payload = _cv_json_for_refinement(current_cv_json)
    current_cv_str = json.dumps(payload, indent=2, ensure_ascii=False)
    if len(current_cv_str) > 28000:
        current_cv_str = current_cv_str[:28000] + "\n… (truncated)"

    user_prompt = f"""Job description (for context; honour the feedback first):
{(job_text or '')[:10000]}

Hiring manager feedback (address all points):
{(feedback_text or '')[:12000]}

Current CV (JSON):
{current_cv_str}

Output improved JSON only."""

    create_kwargs: dict = dict(
        model=_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.4,
    )
    response, err = safe_groq_chat_create(
        client,
        **create_kwargs,
        response_format={"type": "json_object"},
    )
    if err:
        if is_groq_rate_limit_error(err):
            return {"rate_limited": True}
        response, err2 = safe_groq_chat_create(client, **create_kwargs)
        if err2:
            if is_groq_rate_limit_error(err2):
                return {"rate_limited": True}
            return {"error": err2}

    raw = response.choices[0].message.content
    parsed = _parse_json_from_ai(raw)
    if parsed.get("error"):
        return {
            "error": f"Invalid JSON from refinement: {parsed.get('error')}",
            "raw": parsed.get("raw", str(raw)[:2000] if raw else ""),
        }
    return parsed


# Edit this list to suppress low-signal tokens from AI-assembled CVs.
# Do NOT add legitimate modern skills (AI tools, LLM platforms, etc.)
_SKILL_REVIEW_NOISE = frozenset(
    {
        "investigate",
        "requests",
        "workflows",
        "governance",
        "ai tools",
        "gemini",
        "supply chain management",
    }
)


def _strip_noise_skills_from_review_cv(data: dict) -> None:
    """Drop low-signal tokens the model sometimes copies from review layout (in-place)."""
    for key in ("skills", "key_skills"):
        xs = data.get(key)
        if not isinstance(xs, list):
            continue
        cleaned: list[str] = []
        for s in xs:
            if not isinstance(s, str):
                continue
            t = s.strip()
            if not t:
                continue
            if t.lower() in _SKILL_REVIEW_NOISE:
                continue
            cleaned.append(t)
        data[key] = cleaned


def optimize_cv_from_review(
    original_cv_text: str,
    review_text: str,
    job_text: str = "",
) -> dict:
    """
    Assemble CV JSON by copying improved wording from the expert review verbatim where given.
    Falls back to original CV text only where the review does not supply an exact rewrite.
    """
    api_key = _require_groq_api_key()
    client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")

    system_prompt = """You are a text assembler. The user will provide an expert review that contains EXACT rewritten versions of CV sections (profile, bullet points, skills list, etc.). Your job is to extract those exact strings and arrange them into a JSON CV structure. Do not rewrite, rephrase, or improve further. Copy verbatim.

Rules:
- If the review says "Rewritten:" or shows a bullet point with a dash or number, use that exact text.
- For the profile, use the exact paragraph under "Stronger Professional Summary" or similar heading when present.
- For work experience bullets, use the improved bullets exactly as written in the review (e.g. lines starting with action verbs the review provides).
- For skills, use the categorised list exactly as shown (Technical, Analytical, Business) as a flat `skills` array of strings in that order — do not add filler single words.
- Include `key_skills` (array of strings): copy the first 8–12 entries from that same flat skills list in order (still verbatim from the review); if the review only gives a flat list, mirror the first entries into `key_skills`.
- Remove Interests and References as sections: set both to empty strings.
- Use the original CV text only for name, contact fields, LinkedIn, and any section the review does not rewrite; keep employers, titles, and dates aligned with the original CV unless the review explicitly fixes a typo.
- Required JSON keys (all present): name, contact_phone, contact_email, contact_location, linkedin, personal_statement, key_skills, work_experience (list of objects with job_title, employer, dates, bullets), education (list of objects with degree, institution, dates, grade), skills, certifications, interests, references.

If the review does not provide an exact rewrite for a section, keep the original content from the original CV for that section while still applying structural rules (e.g. empty interests/references when dropping those sections).

Output ONLY valid JSON — no markdown fences or commentary."""

    user_prompt = f"""Original CV (for name/contacts and unchanged sections):
{(original_cv_text or '')[:3000]}

Expert review (copy the exact improved versions from here):
{(review_text or '')[:10000]}

Job description (for context, not for rewriting):
{(job_text or '')[:1500]}

Assemble the JSON as instructed. Output only valid JSON."""

    create_kwargs: dict = dict(
        model=_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
    )
    response, err = safe_groq_chat_create(
        client,
        **create_kwargs,
        response_format={"type": "json_object"},
    )
    if err:
        if is_groq_rate_limit_error(err):
            return {"rate_limited": True}
        response, err2 = safe_groq_chat_create(client, **create_kwargs)
        if err2:
            if is_groq_rate_limit_error(err2):
                return {"rate_limited": True}
            return {"error": err2}

    raw = response.choices[0].message.content
    parsed = _parse_json_from_ai(raw)
    if parsed.get("error"):
        return {
            "error": f"Invalid JSON from review-based assembly: {parsed.get('error')}",
            "raw": parsed.get("raw", str(raw)[:2000] if raw else ""),
        }

    _strip_noise_skills_from_review_cv(parsed)
    sk = parsed.get("skills")
    if (not isinstance(parsed.get("key_skills"), list) or not parsed.get("key_skills")) and isinstance(sk, list):
        parsed["key_skills"] = [s for s in sk if isinstance(s, str) and s.strip()][:12]

    return parsed


def generate_interview_questions(resume_text: str, job_text: str, missing_keywords_str: str) -> str | None:
    """Returns None on Groq rate limit (UI silent)."""
    api_key = _require_groq_api_key()
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )

    prompt = f"""You are a hiring manager for this job description. The candidate's resume is missing these key skills: {missing_keywords_str}. Generate 5-7 interview questions that:
- Probe their existing experience (from resume)
- Assess their ability to learn the missing skills
- Mix behavioural and technical questions

Job description:
{job_text[:30000]}

Resume summary:
{resume_text[:15000]}
"""
    response, err = safe_groq_chat_create(
        client,
        model=_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    if err:
        return None if is_groq_rate_limit_error(err) else err
    return response.choices[0].message.content or ""


def generate_improvement_suggestions(resume_text: str, job_text: str, missing_keywords_str: str) -> str | None:
    """Returns None on Groq rate limit (UI silent)."""
    api_key = _require_groq_api_key()
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )

    prompt = f"""You are an expert resume coach. The candidate applied for the job above but their resume lacks these skills: {missing_keywords_str}. Provide 3-5 specific, actionable improvements to add or rewrite parts of the resume to include these missing skills. Be concrete (e.g., "Under 'Experience', add bullet: Prepared monthly financial forecasts using Excel and Power BI").

Job description:
{job_text[:30000]}

Current resume:
{resume_text[:15000]}
"""
    response, err = safe_groq_chat_create(
        client,
        model=_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    if err:
        return None if is_groq_rate_limit_error(err) else err
    return response.choices[0].message.content or ""


def _parse_json_from_ai(raw: str | None) -> dict:
    """Parse JSON from model output; strip markdown fences if present."""
    if not raw or not str(raw).strip():
        return {"error": "Empty AI response"}
    s = str(raw).strip()
    if s.startswith("```"):
        sl = s.splitlines()
        if sl and sl[0].strip().startswith("```"):
            sl = sl[1:]
        if sl and sl[-1].strip().startswith("```"):
            sl = sl[:-1]
        s = "\n".join(sl).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", s)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
        return {"error": "AI returned invalid JSON", "raw": s[:2000]}


def _cv_plaintext_for_ats_scoring(data: dict) -> str:
    """Flatten structured CV JSON for whitelist keyword overlap scoring."""
    parts: list[str] = []
    if data.get("personal_statement"):
        parts.append(str(data["personal_statement"]))
    for exp in data.get("work_experience") or []:
        if not isinstance(exp, dict):
            continue
        parts.append(str(exp.get("job_title", "") or ""))
        parts.append(str(exp.get("employer", "") or ""))
        parts.extend(str(b) for b in (exp.get("bullets") or []) if b)
    parts.extend(str(s) for s in (data.get("key_skills") or []) if s)
    parts.extend(str(s) for s in (data.get("skills") or []) if s)
    for edu in data.get("education") or []:
        if not isinstance(edu, dict):
            continue
        for k in ("degree", "institution", "dates", "grade"):
            v = edu.get(k)
            if v:
                parts.append(str(v))
    parts.extend(str(c) for c in (data.get("certifications") or []) if c)
    return "\n".join(parts)


def render_cv_to_text(cv_json: dict) -> str:
    """Convert optimised CV JSON to a plain text summary for the hiring manager prompt."""
    text = f"Personal Statement: {cv_json.get('personal_statement', '')}\n\n"
    text += "Work Experience:\n"
    for exp in cv_json.get("work_experience") or []:
        if not isinstance(exp, dict):
            continue
        text += f"- {exp.get('job_title', '')} at {exp.get('employer', '')} ({exp.get('dates', '')})\n"
        for bullet in exp.get("bullets") or []:
            text += f"  • {bullet}\n"
    text += f"\nKey Skills: {', '.join(cv_json.get('key_skills') or [])}\n"
    text += f"Skills: {', '.join(cv_json.get('skills') or [])}\n"
    return text




def optimize_cv_generic(
    resume_text: str,
    job_text: str,
    missing_keywords: list[str],
    max_score_target: int = 75,
) -> dict:
    """
    Recruiter-grade generic optimiser (strict rewrite pipeline in ``utils.cv_optimizer``).
    Returns JSON for ``MasterTemplate.docx`` plus ``_ats_score_achieved`` when successful.
    """
    _ = max_score_target
    from utils.cv_optimizer import is_groq_rate_limit_error, run_generic_cv_optimisation_pipeline

    if len(resume_text) > 60000:
        st.warning(_TRUNCATION_WARNING)
    if len(job_text) > 60000:
        st.warning(_TRUNCATION_WARNING)

    rt = (resume_text or "").strip()
    mk = [str(k) for k in (missing_keywords or []) if k]
    print(
        "[applylytics] optimize_cv_generic() → pipeline | resume_chars=",
        len(rt),
        "| missing_keywords=",
        len(mk),
        flush=True,
    )

    out = run_generic_cv_optimisation_pipeline(
        rt,
        job_text=job_text or "",
        missing_keywords=missing_keywords or [],
    )
    if is_groq_rate_limit_error(out):
        return {"rate_limited": True}
    if out.get("error"):
        print("[applylytics] optimize_cv_generic() pipeline error:", out["error"], flush=True)
        return out

    jt = (job_text or "").strip()
    try:
        cv_text = _cv_plaintext_for_ats_scoring(out)
        ats_result = calculate_ats_score(cv_text, jt)
        out["_ats_score_achieved"] = int(ats_result.get("score", 0))
    except Exception:
        out["_ats_score_achieved"] = 0

    ps = str(out.get("personal_statement", "") or "")
    n_jobs = len(out.get("work_experience") or [])
    n_bullets = sum(
        len(exp.get("bullets") or [])
        for exp in (out.get("work_experience") or [])
        if isinstance(exp, dict)
    )
    print(
        "[applylytics] optimize_cv_generic() OK | jobs=",
        n_jobs,
        "| bullets=",
        n_bullets,
        "| ats_estimate=",
        out.get("_ats_score_achieved"),
        flush=True,
    )
    print("[applylytics] personal_statement[:200]=", repr(ps[:200]), flush=True)
    return out


def render_chips(keywords: list[str], chip_type: str) -> str:
    """
    Build HTML for keyword chips (matched = green/dark text, missing = red/white).
    chip_type: 'matched' | 'missing'
    """
    limit = 30
    slice_kw = keywords[:limit]
    extra = max(0, len(keywords) - limit)
    cls = "chip chip-matched" if chip_type == "matched" else "chip chip-missing"
    parts = [f'<span class="{cls}">{html.escape(k)}</span>' for k in slice_kw]
    if extra:
        parts.append(f'<span class="chip chip-more">+{extra} more</span>')
    inner = "".join(parts) if parts else '<span class="chip chip-more">—</span>'
    return f'<div class="chip-row">{inner}</div>'


def display_ats_score(
    score: int,
    matched: list,
    missing: list,
    job_keyword_count: int,
    insight: str | None = None,
) -> None:
    """ATS hero (SVG radial + gradient number), st.metric, and chip cards."""
    if score >= 70:
        accent = "#10B981"
    elif score >= 40:
        accent = "#F59E0B"
    else:
        accent = "#F43F5E"

    r = 42
    c = 264.0  # ~2 * pi * r
    dash = c * (score / 100.0)

    st.markdown(
        f"""
        <div class="score-hero-wrap">
          <div class="score-svg-wrap">
            <svg viewBox="0 0 100 100" aria-hidden="true">
              <circle cx="50" cy="50" r="{r}" fill="none" stroke="#2d2d3d" stroke-width="8" />
              <circle cx="50" cy="50" r="{r}" fill="none" stroke="{accent}" stroke-width="8"
                stroke-dasharray="{dash:.2f} {c:.2f}" stroke-linecap="round"
                transform="rotate(-90 50 50)" />
            </svg>
            <div class="score-center">
              <div class="score-num">{score}<span class="score-pct">%</span></div>
            </div>
          </div>
          <div class="score-sub">ATS Match Score</div>
          <div class="score-kw">{job_keyword_count} whitelisted skill phrases in job text</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if insight:
        st.markdown(
            f'<p class="score-insight">{html.escape(insight)}</p>',
            unsafe_allow_html=True,
        )

    st.metric(label="Match score (summary)", value=f"{score}%", delta=None)

    st.markdown(
        f"""
        <div class="kw-card">
          <div class="kw-card-title">Matched keywords</div>
          {render_chips(matched, "matched")}
        </div>
        <div class="kw-card">
          <div class="kw-card-title">Missing keywords</div>
          {render_chips(missing, "missing")}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(portfolio_url: str) -> None:
    """Glass-styled sidebar: info, stats (pandas), GitHub link with Font Awesome."""
    with st.sidebar:
        st.markdown(
            """
            <div class="glass-card" style="margin-top:0.5rem;">
              <p style="margin:0;font-size:20px;font-weight:600;color:#E0E0E0;">Applylytics</p>
              <p style="margin:0.5rem 0 0;font-size:14px;color:#8E8E9A;line-height:1.45;">
                Resume ↔ job fit, whitelist ATS overlap, and Groq Llama coaching.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<p class="section-label" style="margin-top:1rem;">Quick stats</p>', unsafe_allow_html=True)
        wc = st.session_state.get(SessionKey.UI_WORD_COUNT)
        mp = st.session_state.get(SessionKey.UI_MATCH_PCT)
        jk = st.session_state.get(SessionKey.UI_JOB_KW_COUNT)
        stats_df = pd.DataFrame(
            {
                "Metric": ["Resume words", "Skill match %", "Job skill phrases"],
                "Value": [
                    wc if wc is not None else "—",
                    f"{mp}%" if mp is not None else "—",
                    jk if jk is not None else "—",
                ],
            }
        )
        st.dataframe(stats_df, hide_index=True, use_container_width=True)

        st.markdown('<p class="section-label" style="margin-top:1rem;">Links</p>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <a href="{html.escape(portfolio_url)}" target="_blank" rel="noopener noreferrer"
               style="display:inline-flex;align-items:center;gap:0.5rem;padding:0.5rem 1rem;
               border-radius:40px;background:rgba(92,110,248,0.2);color:#5C6EF8;
               text-decoration:none;font-weight:600;font-size:14px;border:1px solid rgba(92,110,248,0.35);">
              <i class="fab fa-github" aria-hidden="true"></i> GitHub / portfolio
            </a>
            """,
            unsafe_allow_html=True,
        )
        st.caption("Set `PORTFOLIO_URL` in `.env` to customize the link target.")


SKELETON_HTML = """
<div class="skeleton-block" aria-busy="true"></div>
<div class="skeleton-block" style="height:80px;margin-top:0.75rem;"></div>
"""



def _clear_ui_stats() -> None:
    st.session_state[SessionKey.UI_WORD_COUNT] = None
    st.session_state[SessionKey.UI_MATCH_PCT] = None
    st.session_state[SessionKey.UI_JOB_KW_COUNT] = None


def _phase_key(resume_text: str, job_description: str) -> str:
    return (
        f"{hashlib.sha256(resume_text.encode()).hexdigest()[:16]}:"
        f"{hashlib.sha256(job_description.encode()).hexdigest()[:24]}"
    )


def _render_upload_panel() -> tuple[str | None, str]:
    """Left column: file uploader + job description textarea.
    Returns (tmp_path_or_none, job_description)."""
    st.markdown("##### :material/upload_file: Resume")
    with st.container(border=True):
        uploaded = st.file_uploader(
            "Resume (PDF)",
            type=["pdf"],
            help="Upload the CV you want scored against the role.",
            label_visibility="visible",
        )
    st.markdown("##### :material/work: Job description")
    with st.container(border=True):
        job_description = st.text_area(
            "Job description",
            height=240,
            placeholder="Paste the full job description (requirements, responsibilities, tools).",
            help="Used for ATS keyword overlap and all AI prompts.",
            label_visibility="visible",
        )
    tmp_path: str | None = None
    if uploaded is not None:
        suffix = Path(uploaded.name).suffix or ".pdf"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getbuffer())
            tmp_path = tmp.name
    return tmp_path, job_description or ""


def _render_ats_panel(resume_text: str, job_description: str) -> dict:
    """Runs ATS scoring + recruiter penalties + displays results.
    Returns the full ats_result dict."""
    phase_key = _phase_key(resume_text, job_description)
    if st.session_state.get(SessionKey.PHASE3_KEY) != phase_key:
        st.session_state[SessionKey.PHASE3_KEY] = phase_key
        st.session_state.pop(SessionKey.INTERVIEW_QUESTIONS, None)
        st.session_state.pop(SessionKey.IMPROVEMENT_PLAN, None)
        st.session_state.pop(SessionKey.MANAGER_COMMENT, None)
        st.session_state.pop(SessionKey.HM_FEEDBACK_CACHE_KEY, None)
        st.session_state.pop(SessionKey.CV_FROM_REVIEW_PLAN, None)
        st.session_state.pop(SessionKey.OPTIMIZED_DATA, None)
        st.session_state.pop(SessionKey.DOCX_BYTES, None)

    word_count = len(resume_text.split())
    st.session_state[SessionKey.UI_WORD_COUNT] = word_count

    with st.expander(":material/description: Extracted resume text (preview)", expanded=False):
        st.text_area(
            "Resume text",
            value=resume_text[:8000] + ("…" if len(resume_text) > 8000 else ""),
            height=220,
            label_visibility="collapsed",
        )

    with st.spinner("Scoring whitelisted skill overlap against this role…"):
        ats_result = calculate_ats_score(resume_text, job_description)

    st.session_state[SessionKey.UI_MATCH_PCT] = ats_result["score"]
    st.session_state[SessionKey.UI_JOB_KW_COUNT] = ats_result["job_keyword_count"]

    display_ats_score(
        ats_result["score"],
        ats_result["matched_keywords"],
        ats_result["missing_keywords"],
        ats_result["job_keyword_count"],
        insight=ats_result.get("insight"),
    )
    st.caption(
        "How to read this score: it is the share of job-linked skill phrases (from our detector list) "
        "that also appear in the resume text **above after PDF extraction** — not a universal ATS grade. "
        "Saving a DOCX as PDF, changing wording, or column layout can change which phrases are extracted, "
        "so the percentage can go up or down even for the same career story. Always check the preview "
        "expander before trusting the number."
    )

    rp = calculate_recruiter_penalties(resume_text)
    st.markdown("##### :material/rule_folder: Recruiter presentation check")
    st.caption(
        "Heuristic only — **not** mixed into the ATS keyword score. "
        "Starts at 100 and subtracts points for common habits; use as a second opinion."
    )
    r1, r2 = st.columns([1, 2])
    with r1:
        st.metric("Presentation score", f"{rp['adjusted_score']}")
        st.metric("Penalty points", f"{rp['total_deduction']}")
    with r2:
        if rp["penalties"]:
            st.markdown("\n".join(f"- {html.escape(p)}" for p in rp["penalties"]))
        else:
            st.markdown("_No items from this checklist triggered._")
    return ats_result


def _render_ai_coach_panel(
    resume_text: str,
    job_description: str,
    ats_result: dict,
) -> None:
    """Hiring manager comment, AI feedback button,
    interview questions, improvement plan."""
    phase_key = _phase_key(resume_text, job_description)
    od_hm = st.session_state.get(SessionKey.OPTIMIZED_DATA)
    opt_plain = ""
    if isinstance(od_hm, dict) and not od_hm.get("error"):
        opt_plain = render_cv_to_text(od_hm).strip()
    fp = hashlib.sha256(opt_plain.encode()).hexdigest()[:20] if opt_plain else "none"
    hm_fb_key = f"{phase_key}:{fp}"
    if st.session_state.get(SessionKey.HM_FEEDBACK_CACHE_KEY) != hm_fb_key:
        with st.spinner("Getting honest feedback from a hiring manager…"):
            comment = get_hiring_manager_comment(
                resume_text,
                job_description,
                ats_result["score"],
                optimised_text=opt_plain if opt_plain else None,
            )
        if comment is not None:
            st.session_state[SessionKey.MANAGER_COMMENT] = comment
            st.session_state[SessionKey.HM_FEEDBACK_CACHE_KEY] = hm_fb_key
        else:
            st.warning(_RATE_LIMIT_WARNING)
            st.stop()
    else:
        comment = st.session_state.get(SessionKey.MANAGER_COMMENT)

    if comment:
        st.markdown("### 🗣️ Hiring Manager's Honest Feedback")
        st.info(comment)

    od_refine = st.session_state.get(SessionKey.OPTIMIZED_DATA)
    if isinstance(od_refine, dict) and not od_refine.get("error"):
        if st.button("📈 Improve CV based on this feedback", use_container_width=True):
            with st.spinner("Revising CV to address feedback…"):
                improved = refine_cv_with_feedback(
                    od_refine,
                    job_description,
                    st.session_state.get(SessionKey.MANAGER_COMMENT) or "",
                )
            if improved.get("rate_limited"):
                st.warning(_RATE_LIMIT_WARNING)
                st.stop()
            elif improved.get("error"):
                st.error(improved["error"])
            else:
                try:
                    pt = _cv_plaintext_for_ats_scoring(improved)
                    improved["_ats_score_achieved"] = int(
                        calculate_ats_score(pt, job_description)["score"]
                    )
                except Exception:
                    pass
                st.session_state[SessionKey.OPTIMIZED_DATA] = improved
                st.session_state.pop(SessionKey.DOCX_BYTES, None)
                st.session_state.pop(SessionKey.HM_FEEDBACK_CACHE_KEY, None)
                st.session_state.pop(SessionKey.CV_FROM_REVIEW_PLAN, None)
                st.success(
                    "CV updated based on feedback. Scroll down for the preview and DOCX; "
                    "manager feedback above will refresh on the next load."
                )
                st.rerun()

    st.markdown("##### :material/smart_toy: AI coach")
    if st.button(
        "🤖 Get General AI Feedback",
        type="primary",
        use_container_width=True,
    ):
        ph = st.empty()
        ph.markdown(SKELETON_HTML, unsafe_allow_html=True)
        with st.spinner("Analyzing with Groq Llama 3.3…"):
            feedback = analyze_resume_with_job(resume_text, job_description)
        ph.empty()
        if feedback is not None:
            st.markdown("**AI feedback**")
            st.markdown(feedback)
        else:
            st.warning(_RATE_LIMIT_WARNING)
            st.stop()

    missing_kw = ats_result["missing_keywords"]
    missing_keywords_str = ", ".join(missing_kw[:300]) if missing_kw else "(none identified)"
    st.markdown("##### :material/quiz: Interview & resume polish")
    iq, ip = st.columns(2)
    with iq:
        if st.button("🎯 Generate Interview Questions", use_container_width=True):
            ph = st.empty()
            ph.markdown(SKELETON_HTML, unsafe_allow_html=True)
            with st.spinner("Analyzing with Groq Llama 3.3…"):
                iq_md = generate_interview_questions(
                    resume_text, job_description, missing_keywords_str
                )
            ph.empty()
            if iq_md is not None:
                st.session_state[SessionKey.INTERVIEW_QUESTIONS] = iq_md
            else:
                st.session_state.pop(SessionKey.INTERVIEW_QUESTIONS, None)
                st.warning(_RATE_LIMIT_WARNING)
                st.stop()
    with ip:
        if st.button("📝 Personalised Improvement Plan", use_container_width=True):
            ph = st.empty()
            ph.markdown(SKELETON_HTML, unsafe_allow_html=True)
            with st.spinner("Analyzing with Groq Llama 3.3…"):
                imp_md = generate_improvement_suggestions(
                    resume_text, job_description, missing_keywords_str
                )
            ph.empty()
            if imp_md is not None:
                st.session_state[SessionKey.IMPROVEMENT_PLAN] = imp_md
            else:
                st.session_state.pop(SessionKey.IMPROVEMENT_PLAN, None)
                st.warning(_RATE_LIMIT_WARNING)
                st.stop()

    if st.session_state.get(SessionKey.INTERVIEW_QUESTIONS):
        with st.expander(":material/forum: Interview questions", expanded=True):
            st.markdown(st.session_state[SessionKey.INTERVIEW_QUESTIONS])

    imp_plan = st.session_state.get(SessionKey.IMPROVEMENT_PLAN)
    if imp_plan:
        with st.expander(":material/rule: Personalised improvement plan", expanded=True):
            st.markdown(imp_plan)
        if st.button(
            "📄 Generate CV following this review exactly",
            use_container_width=True,
            help="Assembles structured CV JSON by copying improved wording from this plan verbatim where it appears.",
        ):
            ph = st.empty()
            ph.markdown(SKELETON_HTML, unsafe_allow_html=True)
            with st.spinner("Applying every change from the review…"):
                new_cv_data = optimize_cv_from_review(
                    resume_text,
                    imp_plan,
                    job_description,
                )
            ph.empty()
            if new_cv_data.get("rate_limited"):
                st.warning(_RATE_LIMIT_WARNING)
                st.stop()
            elif new_cv_data.get("error"):
                st.error(new_cv_data["error"])
            else:
                try:
                    pt = _cv_plaintext_for_ats_scoring(new_cv_data)
                    new_cv_data["_ats_score_achieved"] = int(
                        calculate_ats_score(pt, job_description)["score"]
                    )
                except Exception:
                    pass
                st.session_state[SessionKey.OPTIMIZED_DATA] = new_cv_data
                st.session_state.pop(SessionKey.HM_FEEDBACK_CACHE_KEY, None)
                st.session_state[SessionKey.CV_FROM_REVIEW_PLAN] = True
                if not CV_TEMPLATE_DOCX.is_file():
                    st.session_state.pop(SessionKey.DOCX_BYTES, None)
                    st.error(f"Template missing: {CV_TEMPLATE_DOCX}")
                else:
                    try:
                        context = cv_context_from_structured(new_cv_data)
                        st.session_state[SessionKey.DOCX_BYTES] = _render_docx_to_bytes(
                            CV_TEMPLATE_DOCX, context
                        )
                    except Exception as exc:
                        st.session_state.pop(SessionKey.DOCX_BYTES, None)
                        st.error(f"DOCX render failed: {exc}")
                st.success(
                    "CV assembled from the review's exact wording where provided. "
                    "Preview and DOCX download are in the section below."
                )


def _render_optimiser_panel(
    resume_text: str,
    job_description: str,
    ats_result: dict,
) -> None:
    """Optimise Resume button, Reset button,
    DOCX preview + download."""
    st.caption(
        "Optimisation uses only the extracted resume text from your PDF (not a stock sample CV). "
        "Open the resume preview expander first — if it is empty or garbled, re-export the PDF as text before optimising. "
        f"DOCX export uses {CV_TEMPLATE_DOCX.name} (generic optimiser + master template)."
    )
    opt_col1, opt_col2 = st.columns(2)
    with opt_col1:
        if st.button(
            "Optimize Resume to 75%+ ATS Score",
            use_container_width=True,
            icon=":material/auto_awesome:",
        ):
            ph = st.empty()
            ph.markdown(SKELETON_HTML, unsafe_allow_html=True)
            with st.spinner("Running generic CV optimiser (Groq)…"):
                result = optimize_cv_generic(
                    resume_text,
                    job_description,
                    missing_kw,
                    max_score_target=75,
                )
            ph.empty()
            if result.get("rate_limited"):
                st.warning(_RATE_LIMIT_WARNING)
                st.stop()
            elif result.get("error"):
                st.error(result["error"])
            else:
                st.session_state[SessionKey.OPTIMIZED_DATA] = result
                st.session_state.pop(SessionKey.CV_FROM_REVIEW_PLAN, None)
                st.session_state.pop(SessionKey.HM_FEEDBACK_CACHE_KEY, None)
                if not CV_TEMPLATE_DOCX.is_file():
                    st.session_state.pop(SessionKey.DOCX_BYTES, None)
                    st.error(f"Template missing: {CV_TEMPLATE_DOCX}")
                else:
                    try:
                        context = cv_context_from_structured(result)
                        st.session_state[SessionKey.DOCX_BYTES] = _render_docx_to_bytes(
                            CV_TEMPLATE_DOCX, context
                        )
                    except Exception as exc:
                        st.session_state.pop(SessionKey.DOCX_BYTES, None)
                        st.error(f"DOCX render failed: {exc}")
                done_msg = (
                    "Optimisation complete — generic optimiser + master template. "
                    "Download DOCX below."
                )
                ach = result.get("_ats_score_achieved")
                if isinstance(ach, int):
                    done_msg += f" Estimated whitelist overlap after this pass: {ach}%."
                st.success(done_msg)
    with opt_col2:
        if st.button(
            "Reset optimised CV",
            use_container_width=True,
            help="Clears cached structured CV and DOCX so the next optimise starts fresh.",
        ):
            st.session_state.pop(SessionKey.OPTIMIZED_DATA, None)
            st.session_state.pop(SessionKey.DOCX_BYTES, None)
            st.session_state.pop(SessionKey.CV_FROM_REVIEW_PLAN, None)
            st.session_state.pop(SessionKey.HM_FEEDBACK_CACHE_KEY, None)
            st.session_state.pop(SessionKey.MANAGER_COMMENT, None)
            st.rerun()

    od = st.session_state.get(SessionKey.OPTIMIZED_DATA)
    if isinstance(od, dict) and not od.get("error"):
        data = od
        with st.expander("Preview optimised data", expanded=False):
            st.json(data)

        col_docx, col_info = st.columns(2)
        with col_docx:
            if st.button("📄 Prepare DOCX download", use_container_width=True):
                if not CV_TEMPLATE_DOCX.is_file():
                    st.error(f"Template missing: {CV_TEMPLATE_DOCX}")
                else:
                    try:
                        context = cv_context_from_structured(data)
                        st.session_state[SessionKey.DOCX_BYTES] = _render_docx_to_bytes(
                            CV_TEMPLATE_DOCX, context
                        )
                    except Exception as exc:
                        st.session_state.pop(SessionKey.DOCX_BYTES, None)
                        st.error(f"DOCX render failed: {exc}")
            docx_bytes = st.session_state.get(SessionKey.DOCX_BYTES)
            if docx_bytes:
                safe_name = re.sub(
                    r"[^\w\-]+",
                    "_",
                    (data.get("name") or "CV").strip(),
                ).strip("_")[:80] or "CV"
                suffix = "_CV_Improved.docx" if st.session_state.get(SessionKey.CV_FROM_REVIEW_PLAN) else "_CV.docx"
                st.download_button(
                    label="Download DOCX",
                    data=docx_bytes,
                    file_name=f"{safe_name}{suffix}",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    use_container_width=True,
                    key="dl_docx_optimised",
                )
        with col_info:
            st.info(
                "PDF export: open the DOCX in Word and use Save as PDF, or use an online converter. "
                "Direct PDF export from this screen may be added later."
            )


_HERO_HTML = """
<div class="ds-hero">
  <h1>Applylytics</h1>
  <p>Production-grade resume ↔ job analysis: whitelist ATS scoring, Groq Llama 3.3 insights, interview prep, structured CV optimisation, and editable DOCX export.</p>
</div>
"""

_GET_STARTED_MD = """
**:material/rocket_launch: Get started** — upload a PDF and paste a job description to unlock:

- Whitelist ATS overlap score
- Matched / missing skill chips
- Groq-powered feedback, interview questions, improvement plan, and optimised DOCX export
"""


def _render_results_container(tmp_path: str | None, job_description: str) -> None:
    st.markdown("##### :material/insights: Results")
    with st.container(border=True):
        if tmp_path is not None and job_description.strip():
            try:
                with st.spinner("Extracting text from your PDF…"):
                    resume_text = extract_text_from_pdf(tmp_path)
            finally:
                Path(tmp_path).unlink(missing_ok=True)
            ats_result = _render_ats_panel(resume_text, job_description)
            _render_ai_coach_panel(resume_text, job_description, ats_result)
            _render_optimiser_panel(resume_text, job_description, ats_result)
        elif tmp_path is not None and not job_description.strip():
            Path(tmp_path).unlink(missing_ok=True)
            _clear_ui_stats()
            st.info("Add a job description on the left to unlock ATS scoring and AI tools.")
        elif job_description.strip() and tmp_path is None:
            _clear_ui_stats()
            st.info("Upload your resume PDF on the left to continue.")
        else:
            _clear_ui_stats()
            st.markdown(_GET_STARTED_MD)


def main() -> None:
    st.set_page_config(
        page_title="Applylytics",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    portfolio_url = os.getenv("PORTFOLIO_URL", "https://github.com").strip() or "https://github.com"
    inject_design_system()
    render_sidebar(portfolio_url)
    st.markdown(_HERO_HTML, unsafe_allow_html=True)
    st.markdown("##### :material/analytics: Overview")
    col_left, col_right = st.columns([1, 1.5], gap="large")
    with col_left:
        tmp_path, job_description = _render_upload_panel()
    with col_right:
        _render_results_container(tmp_path, job_description)


if __name__ == "__main__":
    main()
