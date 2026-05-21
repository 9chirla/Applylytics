"""Reusable Streamlit UI components and HTML fragments."""

from __future__ import annotations

import base64
import html
import math
from functools import lru_cache
from pathlib import Path
from textwrap import dedent
from typing import Any

import streamlit as st

SKELETON_HTML = dedent("""
<div class="skeleton-block" aria-busy="true"></div>
<div class="skeleton-block" style="height:80px;margin-top:0.75rem;"></div>
""").strip()

EMPTY_STATE_HTML = dedent("""
<div class="al-empty-state">
  <div class="al-empty-state-icon" aria-hidden="true">
    <svg width="28" height="28" viewBox="0 0 28 28" fill="none" xmlns="http://www.w3.org/2000/svg">
      <rect x="5" y="3" width="14" height="18" rx="1.5" stroke="currentColor" stroke-width="1.5"/>
      <path d="M19 3v5h5" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
      <path d="M9 11h6M9 14h6M9 17h4" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
    </svg>
  </div>
  <p class="al-empty-state-title">Upload a resume and paste a job description</p>
  <p class="al-empty-state-sub">ATS scoring, keyword chips, AI coaching, and optimised DOCX export will appear here.</p>
</div>
""").strip()


def _dedent_html(markup: str) -> str:
    """Strip Python indentation so Markdown does not treat HTML as a code block."""
    return dedent(markup).strip()


def emit_html(markup: str) -> None:
    """Render HTML without markdown code-fence indentation (keeps global CSS)."""
    st.markdown(_dedent_html(markup), unsafe_allow_html=True)


def render_top_bar(portfolio_url: str, docs_url: str = "https://github.com") -> None:
    """Logo and nav links."""
    emit_html(f"""
    <div class="al-topbar">
      <div class="al-logo">apply<span>lytics</span></div>
      <nav class="al-nav">
        <a class="al-nav-link" href="{html.escape(docs_url)}" target="_blank" rel="noopener noreferrer">Docs</a>
        <a class="al-nav-link" href="{html.escape(portfolio_url)}" target="_blank" rel="noopener noreferrer">GitHub</a>
      </nav>
    </div>
    """)


def scroll_to_snap(marker_class: str) -> None:
    """Scroll Streamlit's main container to a snap anchor (e.g. al-snap-input)."""
    emit_html(f"""
    <script>
    (function() {{
      var doc = window.parent.document;
      var target = doc.querySelector(".{marker_class}");
      if (target) {{
        target.scrollIntoView({{ behavior: "smooth", block: "start" }});
      }}
    }})();
    </script>
    """)


def render_section_header(
    tag: str,
    title_dark: str,
    title_accent: str,
    subtitle: str = "",
) -> None:
    """Section title block matching hero typography (badge, h1, sub)."""
    sub_block = (
        f'<p class="al-hero-sub">{subtitle}</p>' if subtitle else ""
    )
    emit_html(f"""
    <div class="al-section-header">
      <div class="al-hero-tag"><span class="al-hero-tag-dot"></span> {html.escape(tag)}</div>
      <h1 class="al-hero-title">
        <span class="al-hero-title-dark">{html.escape(title_dark)}</span><br>
        <span class="al-hero-accent">{html.escape(title_accent)}</span>
      </h1>
      {sub_block}
    </div>
    """)


@lru_cache(maxsize=1)
def _hero_image_b64() -> str:
    """Inline hero image so it works without Streamlit static file serving."""
    p = Path(__file__).resolve().parent.parent.parent / "app" / "static" / "hero.png"
    if p.is_file():
        return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode("ascii")
    return ""


def _step_class(done: bool, active: bool) -> str:
    if done:
        return "done"
    if active:
        return "active"
    return ""


def render_hero_section(step_status: dict[str, bool]) -> str:
    """Return hero HTML including 4-step indicator."""
    upload = step_status.get("upload", False)
    jd = step_status.get("jd", False)
    analyse = step_status.get("analyse", False)
    optimise = step_status.get("optimise", False)

    s1 = _step_class(upload, not upload)
    s2 = _step_class(jd, upload and not jd)
    s3 = _step_class(analyse, upload and jd and not analyse)
    s4 = _step_class(optimise, analyse and not optimise)

    def num(done: bool, active: bool, n: int) -> str:
        return "✓" if done else str(n)

    hero_src = _hero_image_b64()
    return _dedent_html(f"""
    <div class="al-hero">
      <div class="al-hero-content">
        <h1 class="al-hero-title">
          <span class="al-hero-title-dark">Land the role</span><br>
          <span class="al-hero-title-dark">you </span><span class="al-hero-accent">deserve.</span>
        </h1>
        <p class="al-hero-sub">ATS keyword scoring, AI coaching, and one-click optimised CV export. Built for ambitious people who want results.</p>
        <div class="al-steps">
          <div class="al-step {s1}">
            <div class="al-step-num">{num(upload, not upload, 1)}</div>
            Upload CV
          </div>
          <div class="al-step-line"></div>
          <div class="al-step {s2}">
            <div class="al-step-num">{num(jd, upload and not jd, 2)}</div>
            Add JD
          </div>
          <div class="al-step-line"></div>
          <div class="al-step {s3}">
            <div class="al-step-num">{num(analyse, upload and jd and not analyse, 3)}</div>
            Analyse
          </div>
          <div class="al-step-line"></div>
          <div class="al-step {s4}">
            <div class="al-step-num">{num(optimise, analyse and not optimise, 4)}</div>
            Optimise
          </div>
        </div>
        <div class="al-scroll-cue" onclick="(function(){{var d=window.parent.document;var t=d.querySelector('.al-snap-input');if(t)t.scrollIntoView({{behavior:'smooth',block:'start'}});}})()">
          <span>Scroll to begin</span>
          <div class="al-scroll-arrow">↓</div>
        </div>
      </div>
      <div class="al-hero-image">
        <img src="{hero_src}" alt="" aria-hidden="true" />
      </div>
    </div>
    """)


def score_match_title(score: int) -> str:
    if score >= 80:
        return "Strong Match"
    if score >= 60:
        return "Moderate Match"
    if score >= 40:
        return "Fair Match"
    return "Needs Work"


def score_match_description(score: int, matched_count: int, job_keyword_count: int) -> str:
    total = max(job_keyword_count, 1)
    if score >= 80:
        tail = "You're in great shape for this role."
    elif score >= 60:
        tail = "You're close — a targeted rewrite could push this to 80%+."
    elif score >= 40:
        tail = "Several gaps remain — focus on missing phrases where truthful."
    else:
        tail = "Significant gaps — prioritise the missing keywords in your rewrite."
    return (
        f"{matched_count} of {total} skill phrases found. "
        f"{tail}"
    )


def render_score_ring(score: int) -> str:
    """SVG ring HTML for ATS score (0–100)."""
    r = 48
    circumference = 2 * math.pi * r
    dash = circumference * (max(0, min(100, score)) / 100.0)
    remainder = max(0.0, circumference - dash)
    if score >= 70:
        accent = "#3B82F6"
    elif score >= 40:
        accent = "#F59E0B"
    else:
        accent = "#EF4444"
    return _dedent_html(f"""
    <div class="al-ring-wrap">
      <svg width="120" height="120" viewBox="0 0 120 120" aria-hidden="true">
        <circle cx="60" cy="60" r="{r}" fill="none" stroke="#EEEEE9" stroke-width="9"/>
        <circle cx="60" cy="60" r="{r}" fill="none" stroke="{accent}" stroke-width="9"
          stroke-dasharray="{dash:.1f} {remainder:.1f}" stroke-linecap="round"
          transform="rotate(-90 60 60)"/>
      </svg>
      <div class="al-ring-center">
        <span class="al-ring-value">{score}<span class="al-ring-pct">%</span></span>
      </div>
    </div>
    """)


def render_score_panel(ats_result: dict[str, Any]) -> None:
    """Horizontal score panel: ring + metadata + bar."""
    score = int(ats_result.get("score", 0))
    matched = ats_result.get("matched_keywords") or []
    job_kw = int(ats_result.get("job_keyword_count", 0) or len(matched))
    title = score_match_title(score)
    desc = score_match_description(score, len(matched), job_kw)
    ring = render_score_ring(score)
    emit_html(f"""
    <div class="al-score-panel">
      {ring}
      <div class="al-score-meta">
        <p class="al-score-title">{html.escape(title)}</p>
        <p class="al-score-desc">{html.escape(desc)}</p>
        <div class="al-score-bar-track">
          <div class="al-score-bar-fill" style="width:{score}%;"></div>
        </div>
      </div>
    </div>
    """)


def render_chips(keywords: list[str], chip_type: str) -> str:
    """Build HTML for keyword chips (matched / missing)."""
    limit = 30
    slice_kw = keywords[:limit]
    extra = max(0, len(keywords) - limit)
    cls = "al-chip al-chip-match" if chip_type == "matched" else "al-chip al-chip-miss"
    parts = [f'<span class="{cls}">{html.escape(k)}</span>' for k in slice_kw]
    if extra:
        parts.append(f'<span class="al-chip al-chip-more">+{extra} more</span>')
    inner = "".join(parts) if parts else '<span class="al-chip al-chip-more">—</span>'
    return f'<div class="al-chips">{inner}</div>'


def render_keyword_section(
    title: str,
    keywords: list[str],
    chip_type: str,
    count: int | None = None,
) -> None:
    """Keyword block with header and chips."""
    n = count if count is not None else len(keywords)
    if chip_type == "matched":
        count_label = f"{n} found"
    else:
        count_label = f"{n} gap{'s' if n != 1 else ''}"
    emit_html(f"""
    <div class="al-kw-section">
      <div class="al-kw-header">
        <span class="al-kw-label">{html.escape(title)}</span>
        <span class="al-kw-count">{html.escape(count_label)}</span>
      </div>
      {render_chips(keywords, chip_type)}
    </div>
    """)


def render_insight(insight_text: str) -> None:
    """Left-bordered insight banner."""
    if not insight_text or not insight_text.strip():
        return
    emit_html(
        f'<div class="al-insight">{html.escape(insight_text.strip())}</div>'
    )


def render_hiring_manager_comment(comment: str | None, *, label: str = "Hiring manager view") -> None:
    """Hiring manager simulation card."""
    if not comment or not str(comment).strip():
        return
    body = html.escape(str(comment).strip())
    emit_html(f"""
    <div class="al-mini-coach">
      <div class="al-mini-coach-label">{html.escape(label)}</div>
      <div class="al-mini-coach-text">"{body}"</div>
      <div class="al-mini-coach-name">— AI Hiring Manager Simulation</div>
    </div>
    """)


def render_optimised_results_header(original_score: int, optimised_score: int) -> None:
    """Section title plus before/after score comparison."""
    delta = optimised_score - original_score
    if delta > 0:
        delta_html = f'<span class="al-opt-delta al-opt-delta-up">+{delta} pts</span>'
    elif delta < 0:
        delta_html = f'<span class="al-opt-delta al-opt-delta-down">{delta} pts</span>'
    else:
        delta_html = '<span class="al-opt-delta">no change</span>'
    emit_html(f"""
    <div class="al-opt-results-head">
      <p class="al-results-label">Optimised CV</p>
      <div class="al-opt-compare">
        <div class="al-opt-compare-item">
          <span class="al-opt-compare-k">Original</span>
          <span class="al-opt-compare-v">{original_score}%</span>
        </div>
        <div class="al-opt-compare-arrow">→</div>
        <div class="al-opt-compare-item al-opt-compare-item-new">
          <span class="al-opt-compare-k">Optimised</span>
          <span class="al-opt-compare-v">{optimised_score}%</span>
        </div>
        {delta_html}
      </div>
    </div>
    """)


def render_results_label() -> None:
    emit_html('<p class="al-results-label">03 — Results</p>')


def render_field_label(text: str) -> None:
    emit_html(f'<p class="al-field-label">{html.escape(text)}</p>')


def render_action_buttons_marker() -> None:
    """CSS anchor for the two-column Optimise / Download button row."""
    emit_html('<div class="al-actions-marker"></div>')
