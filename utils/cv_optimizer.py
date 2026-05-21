"""
Recruiter-grade CV rewrite pipeline (Groq + strict prompt + similarity retry + post-process).
"""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from openai import APIError, OpenAI, RateLimitError as OpenAIRateLimitError

logger = logging.getLogger("applylytics")

_GROQ_CLIENT: OpenAI | None = None
_GEMINI_CLIENT: OpenAI | None = None

_DEBUG = os.getenv("APPLYLYTICS_DEBUG", "").strip().lower() in ("1", "true", "yes", "yes")


def get_groq_model() -> str:
    """Model id for Groq chat completions; override with env GROQ_MODEL."""
    try:
        from applylytics.config import settings

        m = (settings.groq_model or "").strip()
        if m:
            return m
    except ImportError:
        pass
    m = (os.getenv("GROQ_MODEL") or "llama-3.3-70b-versatile").strip()
    return m or "llama-3.3-70b-versatile"


def _resolve_groq_api_key() -> str:
    """Prefer applylytics Settings / Streamlit secrets over raw os.environ."""
    try:
        from applylytics.config import resolve_groq_api_key

        return resolve_groq_api_key()
    except ImportError:
        pass
    return (os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY") or "").strip()


def reset_groq_client() -> None:
    """Drop cached client (e.g. after rotating API keys in .env)."""
    global _GROQ_CLIENT
    _GROQ_CLIENT = None


def get_groq_client() -> OpenAI:
    """Single shared OpenAI-compatible client for Groq (do not create per request)."""
    global _GROQ_CLIENT
    if _GROQ_CLIENT is not None:
        return _GROQ_CLIENT
    api_key = _resolve_groq_api_key()
    if not api_key:
        from applylytics.config import api_key_help_message

        raise RuntimeError(f"GROQ_API_KEY is not set. {api_key_help_message()}")
    _GROQ_CLIENT = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
    return _GROQ_CLIENT


def get_gemini_client() -> OpenAI | None:
    """OpenAI-compatible client for Google Gemini, or None if GEMINI_API_KEY is unset."""
    global _GEMINI_CLIENT
    key = (os.getenv("GEMINI_API_KEY") or "").strip()
    if not key:
        try:
            from applylytics.config import settings

            key = (settings.gemini_api_key or "").strip()
        except Exception:
            pass
    if not key:
        return None
    if _GEMINI_CLIENT is None:
        _GEMINI_CLIENT = OpenAI(
            api_key=key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
    return _GEMINI_CLIENT


def get_gemini_model() -> str:
    """Gemini model id for OpenAI-compat chat (env GEMINI_MODEL overrides)."""
    try:
        from applylytics.config import settings

        m = (settings.gemini_model or "").strip()
        if m:
            return m
    except ImportError:
        pass
    return (os.getenv("GEMINI_MODEL") or "gemini-2.5-flash").strip()


def get_gemini_fallback_models() -> list[str]:
    """Models to try in order when Groq is rate limited."""
    primary = get_gemini_model()
    candidates = [
        primary,
        "gemini-2.5-flash",
        "gemini-2.0-flash-lite",
        "gemini-flash-latest",
        "gemini-2.0-flash",
    ]
    seen: set[str] = set()
    out: list[str] = []
    for model in candidates:
        if model and model not in seen:
            seen.add(model)
            out.append(model)
    return out


def _groq_quota_error_response(
    exc: BaseException,
    *,
    gemini_attempted: bool = False,
) -> dict[str, Any]:
    """User-facing payload when Groq returns 429 (rate / token limits)."""
    detail = str(exc)
    if isinstance(exc, APIError):
        try:
            body = exc.body
            if isinstance(body, dict):
                err = body.get("error")
                if isinstance(err, dict) and err.get("message"):
                    detail = str(err["message"])
        except Exception:
            pass
    hint = (
        "Groq applies token and request limits per organization (the org id in the message), "
        "not per API key. A new key created under the same Groq account still shares that org’s quota, "
        "which is why changing the key alone often changes nothing.\n"
        "Options: wait until the daily/minute window resets; upgrade at "
        "https://console.groq.com/settings/billing ; use a different Groq account; or set "
        "GROQ_MODEL=llama-3.1-8b-instant (or another smaller model) to use fewer tokens per run.\n"
        "Put the key in .env as GROQ_API_KEY or OPENAI_API_KEY and fully restart Streamlit after edits."
    )
    if gemini_attempted:
        hint += (
            "\n\nGemini fallback was attempted (GEMINI_API_KEY is set) but also failed — "
            "often because the free tier quota for that model is exhausted. "
            "Set GEMINI_MODEL=gemini-2.5-flash in .env (confirmed on the OpenAI-compat endpoint) "
            "or wait and retry."
        )
    return {"error": f"{detail}\n\n{hint}", "rate_limited": True}


def _groq_rate_limit_message_text(msg: str) -> bool:
    lo = msg.lower()
    return (
        "error code: 429" in lo
        or "rate limit" in lo
        or "rate_limit_exceeded" in lo
        or "tokens per day" in lo
        or "tokens per minute" in lo
    )


def is_groq_rate_limit_error(obj: Any) -> bool:
    """True when Groq hit token/rate limits (suppress UI; do not treat as generic error)."""
    from applylytics.llm.client import is_groq_rate_limit_error as _is_rl

    return _is_rl(obj)


def groq_quota_user_message(exc: BaseException) -> str:
    """Plain-text user message for Groq 429 / token or rate-limit errors."""
    return str(_groq_quota_error_response(exc)["error"])


def _http_status_429(exc: BaseException) -> bool:
    if isinstance(exc, OpenAIRateLimitError):
        return True
    if isinstance(exc, APIError):
        status = getattr(exc, "status_code", None)
        return status == 429 or str(status) == "429"
    return False


def _gemini_fallback_chat(
    fallback_client: OpenAI,
    fallback_model: str,
    kwargs: dict[str, Any],
) -> tuple[Any, str | None]:
    """Try Gemini models in order via direct chat.completions (no Groq wrapper)."""
    from applylytics.llm.client import RateLimitError as LLMRateLimitError
    from applylytics.llm.client import groq_quota_user_message

    fb_kwargs = dict(kwargs)
    models = get_gemini_fallback_models()
    if fallback_model and fallback_model not in models:
        models.insert(0, fallback_model)

    last_rate_exc: BaseException | None = None
    last_err: str | None = None
    logger.warning(
        "Groq rate limit; retrying with provider=gemini (models: %s)",
        ", ".join(models),
    )

    for model in models:
        attempt_kwargs = dict(fb_kwargs)
        attempt_kwargs["model"] = model
        try:
            response = fallback_client.chat.completions.create(**attempt_kwargs)
            logger.info(
                "LLM chat completion via provider=gemini model=%s",
                model,
            )
            return response, None
        except APIError as exc:
            if _http_status_429(exc):
                last_rate_exc = exc
                logger.warning("Gemini model %s returned 429, trying next model", model)
                continue
            if "response_format" in str(exc).lower() or "json" in str(exc).lower():
                plain = dict(attempt_kwargs)
                plain.pop("response_format", None)
                try:
                    response = fallback_client.chat.completions.create(**plain)
                    logger.info(
                        "LLM chat completion via provider=gemini model=%s (no response_format)",
                        model,
                    )
                    return response, None
                except (OpenAIRateLimitError, LLMRateLimitError) as exc2:
                    last_rate_exc = exc2
                    continue
                except APIError as exc2:
                    if _http_status_429(exc2):
                        last_rate_exc = exc2
                        continue
                    last_err = f"Gemini API error ({getattr(exc2, 'status_code', '?')}): {exc2}"
                    logger.warning("Gemini model %s failed: %s", model, last_err)
                    continue
            last_err = f"Gemini API error ({getattr(exc, 'status_code', '?')}): {exc}"
            logger.warning("Gemini model %s failed: %s", model, last_err)
        except (OpenAIRateLimitError, LLMRateLimitError) as exc:
            last_rate_exc = exc
            logger.warning("Gemini model %s rate limited, trying next model", model)
            continue

    if last_rate_exc is not None:
        raise LLMRateLimitError(groq_quota_user_message(last_rate_exc)) from last_rate_exc
    return None, last_err or "All configured Gemini fallback models failed."


def safe_groq_chat_create(
    client: OpenAI,
    *,
    fallback_client: OpenAI | None = None,
    fallback_model: str | None = None,
    **kwargs: Any,
) -> tuple[Any, str | None]:
    """
    Call Groq chat completions; on rate limit, retry once with fallback_client if set.

    Raises applylytics.llm.client.RateLimitError when both providers are rate limited.
    """
    from applylytics.llm.client import RateLimitError as LLMRateLimitError
    from applylytics.llm.client import groq_quota_user_message

    model = kwargs.get("model", "?")
    can_fallback = fallback_client is not None and bool(fallback_model)

    def _rate_limit_fallback(exc: BaseException) -> tuple[Any, str | None]:
        if not can_fallback:
            raise LLMRateLimitError(groq_quota_user_message(exc)) from exc
        return _gemini_fallback_chat(fallback_client, fallback_model, kwargs)

    try:
        response = client.chat.completions.create(**kwargs)
        logger.info("LLM chat completion via provider=groq model=%s", model)
        return response, None
    except OpenAIRateLimitError as exc:
        return _rate_limit_fallback(exc)
    except LLMRateLimitError:
        raise
    except APIError as exc:
        if _http_status_429(exc):
            return _rate_limit_fallback(exc)
        return None, f"Groq API error ({getattr(exc, 'status_code', '?')}): {exc}"


STRICT_SYSTEM_PROMPT = """You are an elite UK recruiter and CV rewriter with 15 years of experience placing candidates in corporate, analytics, and graduate roles. Your job is to transform a weak CV into a recruiter-grade document.
ABSOLUTE RULES — violation of any rule means the output is rejected and retried:
RULE 0 — VOICE
Write in first-person implied style. This means:
- No subject pronouns (never "I", "we", "my")
- Never use the candidate's name in the profile or bullets
- Start sentences with verbs or nouns: "Delivered...", "MBA graduate..."
- NEVER third person: phrases like "[Name] achieved" or "The candidate developed" are immediate failures
- UK CVs are always written in first-person implied style

RULE 1 — FULL REWRITE MANDATORY
You must discard all original wording except: job titles, employer names, dates, and verified numerical facts (percentages, counts). Every sentence must be rewritten from scratch. Do not paraphrase. Do not keep original sentence structures. If your output shares more than 20% word overlap with the input, it will be detected and rejected.
RULE 1 does not override RULE 9: you must still retain the minimum content, named tools, and source numbers listed there.

RULE 2 — PROFILE MUST CONTAIN (exactly 3 sentences — do not merge):

Each of the three profile sentences must be a complete, standalone sentence of at least 12 words. No fragments. No notes. No bullet-style shorthand.

Sentence 1 template: '[Qualification] graduate from [Institution] ([Year]) with hands-on experience in [2-3 specific skills relevant to target role].'
Profile sentence 1 must use the exact phrasing "with hands-on experience in" (never "with experience in").

Sentence 2 template: 'Delivered [specific achievement with number] through [specific method or tool].'

Sentence 3 template: 'Targeting [specific role type] where [specific value proposition].'

Sentence 1 ONLY: qualification + institution + year as in the template — do not place the main quantified percentage achievement in sentence 1.
Sentence 2 ONLY: the quantified achievement (e.g. 30% from the source must appear in sentence 2, woven into a full sentence using the sentence 2 template).
Sentence 3 ONLY: target role and value proposition using the sentence 3 template.

BAD profile example — never produce this:
'MBA graduate from University of East London, 2025. Reduced manual processing time by 30% in a previous role. Targeting a Junior Data Analyst position.'

GOOD profile example:
'MBA graduate (University of East London, 2025) with hands-on experience in data analysis, process improvement, and Power BI reporting within a UK public sector environment. Delivered a 30% reduction in manual processing time through workflow redesign and developed dashboards that informed senior management decision-making. Targeting Junior Data Analyst roles where rigorous data governance and clear insight communication drive operational performance.'

Maximum 3 sentences total
BANNED from profile: "analytical", "passionate", "dynamic", "results-driven", "motivated", "seeking opportunities", "operational excellence", "proficient in", "expertise in"

RULE 3 — BULLET QUALITY (minimum viable bullet)
A bullet is only acceptable if it contains ALL THREE of these elements:
  A. A specific action verb
  B. A specific tool, dataset, process, or audience
  C. An outcome, scope, or result

Test every bullet before outputting it:
- "Optimised business processes, cutting manual time by 30%"
  FAILS — missing element B (which processes? what system?)
- "Built visual reports with Excel and Power BI"
  FAILS — missing element C (for whom? what decision did it support?)
- "Facilitated stakeholder meetings, driving data-driven recommendations"
  FAILS — "driving data-driven recommendations" is filler, not an outcome

Maximum 20 words per bullet
One idea per bullet — no "and" chains joining two discrete ideas
No self-assessment phrases: banned words are "demonstrating", "utilizing", "showcasing", "leveraging", "highlighting", "proving", "showing"
No passive constructions: "was responsible for", "helped to", "assisted with", "contributed to"
Must start with a strong past-tense action verb
Quantify wherever the source CV contains a number — do not invent numbers

PASSING examples:
✅ "Redesigned graduate outcomes data workflows in Excel and Power BI, reducing manual processing time by 30%."
✅ "Produced Power BI dashboards for senior leadership tracking graduate employment KPIs across 3 academic departments."
✅ "Analysed 50+ employee wellness surveys in Excel, presenting root cause findings and recommendations to HR leadership."

Bullets about data integrity, datasets, workflows, or governance must name at least one concrete tool, platform, or data source from the CV (e.g. Excel, Power BI, SQL, graduate outcomes database) — never a tool-free abstraction.

If a bullet from the source CV is vague (e.g. 'Delivered gap analysis' with no context), do not keep it as-is and do not drop it. Expand it using context from the rest of that role's description.

'Delivered gap analysis' from a data engagement role should become: 'Conducted gap analysis of graduate survey response data, identifying key barriers and reporting findings to academic leadership.'

Never output a bullet under 8 words. A bullet under 8 words is always a truncation failure or an expansion failure.

The final clause of every bullet must state a concrete outcome or recipient. Endings like "promoting data governance", "supporting best practices", "driving operational improvements" are banned filler endings — they describe intent, not outcome. Replace them with who benefited or what measurably changed.

RULE 4 — BULLET COUNT PER ROLE:

Primary role (most recent, most relevant): minimum 4 bullets, maximum 5 bullets
Secondary roles under 12 months: minimum 2 bullets, maximum 2 bullets
Internships or roles under 3 months: minimum 2 bullets, maximum 2 bullets — never reduce to 1

RULE 5 — SKILLS CLEANUP:
Return between 12 and 18 skills total, across three categories (Technical | Analytical | Business). If the source has more, keep the strongest evidenced skills while respecting RULE 9 on named tools.
Protected skills — if evidenced in the source CV (work experience, profile, or certifications), these names must appear in the output skills list unless wholly unsubstantiated: Power BI, Excel, SQL, Python, R, Microsoft 365.
Remove any skill that is not evidenced by the work experience or certifications. Specifically:

Remove "Supply Chain Management" unless the candidate has supply chain job experience
Remove "AI Tools (ChatGPT, Gemini, Claude)" — this is assumed background noise, not a skill
Remove "Google Cloud" if the only evidence is a badge from an intro course — move to certifications only
Remove "Operational Excellence" — meaningless filler
Organise remaining skills into three categories: Technical | Analytical | Business
Return skills as a flat list with category prefix: "Technical: Power BI", "Analytical: Gap Analysis", etc.

RULE 6 — PROJECTS (source only):
If the source CV has no projects section and no project entries, set "projects": [].
Never invent projects. Never output coaching text, placeholders, or advice in the projects field — only real project name + description from the source.
If the source lists projects, extract each with name, description, technologies, and date when present.
RULE 7 — BANNED PHRASES (strip from every field):
"demonstrating", "utilizing", "showcasing", "leveraging", "passionate about", "strong organizational skills",
"fast-paced environment", "go-getter", "team player", "results-driven", "detail-oriented",
"and investigated", "and provided", "and supported", "and developed", "and responded",
"ability to", "demonstrating ability to", "data-driven culture", "best practices" (unless specific)
RULE 9 — CONTENT FLOOR
You must preserve the following minimum content from the source CV. If the source CV contains more substance than this minimum, keep it:
- Primary role (most recent): minimum 4 bullets, maximum 5
- Secondary roles under 12 months: minimum 2 bullets
- Internships under 3 months: minimum 2 bullets — do NOT reduce to 1
- Skills: minimum 12 skills, maximum 18, across three categories
- Never drop a specific tool name (Power BI, Excel, SQL, Python, R) that appears in the source CV unless it is completely unsubstantiated
- Never drop a specific number (50+, 5 team members, 30%) that appears in the source CV
- If the source CV has no certifications section, return "certifications": [] — do not invent certificates or leave placeholder entries

RULE 8 — OUTPUT FORMAT:
Return ONLY a valid JSON object. No markdown. No commentary. No code fences. No explanation.
The JSON must include every top-level key below. Use [] for empty arrays (never null).
Only include work_experience entries and projects that exist in the source CV — do not invent roles or projects.
If the source has projects but no jobs, use "work_experience": [] and populate "projects".
If both exist, include both. key_skills: 8–12 items; skills: fuller list (12–18 where possible).
{
"name": "string",
"contact_phone": "string",
"contact_email": "string",
"contact_location": "string",
"linkedin": "string",
"personal_statement": "string — max 3 sentences",
"key_skills": ["string"],
"skills": ["string"],
"work_experience": [
{"job_title": "string", "employer": "string", "dates": "string", "bullets": ["string"]}
],
"projects": [
{"name": "string", "description": "string", "technologies": ["string"], "date": "string", "link": "string"}
],
"education": [
{"degree": "string", "institution": "string", "dates": "string", "grade": "string"}
],
"certifications": ["string"],
"interests": ["string"],
"references": "string"
}

RULE KEYWORDS — PRESERVATION
Never remove existing skill keywords from the source CV – only add new ones where truthful.
When REQUIRED KEYWORDS are listed in the user message, include those exact phrases verbatim
in personal_statement, work_experience bullets, project descriptions, technologies, or skills where truthful.

CRITICAL FORMATTING RULE: Always insert spaces between words. Never concatenate multiple words together like 'CollegeOfEngineering' – write 'College of Engineering'. Every word must be separated by a space. Acronyms are fine, but normal phrases must have spaces.

SKILLS OUTPUT — NON-NEGOTIABLE:
- Output between 8 and 10 skill items. Never more. Never fewer.
- No category prefixes of any kind (Technical:, Analytical:,
  Business:, or similar). Plain skill names only.
- No duplicates. Check case-insensitively before including each item.
- Every skill must be present in the job description OR directly
  evidenced by a named task in the work experience. If you cannot
  point to a specific bullet that demonstrates it, exclude it.
- Output skills as a single bullet list with • separator.
- If you are about to exceed 10 skills, drop the least relevant ones.
  Do not add placeholder, generic, or unexplained terms under any
  circumstances.
"""


def build_strict_system_prompt() -> str:
    """Return the strict recruiter system prompt (verbatim)."""
    return STRICT_SYSTEM_PROMPT


def extract_cv_data(raw_text: str) -> dict[str, Any]:
    """Lightweight extraction of hints from raw CV text (no LLM)."""
    text = raw_text or ""
    email_m = re.search(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", text)
    phone_m = re.search(r"\+?\d[\d\s\-()]{8,}\d", text)
    return {
        "char_len": len(text),
        "hint_email": email_m.group(0).strip() if email_m else "",
        "hint_phone": phone_m.group(0).strip() if phone_m else "",
    }


def build_user_message(
    raw_text: str,
    job_text: str = "",
    missing_keywords: list[str] | None = None,
) -> str:
    """User message with raw CV plus optional job / keyword context (not part of system prompt)."""
    hints = extract_cv_data(raw_text)
    parts = [
        "Below is the candidate's raw CV text. Apply all rewriting rules from your system prompt.",
        "Return ONLY the JSON object — no markdown, no commentary, no code fences.",
        "",
        "RAW CV:",
        raw_text,
        "",
        "STRUCTURED HINTS (optional corroboration; facts must still match the CV above):",
        f"- Email hint: {hints['hint_email'] or '(none)'}",
        f"- Phone hint: {hints['hint_phone'] or '(none)'}",
    ]
    jt = (job_text or "").strip()
    if jt:
        parts.extend(["", "TARGET JOB DESCRIPTION (tailor honestly; do not invent experience):", jt[:12000]])
    mk = [str(k).strip() for k in (missing_keywords or []) if k and str(k).strip()][:40]
    if mk:
        bullets = "\n".join(f'  • "{kw}"' for kw in mk)
        parts.extend(
            [
                "",
                "REQUIRED KEYWORDS — include these exact phrases verbatim in personal_statement,",
                "work_experience bullets, skills, or key_skills where truthful (do not paraphrase):",
                bullets,
            ]
        )
    parts.extend(
        [
            "",
            "Ensure all institution names, project titles, and skill phrases have proper spaces. "
            "For example, write 'Gayatri Vidya Parishad College of Engineering' not "
            "'GayatriVidyaParishadCollegeOfEngineering'.",
        ]
    )
    return "\n".join(parts)


def _parse_json_from_response(raw: str | None) -> dict[str, Any] | None:
    if not raw or not str(raw).strip():
        return None
    s = str(raw).strip()
    if s.startswith("```"):
        parts = s.split("```")
        if len(parts) >= 2:
            s = parts[1]
            if s.lstrip().lower().startswith("json"):
                s = s.lstrip()[4:].lstrip()
        s = s.strip()
    candidates = [s]
    m = re.search(r"\{[\s\S]*\}", s)
    if m:
        candidates.append(m.group(0).strip())
    for candidate in candidates:
        try:
            out = json.loads(candidate)
            if isinstance(out, dict):
                return out
        except json.JSONDecodeError:
            continue
    return None


def _compute_similarity(text1: str, text2: str) -> float:
    """Cosine similarity between two texts using TF-IDF (0–1)."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        return 0.0
    t1 = (text1 or "").strip().lower()
    t2 = (text2 or "").strip().lower()
    if not t1 or not t2:
        return 0.0
    try:
        vectorizer = TfidfVectorizer(max_features=5000)
        mat = vectorizer.fit_transform([t1, t2])
        return float(cosine_similarity(mat[0:1], mat[1:2])[0][0])
    except Exception:
        return 0.0


def _flatten_profile_and_bullets(parsed: dict[str, Any]) -> str:
    ps = str(parsed.get("personal_statement", "") or "")
    bullets: list[str] = []
    for job in parsed.get("work_experience") or []:
        if not isinstance(job, dict):
            continue
        for b in job.get("bullets") or []:
            if isinstance(b, str) and b.strip():
                bullets.append(b.strip())
    for proj in parsed.get("projects") or []:
        if isinstance(proj, dict) and proj.get("description"):
            bullets.append(str(proj["description"]).strip())
    return " ".join([ps, *bullets])


def _fetch_chat_completion(
    client: OpenAI,
    *,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    json_mode: bool,
) -> tuple[Any, str | None]:
    """Groq chat with optional Gemini fallback; returns (response, error_message)."""
    from applylytics.llm.client import RateLimitError as LLMRateLimitError

    fallback_client = get_gemini_client()
    fallback_model = get_gemini_model()
    logger.warning(
        "Gemini client available: %s (fallback models: %s)",
        fallback_client is not None,
        ", ".join(get_gemini_fallback_models()) if fallback_client else "n/a",
    )
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 8192,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    try:
        response, err = safe_groq_chat_create(
            client,
            fallback_client=fallback_client,
            fallback_model=fallback_model,
            **kwargs,
        )
        if err is None:
            return response, None
        if json_mode:
            plain_kwargs = {k: v for k, v in kwargs.items() if k != "response_format"}
            return safe_groq_chat_create(
                client,
                fallback_client=fallback_client,
                fallback_model=fallback_model,
                **plain_kwargs,
            )
        return None, err
    except (OpenAIRateLimitError, LLMRateLimitError):
        raise
    except APIError as exc:
        if _http_status_429(exc):
            from applylytics.llm.client import groq_quota_user_message

            raise LLMRateLimitError(groq_quota_user_message(exc)) from exc
        raise


def call_groq_with_retry(
    client: OpenAI,
    system_prompt: str,
    user_message_base: str,
    raw_text: str,
    *,
    max_retries: int = 3,
    initial_temperature: float = 0.7,
) -> dict[str, Any]:
    """
    Calls Groq. Retries on JSON failure or excessive similarity to the raw CV
    (profile + bullets only for the similarity slice).
    """
    warnings: list[str] = []
    temperature = initial_temperature
    last_parsed: dict[str, Any] | None = None
    last_similarity: float | None = None

    model = get_groq_model()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": ""},
    ]

    for attempt in range(max_retries):
        user_message = user_message_base + "".join(warnings)
        messages[1] = {"role": "user", "content": user_message}
        try:
            response, api_err = _fetch_chat_completion(
                client,
                model=model,
                messages=messages,
                temperature=temperature,
                json_mode=True,
            )
        except Exception as exc:
            from applylytics.llm.client import RateLimitError as LLMRateLimitError

            if isinstance(exc, (LLMRateLimitError, OpenAIRateLimitError)):
                return _groq_quota_error_response(
                    exc,
                    gemini_attempted=get_gemini_client() is not None,
                )
            raise
        if api_err:
            return {"error": api_err}

        raw_output = (response.choices[0].message.content or "").strip()
        parsed = _parse_json_from_response(raw_output)
        if parsed is None:
            temperature = min(temperature + 0.1, 1.0)
            logger.warning(
                "CV optimiser JSON parse failed on attempt %s (output_len=%s, preview=%r)",
                attempt + 1,
                len(raw_output),
                raw_output[:200],
            )
            warnings.append(
                "\n\nCRITICAL: Your last response was not valid JSON. "
                "Return ONLY one JSON object matching the schema in the system prompt. "
                "No markdown fences, no commentary, no text before or after the JSON."
            )
            if _DEBUG:
                print(f"[applylytics-debug] attempt={attempt + 1} JSON parse failed; temp={temperature}", flush=True)
            continue

        processed, pstats = post_process_output(parsed)

        if processed.get("_needs_retry"):
            reason = str(processed.pop("_retry_reason", "Content floor violated"))
            processed.pop("_needs_retry", None)
            warnings.append(
                f"\n\nCRITICAL FAILURE: {reason} "
                "You must output at least 2 complete bullets for every role. "
                "Expand the existing content — do not invent new experience."
            )
            temperature = min(temperature + 0.1, 1.0)
            if _DEBUG:
                print(
                    f"[applylytics-debug] attempt={attempt + 1} content_floor_retry temp={temperature}",
                    flush=True,
                )
            continue

        last_parsed = processed
        output_text = _flatten_profile_and_bullets(processed)
        similarity = _compute_similarity(raw_text, output_text)
        last_similarity = similarity

        if _DEBUG:
            print(
                f"[applylytics-debug] attempt={attempt + 1} similarity={similarity:.4f} "
                f"temp={temperature} len_out={len(output_text)} "
                f"banned_removals={pstats['banned_removals']} bullets_truncated={pstats['bullets_truncated']}",
                flush=True,
            )

        if similarity < 0.60:
            if _DEBUG:
                print(f"[applylytics-debug] accepted on attempt {attempt + 1}", flush=True)
            return processed

        warnings.append(
            f"\n\nWARNING: Your previous output had {similarity:.0%} word overlap "
            "with the original (measured on profile + bullets). This is too high. "
            "You MUST rewrite every sentence from scratch. Do not reuse phrases from the input."
        )
        temperature = min(temperature + 0.15, 1.0)

    if last_parsed is not None:
        if _DEBUG:
            print(
                "[applylytics-debug] returning last attempt despite similarity="
                f"{last_similarity:.4f}",
                flush=True,
            )
        return last_parsed
    return {
        "error": (
            "The model did not return valid JSON after several attempts. "
            "Wait a minute and try again, or set GROQ_MODEL / GEMINI_MODEL to a model with higher limits. "
            "If using Gemini fallback, ensure GEMINI_MODEL=gemini-2.5-flash in .env."
        )
    }


BANNED_PHRASES = [
    "passionate about",
    "strong organizational skills",
    "fast-paced environment",
    "team player",
    "and investigated",
    "and provided",
    "and supported",
    "and developed",
    "and responded",
    "ability to",
    "demonstrating ability to",
    "100% accuracy",
    "driving data-driven recommendations",
    "in a previous role",
    "best practices and team efficiency",
    "promoting adherence to",
    "supporting strategic improvements",
]


def post_process_output(parsed: dict[str, Any]) -> tuple[dict[str, Any], dict[str, int]]:
    """
    Strips banned phrases, enforces bullet word limits, trims bullet counts.
    Returns (cleaned dict, stats) for optional debug logging.
    """
    stats = {"banned_removals": 0, "bullets_truncated": 0}

    def clean_text(text: str) -> str:
        t = text or ""
        for phrase in BANNED_PHRASES:
            if len(phrase.split()) >= 2:
                pattern = r"\b" + re.escape(phrase) + r"\b"
                if re.search(pattern, t, flags=re.IGNORECASE):
                    stats["banned_removals"] += 1
                t = re.sub(pattern, "", t, flags=re.IGNORECASE)
        t = re.sub(r"\s+,", ",", t)
        t = re.sub(r",\s*,", ",", t)
        t = re.sub(r"\s{2,}", " ", t)
        t = re.sub(r",\s*\.", ".", t)
        return t.strip()

    def enforce_bullet_rules(bullet: str) -> str:
        cleaned = clean_text(bullet)

        cleaned = re.sub(r"\s+with\s*,\s*", ", ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r",?\s+in\s+a\s*\.", ".", cleaned, flags=re.IGNORECASE)

        words = cleaned.split()
        if len(words) > 35:
            stats["bullets_truncated"] += 1
            truncated = " ".join(words[:35])
            last_stop = truncated.rfind(".")
            if last_stop > len(truncated) * 0.5:
                cleaned = truncated[: last_stop + 1]
            else:
                last_comma = truncated.rfind(",")
                if last_comma > len(truncated) * 0.5:
                    cleaned = truncated[:last_comma] + "."
                else:
                    cleaned = truncated.rstrip(".,;") + "."

        dangling = [
            r",?\s+with\s*,?\s*$",
            r",?\s+and\s*,?\s*$",
            r",?\s+or\s*,?\s*$",
            r",?\s+to\s*,?\s*$",
            r",?\s+for\s*,?\s*$",
            r",?\s+in\s*,?\s*$",
            r",?\s+of\s*,?\s*$",
            r"\s*,\s*$",
        ]
        for pattern in dangling:
            cleaned = re.sub(pattern, ".", cleaned, flags=re.IGNORECASE)

        cleaned = re.sub(r"\.{2,}", ".", cleaned)
        cleaned = cleaned.strip()
        if cleaned and not cleaned.endswith("."):
            cleaned += "."
        return cleaned

    out = dict(parsed)
    ps = clean_text(str(out.get("personal_statement", "") or ""))
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", ps) if p.strip()]
    out["personal_statement"] = " ".join(parts[:3]).strip()

    px = out.get("projects")
    if isinstance(px, list):
        for proj in px:
            if isinstance(proj, dict) and proj.get("description"):
                proj["description"] = clean_text(str(proj["description"]))

    wx = out.get("work_experience")
    if isinstance(wx, list) and wx:
        for job in wx:
            if not isinstance(job, dict):
                continue
            bs = job.get("bullets")
            if not isinstance(bs, list):
                job["bullets"] = []
                continue
            job["bullets"] = [enforce_bullet_rules(str(b)) for b in bs if str(b).strip()]
            job["bullets"] = [b for b in job["bullets"] if len(b.split()) >= 8]
        if wx:
            if isinstance(wx[0], dict) and isinstance(wx[0].get("bullets"), list):
                wx[0]["bullets"] = wx[0]["bullets"][:5]
            for job in wx[1:]:
                if isinstance(job, dict) and isinstance(job.get("bullets"), list):
                    job["bullets"] = job["bullets"][:2]

            for i, job in enumerate(wx):
                if not isinstance(job, dict):
                    continue
                min_bullets = 4 if i == 0 else 2
                actual = len(job.get("bullets") or [])
                if actual < min_bullets:
                    print(
                        f"[CONTENT FLOOR VIOLATION] {job.get('job_title')} "
                        f"has {actual} bullets, minimum is {min_bullets}. "
                        f"Triggering retry with explicit instruction.",
                        flush=True,
                    )
                    out["_needs_retry"] = True
                    out["_retry_reason"] = (
                        f"{job.get('job_title')} needs {min_bullets} bullets, "
                        f"currently has {actual}."
                    )
                    break

    if not isinstance(out.get("interests"), list):
        out["interests"] = []

    out["skills"] = _filter_and_bucket_skills(out)
    return out, stats


def _filter_and_bucket_skills(parsed: dict[str, Any]) -> list[str]:
    """Apply RULE 5 removals on the skills list strings."""
    skills = parsed.get("skills")
    if not isinstance(skills, list):
        return []
    work_blob = " ".join(
        " ".join(str(b) for b in (j.get("bullets") or []) if isinstance(j, dict))
        for j in (parsed.get("work_experience") or [])
        if isinstance(j, dict)
    ).lower()
    project_blob = " ".join(
        str(p.get("description", "")).lower()
        + " "
        + " ".join(str(t).lower() for t in (p.get("technologies") or []) if isinstance(p, dict))
        for p in (parsed.get("projects") or [])
        if isinstance(p, dict)
    )
    work_blob = work_blob + " " + project_blob
    job_titles = " ".join(
        str(j.get("job_title", "")).lower() + " " + str(j.get("employer", "")).lower()
        for j in (parsed.get("work_experience") or [])
        if isinstance(j, dict)
    )
    cert_blob = " ".join(str(c).lower() for c in (parsed.get("certifications") or []) if c).lower()
    evidence = work_blob + " " + cert_blob + " " + job_titles

    out: list[str] = []
    for s in skills:
        if not isinstance(s, str):
            continue
        t = s.strip()
        if not t:
            continue
        low = t.lower()
        if "supply chain" in low:
            sc_ev = any(
                k in evidence
                for k in (
                    "supply chain",
                    "logistics",
                    "inventory",
                    "procurement",
                    "warehouse",
                    "demand planning",
                )
            )
            if not sc_ev:
                continue
        if "chatgpt" in low or "gemini" in low or "claude" in low or "ai tools" in low:
            continue
        if "operational excellence" in low:
            continue
        if "google cloud" in low and "google cloud" not in work_blob and "gcp" not in work_blob:
            if "introduction to generative ai" in cert_blob or "intro" in cert_blob:
                continue
        out.append(t)
    return out


def validate_schema(parsed: dict[str, Any]) -> dict[str, Any]:
    """Ensure required keys and nested shapes; fill safe defaults."""
    from applylytics.cv.schema import normalize_cv_json

    p = dict(parsed)
    p.pop("_needs_retry", None)
    p.pop("_retry_reason", None)
    p.pop("rate_limited", None)
    return normalize_cv_json(p)


def run_generic_cv_optimisation_pipeline(
    raw_cv_text: str,
    *,
    job_text: str = "",
    missing_keywords: list[str] | None = None,
    keyword_requirements_block: str = "",
) -> dict[str, Any]:
    """
    Full rewrite: strict system prompt → Groq with similarity retry → post-process → validate.
    """
    try:
        from dotenv import load_dotenv

        env_path = Path(__file__).resolve().parent.parent / ".env"
        if env_path.is_file():
            load_dotenv(env_path, override=True)
    except Exception:
        pass
    reset_groq_client()

    rt = (raw_cv_text or "").strip()
    if len(rt) < 80:
        return {
            "error": "Resume text is too short to optimise (PDF may be image-only or empty). "
            "Re-upload a text-based PDF or check the extracted preview.",
        }

    try:
        client = get_groq_client()
    except RuntimeError as e:
        return {"error": str(e)}

    system_prompt = build_strict_system_prompt()
    user_message = build_user_message(rt, job_text=job_text, missing_keywords=missing_keywords)
    if keyword_requirements_block:
        user_message = user_message + keyword_requirements_block

    groq_out = call_groq_with_retry(
        client,
        system_prompt,
        user_message,
        rt,
        max_retries=3,
        initial_temperature=0.2,
    )
    if groq_out.get("error"):
        return groq_out

    validated = validate_schema(groq_out)

    return validated


if __name__ == "__main__":
    import sys

    from dotenv import load_dotenv

    load_dotenv()

    root = Path(__file__).resolve().parent.parent
    sample_path = root / "Naveen_Reddy_Chirla_CV_Buitelaar.docx"
    text = ""
    if sample_path.is_file():
        from docx import Document

        text = "\n".join(p.text for p in Document(sample_path).paragraphs if p.text.strip())
    if not text.strip():
        print("Add Naveen_Reddy_Chirla_CV_Buitelaar.docx at project root to smoke-test.", file=sys.stderr)
        sys.exit(0)
    os.environ.setdefault("APPLYLYTICS_DEBUG", "true")
    out = run_generic_cv_optimisation_pipeline(text[:15000], job_text="", missing_keywords=[])
    print(json.dumps(out, indent=2, ensure_ascii=False))


def build_targeted_keyword_requirements(
    missing_keywords: list[str],
    matched_keywords: list[str],
    job_description: str,
) -> str:
    """Formatted block appended to the Groq user message for ATS-gap-targeted rewrites."""
    missing = ", ".join(
        str(k).strip() for k in (missing_keywords or []) if k and str(k).strip()
    )
    _ = job_description  # industry context is in the user message job_text; kept for API symmetry
    _ = matched_keywords  # ATS overlap context; skills rules live in STRICT_SYSTEM_PROMPT
    keyword_section = ""
    if missing:
        keyword_section = f"""
KEYWORD REQUIREMENTS:
The following skills appear in the job description but are ABSENT from
the resume. You MUST weave each one naturally into a work bullet or the
profile — only where it is plausible given the candidate's real history.
Do not invent employers, qualifications, or dates.
Missing: {missing}
"""
    jd_alignment_section = """
JD ALIGNMENT — use these mappings where truthful:
- Gap analysis work → frame as 'as-is process analysis'
- Requirements gathering from stakeholders → frame as
  'requirements elicitation'
- Dashboard/report design → frame as 'producing mock-ups or
  visual specifications'
- Do not use these framings if the underlying task does not
  genuinely match. Preserve all original facts.
"""
    return f"""{keyword_section}{jd_alignment_section}
BULLET REWRITE RULES:
Do NOT append keywords as trailing phrases to existing bullets.
If a skill needs evidencing, rewrite the bullet so the skill is
the main verb or subject of a real action.
Example of what NOT to do:
  'Processed transactions ensuring data accuracy, supporting business analysis.'
Example of correct approach:
  'Mapped current-state stock replenishment process using HHT system data,
   identifying 2 bottlenecks and recommending process improvements.'
Every bullet must describe a concrete action with a visible output.

PROFILE REWRITE RULES:
- Identify the target industry from the job description.
- Rewrite the profile opening to reference that industry by name.
- Frame any lack of direct experience as genuine interest plus
  transferable skills. Do not claim experience that does not exist.
- Preserve all quantified achievements (percentages, numbers, counts).

HARD CONSTRAINTS:
- Do NOT change job titles, employer names, dates, or qualifications.
- Do NOT add new employers or roles.
- Return the complete rewritten CV as structured JSON matching the
  existing schema — same format as before.
"""
