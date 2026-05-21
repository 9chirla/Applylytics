"""ATS-style keyword overlap with improved stopword filtering."""
import re
from typing import Set

from .skill_loader import SKILL_KEYWORDS as _SKILLS_FROM_FILE

# ~160 common English stopwords (no NLTK dependency)
STOPWORDS = frozenset({
    "a", "about", "above", "after", "again", "against", "all", "also",
    "although", "am", "among", "an", "and", "any", "are", "as",
    "at", "be", "because", "been", "before", "being", "below", "between",
    "both", "but", "by", "can", "cannot", "could", "did", "do",
    "does", "doing", "done", "down", "during", "each", "even", "ever",
    "every", "everyone", "everything", "few", "for", "from", "further", "had",
    "has", "have", "having", "he", "her", "here", "hers", "herself", "him",
    "himself", "his", "how", "i", "if", "in", "instead", "into",
    "is", "it", "its", "itself", "just", "many", "may", "maybe",
    "me", "might", "more", "most", "much", "must", "my", "myself",
    "neither", "never", "next", "no", "nobody", "none", "nor", "not",
    "nothing", "now", "nowhere", "of", "off", "on", "once", "only",
    "or", "other", "others", "our", "ours", "ourselves", "out", "over",
    "own", "same", "shall", "she", "should", "so", "some", "such",
    "than", "that", "the", "their", "them", "theirs", "themselves", "then", "there",
    "these", "they", "this", "those", "through", "to", "too", "under",
    "until", "up", "us", "used", "using", "very", "was", "we",
    "were", "what", "when", "where", "which", "while", "who", "whom",
    "whose", "why", "will", "with", "within", "without", "would", "yet",
    "you", "your", "yours", "yourself", "yourselves",
})

# Extra multi-word / role phrases (not always present as single CSV tokens).
_LEGACY_PHRASES = frozenset({
    "financial analysis", "decision making", "governance", "monthly business forecasting",
    "strategic planning", "project accounting", "journal preparation", "account reconciliations", "post close activities",
    "financial aid", "key stakeholders", "quarterly reporting", "pricing activity", "commercial bid",
    "balance multiple priorities", "tight deadlines", "complex financial information", "non-finance staff",
    "ad hoc analysis", "finance roles", "organised", "team goals", "verbal",
    "written", "business works", "motivated to learn", "financial capability", "proactive", "new ideas",
    "finance systems", "different teams", "matrix environment", "problem-solving",
    "issue resolution", "academic", "placement", "reporting requirements", "booking code", "time recording",
    "namerun reviews", "month-end close", "eac programme reporting", "lrp activities",
    "commercial bid activities", "communicate complex financial information", "wider finance team",
    "aat qualified", "cima preferred", "relevant years experience", "comfortable working with data",
    "summarise simple clear messages", "manage multiple tasks", "stay organised", "working to deadlines",
    "working collaboratively", "contributing positively to team goals", "strong verbal communication",
    "written communication skills", "curious about how the business works", "developing analytical skills",
    "building financial capability", "confident using microsoft office", "especially excel",
    "proactive eager to learn", "bring forward new ideas", "exposure to sap", "other finance systems",
    "experience working with stakeholders", "across different teams", "experience working in matrix environment",
    "early experience supporting problem-solving", "issue resolution in academic work placement settings",
})

SKILL_KEYWORDS = _SKILLS_FROM_FILE | _LEGACY_PHRASES

# Short unigrams allowed through extraction when len <= 3 (plus common abbreviations not in set).
_SHORT_SKILL_EXTRA_UNIGRAMS = frozenset({"bi"})
SHORT_SKILL_UNIGRAMS = (
    frozenset(t for t in SKILL_KEYWORDS if " " not in t and len(t) <= 3)
    | _SHORT_SKILL_EXTRA_UNIGRAMS
)


def clean_text(text: str) -> str:
    """Lowercase, keep letters/digits/spaces, collapse spaces."""
    lowered = text.lower()
    kept = re.sub(r"[^a-z0-9\s]+", " ", lowered)
    return re.sub(r"\s+", " ", kept).strip()


def _is_meaningful(token: str, min_keyword_length: int = 2) -> bool:
    """Return False if token is a stopword, purely numeric, or too short without abbreviation."""
    if token in STOPWORDS:
        return False
    if token.isdigit():
        return False
    if len(token) <= 3:
        return token in SHORT_SKILL_UNIGRAMS
    if len(token) < min_keyword_length:
        return False
    return True


def extract_keywords(text: str, min_keyword_length: int = 2) -> Set[str]:
    """
    Extract meaningful unigrams and bigrams from cleaned text,
    filtering out stopwords, numeric-only tokens, and short tokens.
    """
    cleaned = clean_text(text)
    raw_words = cleaned.split()
    filtered = [w for w in raw_words if _is_meaningful(w, min_keyword_length)]

    unigrams: Set[str] = set(filtered)
    bigrams: Set[str] = set()
    for i in range(len(filtered) - 1):
        bigrams.add(f"{filtered[i]} {filtered[i + 1]}")

    return unigrams | bigrams


def _is_skill_keyword(token: str) -> bool:
    """
    True only for whitelist hits: full phrases (including bigrams) or short unigrams
    that explicitly appear in ``SKILL_KEYWORDS`` (e.g. ``sap``, ``excel``).
    """
    return token in SKILL_KEYWORDS


def get_score_insight(score: int) -> str:
    """Short interpretation of the ATS score for UI copy."""
    if score >= 70:
        return "Strong overlap on whitelisted skill phrases for this job."
    if score >= 40:
        return "Moderate overlap; consider weaving in missing phrases where truthful."
    return "Low overlap on whitelisted skills; prioritize the missing keywords that fit your experience."


def calculate_ats_score(resume_text: str, job_text: str, min_keyword_length: int = 2) -> dict:
    job_candidates = extract_keywords(job_text, min_keyword_length=min_keyword_length)
    resume_candidates = extract_keywords(resume_text, min_keyword_length=min_keyword_length)

    job_skills = {kw for kw in job_candidates if _is_skill_keyword(kw)}
    resume_skills = {kw for kw in resume_candidates if _is_skill_keyword(kw)}

    matched = resume_skills & job_skills
    missing = job_skills - resume_skills

    score = round((len(matched) / len(job_skills)) * 100) if job_skills else 0

    return {
        "score": score,
        "matched_keywords": sorted(matched),
        "missing_keywords": sorted(missing),
        "job_keyword_count": len(job_skills),
        "insight": get_score_insight(score),
    }


def calculate_recruiter_penalties(resume_text: str) -> dict:
    """
    Heuristic penalties for common recruiter turn-offs on the resume text.
    Does not affect the whitelist ATS keyword score — use for a separate presentation read.
    """
    text = resume_text or ""
    lowered = text.lower()
    penalties: list[str] = []
    total_deduction = 0

    # 1. Generic profile language
    generic = [
        "hardworking",
        "team player",
        "motivated",
        "passionate",
        "results-driven",
        "detail-oriented",
    ]
    found: list[str] = []
    for w in generic:
        if re.search(rf"\b{re.escape(w)}\b", text, re.I):
            found.append(w)
    if found:
        penalties.append(
            f"Generic profile phrases: {', '.join(found)} – replace with specific achievements"
        )
        total_deduction += 10

    # 2. Interests section
    if re.search(r"\bINTERESTS?\b", text, re.I):
        penalties.append("Remove 'Interests' section – it adds no value")
        total_deduction += 5

    # 3. References section
    if re.search(r"\bREFERENCES?\b", text, re.I):
        penalties.append("Remove 'References' section – it's assumed")
        total_deduction += 5

    # 4. No Projects section
    if not re.search(r"\bPROJECTS?\b", text, re.I):
        penalties.append(
            "Missing 'Projects' section – add personal projects (GitHub, Kaggle, dashboards)"
        )
        total_deduction += 15

    # 5. Retail role with more than 2 bullets
    lines = text.split("\n")
    in_retail = False
    bullet_count = 0
    for line in lines:
        if "Iceland" in line or "Supermarket" in line or "Retail Assistant" in line:
            in_retail = True
        elif in_retail and line.strip().startswith("-"):
            bullet_count += 1
        elif in_retail and len(line.strip()) > 0 and not line.strip().startswith("-"):
            in_retail = False
            break
    if bullet_count > 2:
        penalties.append(f"Retail role has {bullet_count} bullets – cut to max 2")
        total_deduction += 8

    # 6. Bloated bullets (multiple "and"s in one sentence)
    sentences = re.split(r"[.!?]", text)
    bloated = 0
    for s in sentences:
        if s.count(" and ") >= 2:
            bloated += 1
    if bloated > 3:
        penalties.append(
            f"Bloated bullets: {bloated} sentences have multiple 'and's – split them"
        )
        total_deduction += min(10, bloated * 2)

    # 7. AI filler phrases
    ai_phrases = [
        "translating business needs",
        "data-driven recommendations",
        "demonstrated analytical troubleshooting",
    ]
    for phrase in ai_phrases:
        if phrase.lower() in lowered:
            penalties.append(f"AI‑sounding filler: '{phrase}' – rewrite in plain English")
            total_deduction += 5
            break

    # 8. Unsubstantiated skills / AI tools in skills
    if "supply chain management" in lowered and not re.search(
        r"logistics|inventory|stock|procurement",
        lowered,
    ):
        penalties.append(
            "'Supply Chain Management' listed but no evidence – remove or add proof"
        )
        total_deduction += 5
    if any(ai in lowered for ai in ["chatgpt", "gemini", "claude"]):
        penalties.append(
            "'AI Tools (ChatGPT etc.)' – remove; it's expected, not a differentiator"
        )
        total_deduction += 3

    return {
        "penalties": penalties,
        "total_deduction": total_deduction,
        "adjusted_score": max(0, 100 - total_deduction),
    }
