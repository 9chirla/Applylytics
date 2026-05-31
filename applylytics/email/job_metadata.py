"""Parse job title, company, and location from pasted job descriptions."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class JobMetadata:
    job_title: str
    company_name: str
    location: str = ""

    @property
    def has_company(self) -> bool:
        return bool(self.company_name) and not is_junk_metadata_text(self.company_name)

    @property
    def display_title(self) -> str:
        """Role label for email/subject (never section headings like 'About the job')."""
        if is_junk_metadata_text(self.job_title):
            return "your target role"
        return self.job_title

    def role_line(self) -> str:
        title = self.display_title
        if self.has_company:
            return f"{title} at {self.company_name}"
        return title


_JUNK_METADATA = frozenset(
    {
        "about the job",
        "the job",
        "this job",
        "job description",
        "about us",
        "about the role",
        "the role",
        "role overview",
        "job overview",
        "company",
        "the company",
        "unknown",
        "responsibilities",
        "requirements",
        "qualifications",
        "benefits",
        "description",
    }
)


def is_junk_metadata_text(text: str) -> bool:
    """True for JD section headings, not real titles or employers."""
    low = re.sub(r"\s+", " ", (text or "").strip().lower())
    if not low or low in _JUNK_METADATA:
        return True
    if low.startswith("about "):
        return True
    if re.match(r"^the\s+job\b", low):
        return True
    if re.match(r"^job\s+(description|details|summary)\b", low):
        return True
    if re.match(r"^(responsibilities|requirements|qualifications|benefits)\b", low):
        return True
    return False


_TITLE_LABEL = re.compile(
    r"^(?:job\s*title|position|role|vacancy|opening)\s*[:\-\u2013]\s*(.+)$",
    re.I,
)
_COMPANY_LABEL = re.compile(
    r"^(?:company|employer|organization|organisation|hiring\s+company)\s*[:\-\u2013]\s*(.+)$",
    re.I,
)
_LOCATION_LABEL = re.compile(
    r"^(?:location|office|based\s+in|workplace)\s*[:\-\u2013]\s*(.+)$",
    re.I,
)
_AT_SPLIT = re.compile(r"\s+at\s+", re.I)
_JOIN_LINE = re.compile(
    r"^(?:join|work\s+at|careers\s+at)\s+(.+?)(?:\s*[\.\!\|]|$)",
    re.I,
)
_HIRING_LINE = re.compile(
    r"^(.+?)\s+(?:is\s+)?(?:hiring|recruiting|careers|jobs)\b",
    re.I,
)
_LOCATION_TAIL = re.compile(
    r"^(.+?)\s*[\(\[]\s*([^)\]]{2,60})\s*[\)\]]\s*$",
)


def _clean(s: str, *, max_len: int = 80) -> str:
    t = re.sub(r"\s+", " ", (s or "").strip())
    t = t.strip("·•|-–—:;,")
    return t[:max_len] if t else ""


def _looks_like_location(line: str) -> bool:
    low = line.lower()
    if re.search(r"\b(remote|hybrid|on-?site|full-?time|part-?time|contract)\b", low):
        return True
    if re.search(
        r"\b(uk|usa|u\.s\.|london|manchester|birmingham|leeds|glasgow|dublin|europe)\b",
        low,
    ):
        return True
    return bool(re.match(r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}$", line)) and len(line) < 40


def _looks_like_title(line: str) -> bool:
    if len(line) < 4 or len(line) > 100:
        return False
    if is_junk_metadata_text(line):
        return False
    if _looks_like_location(line):
        return False
    low = line.lower()
    if any(
        x in low
        for x in (
            "responsibilities",
            "requirements",
            "qualifications",
            "description",
            "benefits",
            "apply now",
            "http",
            "www.",
        )
    ):
        return False
    return True


def extract_job_metadata(job_description: str) -> JobMetadata:
    """Best-effort job title, company, and location from pasted job text."""
    lines = [ln.strip() for ln in (job_description or "").splitlines() if ln.strip()]
    if not lines:
        return JobMetadata(job_title="this role", company_name="")

    title = ""
    company = ""
    location = ""

    for line in lines[:12]:
        m = _TITLE_LABEL.match(line)
        if m and not title:
            title = _clean(m.group(1))
            continue
        m = _COMPANY_LABEL.match(line)
        if m and not company:
            company = _clean(m.group(1))
            continue
        m = _LOCATION_LABEL.match(line)
        if m and not location:
            location = _clean(m.group(1), max_len=60)
            continue
        m = _JOIN_LINE.match(line)
        if m and not company:
            cand = _clean(m.group(1))
            if not is_junk_metadata_text(cand):
                company = cand
            continue
        m = _HIRING_LINE.match(line)
        if m and not company:
            company = _clean(m.group(1), max_len=60)

    if not title and lines:
        first = lines[0]
        paren = _LOCATION_TAIL.match(first)
        if paren:
            title = _clean(paren.group(1))
            location = location or _clean(paren.group(2), max_len=60)
        elif _AT_SPLIT.search(first):
            parts = _AT_SPLIT.split(first, maxsplit=1)
            if parts[0].strip():
                title = _clean(parts[0])
            if len(parts) > 1 and parts[1].strip():
                cand = _clean(parts[1])
                if not is_junk_metadata_text(cand):
                    company = cand
        elif _looks_like_title(first):
            title = _clean(first)

    if not company:
        for line in lines[1:6]:
            if _looks_like_location(line):
                location = location or _clean(line, max_len=60)
                continue
            if 3 < len(line) < 70 and not _looks_like_title(line):
                if line.endswith(":"):
                    continue
                if "@" not in line and "http" not in line.lower():
                    cand = _clean(line)
                    if not is_junk_metadata_text(cand):
                        company = cand
                        break

    if not title and lines:
        title = _clean(lines[0]) if _looks_like_title(lines[0]) else "this role"
    if not company:
        company = ""

    if is_junk_metadata_text(title):
        title = "this role"
    if is_junk_metadata_text(company):
        company = ""

    return JobMetadata(
        job_title=title or "this role",
        company_name=company,
        location=location,
    )


def build_job_metadata(
    job_description: str,
    *,
    company_override: str = "",
    job_title_override: str = "",
) -> JobMetadata:
    """Parse JD text; user-entered company/title override parsed values when set."""
    meta = extract_job_metadata(job_description)
    company = (company_override or "").strip()
    title = (job_title_override or "").strip()
    if company and not is_junk_metadata_text(company):
        meta = JobMetadata(
            job_title=meta.job_title,
            company_name=company,
            location=meta.location,
        )
    if title and not is_junk_metadata_text(title):
        meta = JobMetadata(
            job_title=title,
            company_name=meta.company_name,
            location=meta.location,
        )
    return meta
