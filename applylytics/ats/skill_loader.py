"""Load cross-industry skill keywords from bundled data file."""

from __future__ import annotations

from pathlib import Path

_DATA_FILE = Path(__file__).resolve().parent / "data" / "skills_base.txt"


def load_skill_keywords(file_path: str | Path | None = None) -> frozenset[str]:
    """Load skills from a comma-separated file; return a fallback set on failure."""
    path = Path(file_path) if file_path else _DATA_FILE
    try:
        content = path.read_text(encoding="utf-8").strip()
        raw_skills = [s.strip().lower() for s in content.split(",") if s.strip()]
        return frozenset(dict.fromkeys(raw_skills))
    except OSError as exc:
        print(f"Warning: Could not load skills file: {exc}. Using fallback set.")
        fallback = {
            "python",
            "java",
            "sql",
            "excel",
            "project management",
            "communication",
            "leadership",
            "data analysis",
            "sap",
            "power bi",
            "machine learning",
            "agile",
            "aws",
            "git",
        }
        return frozenset(fallback)


SKILL_KEYWORDS = load_skill_keywords()
