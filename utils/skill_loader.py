"""Load cross‑industry skill keywords from a CSV file."""
from pathlib import Path

def load_skill_keywords(file_path: str = None) -> frozenset:
    """
    Load skills from a comma‑separated CSV file (one row, many columns).
    Returns a frozenset of lowercased, stripped skill strings.
    If the file is missing or parsing fails, return a fallback set.
    """
    if file_path is None:
        file_path = Path(__file__).parent / "skills_base.txt"

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        # Split by commas and clean each token
        raw_skills = [s.strip().lower() for s in content.split(",") if s.strip()]
        # Remove duplicates while preserving order (optional)
        unique_skills = list(dict.fromkeys(raw_skills))
        return frozenset(unique_skills)
    except Exception as e:
        print(f"Warning: Could not load skills CSV: {e}. Using fallback set.")
        # Fallback set of essential cross‑industry skills
        fallback = {
            "python", "java", "sql", "excel", "project management", "communication",
            "leadership", "data analysis", "marketing", "accounting", "design",
            "customer service", "problem solving", "teamwork", "cloud computing",
            "agile", "scrum", "sap", "power bi", "tableau", "javascript", "html",
            "css", "machine learning", "ai", "devops", "ci/cd", "docker", "kubernetes",
            "aws", "azure", "gcp", "rest api", "git", "jenkins", "salesforce"
        }
        return frozenset(fallback)

SKILL_KEYWORDS = load_skill_keywords()
