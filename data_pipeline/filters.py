import re
from pathlib import Path


DEFAULT_BLOCKED_PATH_KEYWORDS = (
    "wikipedia",
    "wikidata",
    "encyclopedia",
    "britannica",
    "fandom/wiki",
)

FACTUAL_PATTERNS = [
    re.compile(r"\b(capital of|population of|located in|born in|died in)\b", re.IGNORECASE),
    re.compile(r"\b(president of|prime minister of|ceo of)\b", re.IGNORECASE),
    re.compile(r"\b(on|in)\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2},\s+\d{4}\b", re.IGNORECASE),
    re.compile(r"\b(19|20)\d{2}\b"),
]


def should_skip_file(path: Path, blocked_keywords: tuple[str, ...]) -> bool:
    lower = str(path).lower().replace("\\", "/")
    return any(keyword in lower for keyword in blocked_keywords)


def line_is_probably_factual(line: str, factual_patterns: list[re.Pattern]) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if stripped.startswith(("http://", "https://")):
        return True
    if "[citation needed]" in stripped.lower():
        return True
    return any(pattern.search(stripped) for pattern in factual_patterns)
