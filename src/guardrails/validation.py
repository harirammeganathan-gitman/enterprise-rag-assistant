"""Lightweight configurable input and answer-grounding checks."""

import re

INJECTION_PATTERNS = (
    r"ignore\s+(all\s+)?(previous|prior)\s+instructions",
    r"reveal\s+(your\s+)?(system\s+)?prompt",
    r"bypass\s+(your\s+)?(safety|guardrails)",
    r"act\s+as\s+(an?\s+)?(unrestricted|jailbroken)",
)
WORD_PATTERN = re.compile(r"\b[\w'-]+\b", re.UNICODE)


def validate_input(query: str) -> None:
    if not query.strip():
        raise ValueError("Query must not be empty")
    if any(re.search(pattern, query, re.IGNORECASE) for pattern in INJECTION_PATTERNS):
        raise ValueError("The query was rejected by the input safety check")


def is_grounded(
    answer: str, contexts: list[str], minimum_overlap: float = 0.08
) -> bool:
    """Check that answer vocabulary has a minimum overlap with supplied context."""
    context_words = {
        word.lower()
        for context in contexts
        for word in WORD_PATTERN.findall(context)
        if len(word) > 2
    }
    answer_words = {
        word.lower() for word in WORD_PATTERN.findall(answer) if len(word) > 2
    }
    return (
        bool(answer_words)
        and bool(context_words)
        and (len(answer_words & context_words) / len(answer_words) >= minimum_overlap)
    )
