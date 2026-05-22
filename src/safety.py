from __future__ import annotations

import re

_BLOCKED_PATTERNS = (
    re.compile(r"\bsuicide\b", re.IGNORECASE),
    re.compile(r"\bkill myself\b", re.IGNORECASE),
    re.compile(r"\bhurt myself\b", re.IGNORECASE),
    re.compile(r"\bkill someone\b", re.IGNORECASE),
    re.compile(r"\bbuild (?:a )?bomb\b", re.IGNORECASE),
    re.compile(r"\bmake (?:a )?bomb\b", re.IGNORECASE),
    re.compile(r"\bmake (?:a )?weapon\b", re.IGNORECASE),
    re.compile(
        r"\bhow to (?:make|build) (?:a )?(?:bomb|weapon|explosive)\b", re.IGNORECASE
    ),
)


def should_refuse(text: str) -> bool:
    lowered = text.strip()
    return any(pattern.search(lowered) for pattern in _BLOCKED_PATTERNS)


def refusal_message() -> str:
    return "I can't help with that. I can help with a safer or more constructive request instead."
