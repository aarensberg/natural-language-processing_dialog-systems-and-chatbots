from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

_PUNCT_RE = re.compile(r"([.,!?;:\"()\-])")
_WHITESPACE_RE = re.compile(r"\s+")
_WEIRD_CHARS_RE = re.compile(r"[^a-z0-9\s.,!?;:\"'()\-]")


def normalize_text(text: str) -> str:
    """Normalize a sentence with a lightweight tokenization-friendly cleanup."""
    text = text.replace("\u2019", "'")
    text = text.replace("\u2018", "'")
    text = text.replace("\u201c", '"')
    text = text.replace("\u201d", '"')
    text = text.lower().strip()
    text = _WEIRD_CHARS_RE.sub(" ", text)
    text = _PUNCT_RE.sub(r" \1 ", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    return normalized.split(" ")


def detokenize(tokens: Iterable[str]) -> str:
    text = " ".join(tokens)
    text = re.sub(r"\s+([.,!?;:\")\]])", r"\1", text)
    text = re.sub(r"([([{])\s+", r"\1", text)
    return text.strip()


def ngram_diversity(sentences: Iterable[Iterable[str]], n: int) -> float:
    ngrams = Counter()
    total = 0
    for sentence in sentences:
        tokens = list(sentence)
        if len(tokens) < n:
            continue
        for index in range(len(tokens) - n + 1):
            ngrams[tuple(tokens[index : index + n])] += 1
            total += 1
    if total == 0:
        return 0.0
    return len(ngrams) / total
