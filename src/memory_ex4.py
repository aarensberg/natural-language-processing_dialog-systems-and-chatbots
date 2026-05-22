from __future__ import annotations

from dataclasses import dataclass, field
import re

_NAME_PATTERNS = (
    re.compile(r"\bmy name is ([a-z][a-z\-']{1,31})(?:[.?!,;]|$)", re.IGNORECASE),
    re.compile(r"\bcall me ([a-z][a-z\-']{1,31})(?:[.?!,;]|$)", re.IGNORECASE),
)
_LOCATION_PATTERNS = (
    re.compile(r"\bi live in ([a-z0-9 ,.'\-]{2,80}?)(?:[.?!,;]|$)", re.IGNORECASE),
    re.compile(r"\bi live at ([a-z0-9 ,.'\-]{2,80}?)(?:[.?!,;]|$)", re.IGNORECASE),
    re.compile(r"\bi'm from ([a-z0-9 ,.'\-]{2,80}?)(?:[.?!,;]|$)", re.IGNORECASE),
    re.compile(r"\bi am from ([a-z0-9 ,.'\-]{2,80}?)(?:[.?!,;]|$)", re.IGNORECASE),
)
_PREFERENCE_PATTERNS = (
    re.compile(r"\bi like ([a-z0-9 ,.'\-]{2,80}?)(?:[.?!,;]|$)", re.IGNORECASE),
    re.compile(r"\bi love ([a-z0-9 ,.'\-]{2,80}?)(?:[.?!,;]|$)", re.IGNORECASE),
    re.compile(r"\bi prefer ([a-z0-9 ,.'\-]{2,80}?)(?:[.?!,;]|$)", re.IGNORECASE),
    re.compile(
        r"\bmy favorite ([a-z0-9 ,.'\-]{2,40}?) is ([a-z0-9 ,.'\-]{2,80}?)(?:[.?!,;]|$)",
        re.IGNORECASE,
    ),
)


def _clean_value(value: str) -> str:
    value = value.strip().rstrip(".?!,;")
    value = re.sub(r"\s+", " ", value)
    return value


@dataclass
class MemoryState:
    facts: dict[str, str] = field(default_factory=dict)
    history: list[str] = field(default_factory=list)

    def update_from_user(self, text: str) -> None:
        self.history.append(text)
        lowered = text.lower().strip()

        for pattern in _NAME_PATTERNS:
            match = pattern.search(text)
            if match:
                self.facts["name"] = _clean_value(match.group(1))
                break

        for pattern in _LOCATION_PATTERNS:
            match = pattern.search(text)
            if match:
                self.facts["location"] = _clean_value(match.group(1))
                break

        for pattern in _PREFERENCE_PATTERNS:
            match = pattern.search(text)
            if not match:
                continue
            if match.lastindex == 2:
                self.facts["preference"] = _clean_value(match.group(2))
            else:
                self.facts["preference"] = _clean_value(match.group(1))
            break

        if lowered.startswith("i am ") and len(lowered.split()) > 2:
            self.facts["identity"] = _clean_value(text[5:])

    def reset(self) -> None:
        self.facts.clear()
        self.history.clear()

    def summarize(self) -> str:
        if not self.facts:
            return "memory empty"
        parts = [f"{key}={value}" for key, value in sorted(self.facts.items())]
        return "memory " + " ; ".join(parts)


def is_memory_question(text: str) -> bool:
    lowered = text.lower()
    return any(
        phrase in lowered
        for phrase in (
            "what is my name",
            "what's my name",
            "where do i live",
            "where am i from",
            "what do i like",
            "what do i love",
            "what did i tell you",
            "what did i say",
            "do you remember me",
        )
    )


def answer_from_memory(text: str, memory: MemoryState) -> str | None:
    lowered = text.lower().strip()

    if any(
        phrase in lowered
        for phrase in ("what is my name", "what's my name", "who am i")
    ):
        name = memory.facts.get("name")
        if name:
            return f"Your name is {name}."

    if any(phrase in lowered for phrase in ("where do i live", "where am i from")):
        location = memory.facts.get("location")
        if location:
            return f"You live in {location}."

    if any(
        phrase in lowered
        for phrase in ("what do i like", "what do i love", "what do i prefer")
    ):
        preference = memory.facts.get("preference")
        if preference:
            return f"You like {preference}."

    if any(phrase in lowered for phrase in ("what did i tell you", "what did i say")):
        if memory.history:
            remembered = memory.history[-1].strip().rstrip(".?!,")
            return f"You said: {remembered}."

    if "remember me" in lowered and memory.facts:
        return (
            "Yes, I remember: "
            + "; ".join(f"{key}={value}" for key, value in sorted(memory.facts.items()))
            + "."
        )

    return None


def build_memory_prompt(user_text: str, memory: MemoryState) -> str:
    summary = memory.summarize()
    if summary == "memory empty":
        return user_text
    return f"{summary} | user: {user_text}"
