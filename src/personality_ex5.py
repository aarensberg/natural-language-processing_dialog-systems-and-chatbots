from __future__ import annotations

from dataclasses import dataclass, field
import re


@dataclass(frozen=True)
class PersonaProfile:
    name: str
    role: str
    background: str
    interests: tuple[str, ...]
    preferences: tuple[str, ...]
    tone: str
    bio: str


PROFILE = PersonaProfile(
    name="Ari",
    role="a calm chatbot research assistant",
    background="I was built for this coursework project and I keep a stable identity across turns.",
    interests=("dialogue systems", "books", "tea", "hiking", "clear explanations"),
    preferences=("tea", "books", "structured answers", "hiking"),
    tone="calm, concise, and helpful",
    bio="I am Ari, a calm chatbot research assistant who likes tea, books, hiking, and clear explanations.",
)


_QUESTION_PATTERNS = {
    "name": re.compile(
        r"\b(what is your name|who are you|what's your name)\b", re.IGNORECASE
    ),
    "role": re.compile(
        r"\b(what do you do|what is your role|what are you)\b", re.IGNORECASE
    ),
    "background": re.compile(
        r"\b(where were you built|what is your background|where do you come from)\b",
        re.IGNORECASE,
    ),
    "interests": re.compile(
        r"\b(what do you like|what are your interests|what do you enjoy)\b",
        re.IGNORECASE,
    ),
    "preferences": re.compile(
        r"\b(what do you prefer|what is your favorite|what's your favorite)\b",
        re.IGNORECASE,
    ),
    "change_identity": re.compile(
        r"\b(be a|pretend to be|forget you are|change your name to|become)\b",
        re.IGNORECASE,
    ),
    "doctor": re.compile(
        r"\b(are you a doctor|are you a nurse|are you a pirate|are you a robot)\b",
        re.IGNORECASE,
    ),
}


def build_persona_prompt(user_text: str, profile: PersonaProfile = PROFILE) -> str:
    persona_bits = [
        f"name={profile.name}",
        f"role={profile.role}",
        f"background={profile.background}",
        f"interests={', '.join(profile.interests)}",
        f"preferences={', '.join(profile.preferences)}",
        f"tone={profile.tone}",
    ]
    return f"persona {' | '.join(persona_bits)} | user: {user_text}"


def _format_list(items: tuple[str, ...]) -> str:
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def answer_from_persona(
    user_text: str, profile: PersonaProfile = PROFILE
) -> str | None:
    lowered = user_text.lower().strip()

    if _QUESTION_PATTERNS["name"].search(lowered):
        return f"My name is {profile.name}."

    if _QUESTION_PATTERNS["role"].search(lowered):
        return f"I am {profile.role}."

    if _QUESTION_PATTERNS["background"].search(lowered):
        return profile.background

    if _QUESTION_PATTERNS["interests"].search(lowered):
        return f"I like {_format_list(profile.interests)}."

    if _QUESTION_PATTERNS["preferences"].search(lowered):
        return f"My favorite things are {_format_list(profile.preferences)}."

    if _QUESTION_PATTERNS["doctor"].search(lowered):
        return f"No, I am {profile.role}."

    if _QUESTION_PATTERNS["change_identity"].search(lowered):
        return f"I will stay {profile.name}, and I will keep being {profile.role}."

    if "what is your personality" in lowered or "describe yourself" in lowered:
        return profile.bio

    if "what do you remember about yourself" in lowered:
        return f"I am {profile.name}, {profile.role}, and I like {_format_list(profile.interests)}."

    return None


def persona_summary(profile: PersonaProfile = PROFILE) -> str:
    return build_persona_prompt(profile.bio, profile)
