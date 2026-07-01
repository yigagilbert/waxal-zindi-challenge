"""Text normalization policies for WAXAL scoring experiments."""

from __future__ import annotations

import re
import unicodedata

POLICIES = ("raw", "starter_lower", "no_punct_lower", "language_safe")

_MODEL_TOKEN_RE = re.compile(r"<\|[^|]+?\|>")
_BRACKET_ARTIFACT_RE = re.compile(
    r"\[(?:music|noise|silence|laughter|applause|inaudible|unk)\]",
    flags=re.IGNORECASE,
)


def normalize_whitespace(text: str) -> str:
    """Collapse all whitespace, including newlines, to single spaces."""
    return re.sub(r"\s+", " ", str(text).replace("\u00a0", " ")).strip()


def strip_unicode_punctuation(text: str) -> str:
    """Remove Unicode punctuation while preserving letters, digits, and spaces."""
    chars = []
    for char in text:
        if unicodedata.category(char).startswith("P"):
            chars.append(" ")
        else:
            chars.append(char)
    return "".join(chars)


def remove_control_artifacts(text: str) -> str:
    """Remove ASR/model artifacts that are not expected spoken text."""
    text = _MODEL_TOKEN_RE.sub(" ", text)
    text = _BRACKET_ARTIFACT_RE.sub(" ", text)
    text = text.replace("♪", " ").replace("♫", " ").replace("♬", " ")
    return "".join(
        " " if unicodedata.category(char).startswith("C") else char for char in text
    )


def normalize_text(text: str, policy: str = "raw") -> str:
    """Normalize text under a named policy."""
    if policy not in POLICIES:
        raise ValueError(f"Unknown normalization policy {policy!r}; expected {POLICIES}")

    text = unicodedata.normalize("NFC", str(text))
    if policy == "raw":
        return normalize_whitespace(text)

    if policy == "starter_lower":
        return normalize_whitespace(text).lower()

    if policy == "no_punct_lower":
        text = strip_unicode_punctuation(text.lower())
        return normalize_whitespace(text)

    if policy == "language_safe":
        text = remove_control_artifacts(text.lower())
        return normalize_whitespace(text)

    raise AssertionError("unreachable")

