"""PII scrubbing. Runs before any message is stored or shared."""

from __future__ import annotations

import re

_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"https?://\S+|www\.\S+"), "<URL>"),
    (re.compile(r"(?:\+94|0)\s?\d{2}\s?\d{3}\s?\d{4}"), "<PHONE>"),
    (re.compile(r"\b(?:\d{9}[VvXx]|\d{12})\b"), "<NIC>"),
    (re.compile(r"\b\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}\b"), "<CARD>"),
]


def scrub(text: str) -> str:
    for pattern, token in _PATTERNS:
        text = pattern.sub(token, text)
    return text
