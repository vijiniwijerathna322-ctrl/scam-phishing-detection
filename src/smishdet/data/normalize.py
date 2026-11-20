"""Text normalisation applied identically at training and inference time."""

from __future__ import annotations

import re
import unicodedata

# Zero-width chars are used both by Sinhala/Tamil input methods and by scammers
# to evade filters. ZWJ (U+200D) is meaningful in Sinhala conjuncts, so it is kept.
_STRIP = dict.fromkeys(map(ord, "​‌⁠﻿"))
_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text).translate(_STRIP)
    return _WS.sub(" ", text).strip()


def dedupe_key(text: str) -> str:
    """Aggressive key for exact-duplicate detection (case/punctuation-insensitive)."""
    t = normalize(text).casefold()
    return re.sub(r"[\W_]+", " ", t).strip()
