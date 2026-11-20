"""Unicode-range script detection for Sinhala / Tamil / Latin text."""

from __future__ import annotations

from smishdet.domain.schema import Script

_SINHALA = range(0x0D80, 0x0E00)
_TAMIL = range(0x0B80, 0x0C00)


def detect_script(text: str) -> Script:
    """Classify text by the scripts of its letters (digits/punctuation ignored)."""
    found: set[Script] = set()
    for ch in text:
        cp = ord(ch)
        if cp in _SINHALA:
            found.add(Script.SINHALA)
        elif cp in _TAMIL:
            found.add(Script.TAMIL)
        elif ch.isascii() and ch.isalpha():
            found.add(Script.LATIN)
    if len(found) == 1:
        return next(iter(found))
    return Script.MIXED if found else Script.LATIN
