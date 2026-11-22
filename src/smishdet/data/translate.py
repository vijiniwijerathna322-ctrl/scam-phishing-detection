"""Machine-translate the English seed into Sinhala and Tamil with NLLB-200.

Placeholders produced by PII scrubbing (<URL>, <PHONE>, ...) are masked before
translation and restored afterwards, because MT models mangle angle-bracket tokens.
Output rows are marked source="translated:<orig source>" so they can be evaluated
separately from real messages and sent for human review (see docs/annotation).

NLLB-200 is CC-BY-NC 4.0: translated data inherits a non-commercial restriction.
"""

from __future__ import annotations

import csv
import re
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path

NLLB_MODEL = "facebook/nllb-200-distilled-600M"
LANG_CODES = {"si": "sin_Sinh", "ta": "tam_Taml"}

_PLACEHOLDER = re.compile(r"<(URL|PHONE|NIC|CARD)>")

def mask_placeholders(text: str) -> tuple[str, list[str]]:
    """Replace <URL>-style tokens with numbered markers like [1], returning the originals."""
    found: list[str] = []

    def repl(m: re.Match[str]) -> str:
        found.append(m.group(0))
        return f"[{len(found)}]"

    return _PLACEHOLDER.sub(repl, text), found


def unmask_placeholders(text: str, originals: Sequence[str]) -> tuple[str, bool]:
    """Restore tokens. Returns (text, ok); ok is False if any marker was lost in translation."""
    ok = True
    for i, orig in enumerate(originals, start=1):
        marker = f"[{i}]"
        if marker in text:
            text = text.replace(marker, orig, 1)
        else:
            ok = False
    return text, ok


def translate_batch_fn(target: str, batch_size: int = 16, max_length: int = 200) -> Callable[
    [list[str]], list[str]
]:
    """Build a translate(list[str]) -> list[str] function backed by NLLB (lazy heavy imports)."""
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(NLLB_MODEL, src_lang="eng_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL).eval()
    forced = tok.convert_tokens_to_ids(LANG_CODES[target])

    def run(texts: list[str]) -> list[str]:
        # sort by length so batches have similar padding, then restore order
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        result: dict[int, str] = {}
        for s in range(0, len(order), batch_size):
            ids = order[s : s + batch_size]
            enc = tok([texts[i] for i in ids], return_tensors="pt", padding=True,
                      truncation=True, max_length=max_length)
            with torch.inference_mode():
                gen = model.generate(**enc, forced_bos_token_id=forced, num_beams=2,
                                     max_new_tokens=max_length)
            for i, t in zip(ids, tok.batch_decode(gen, skip_special_tokens=True), strict=True):
                result[i] = t
        return [result[i] for i in range(len(texts))]

    return run


_SCRIPT_RANGES = {"sinhala": range(0x0D80, 0x0E00), "tamil": range(0x0B80, 0x0C00)}
_NUMBER = re.compile(r"\d{3,}")

MIN_SCRIPT_FRACTION = 0.5
MIN_WORD_RATIO = 0.4
MIN_WORDS_FOR_RATIO = 8


def script_fraction(text: str, script: str) -> float:
    """Share of letters (placeholders excluded) that belong to the target script."""
    letters = [c for c in _PLACEHOLDER.sub("", text) if c.isalpha()]
    if not letters:
        return 0.0
    rng = _SCRIPT_RANGES[script]
    return sum(ord(c) in rng for c in letters) / len(letters)


def check_translation(source: str, output: str, script: str) -> str | None:
    """Return a drop reason, or None if the translation looks usable.

    Checks target machine-translation failure modes seen in the pilot:
    - untranslated output (left in English / noisy SMS text);
    - truncation, caught by a length ratio and by missing numbers (short codes, prices,
      phone fragments), because dropping those removes the scam's call-to-action.
    """
    if not output.strip():
        return "empty"
    if script_fraction(output, script) < MIN_SCRIPT_FRACTION:
        return "wrong_script"
    src_words = len(source.split())
    if src_words >= MIN_WORDS_FOR_RATIO and len(output.split()) / src_words < MIN_WORD_RATIO:
        return "too_short"
    if any(n not in output for n in _NUMBER.findall(source)):
        return "number_lost"
    return None


def translate_rows(
    rows: list[dict[str, str]],
    translate: Callable[[list[str]], list[str]],
    lang: str,
    script: str,
    stats: Counter[str] | None = None,
) -> list[dict[str, str]]:
    """Translate message rows, dropping failed ones (reasons are counted in `stats`)."""
    stats = stats if stats is not None else Counter()
    masked = [mask_placeholders(r["text"]) for r in rows]
    translated = translate([m for m, _ in masked])
    out: list[dict[str, str]] = []
    for r, (_, originals), t in zip(rows, masked, translated, strict=True):
        text, ok = unmask_placeholders(t.strip(), originals)
        reason = "placeholder_lost" if not ok else check_translation(r["text"], text, script)
        if reason:
            stats[reason] += 1
            continue
        out.append({
            **r,
            "id": f"{lang}-{r['id']}",
            "text": text,
            "language": lang,
            "script": script,
            "source": f"translated:{r['source']}",
            # translated rows keep the English row's group and split: no cross-lingual leakage
        })
    return out


def stratified_sample(
    rows: list[dict[str, str]], n: int, max_chars: int = 300, seed: int = 42
) -> list[dict[str, str]]:
    """Pick ~n short rows with an equal quota per label, so rare classes are well represented."""
    import random

    rng = random.Random(seed)
    pool = [r for r in rows if len(r["text"]) <= max_chars]
    by_label: dict[str, list[dict[str, str]]] = {}
    for r in pool:
        by_label.setdefault(r["label"], []).append(r)
    for v in by_label.values():
        rng.shuffle(v)
    # equal-ish quota per label so rare classes (spam/smishing) are well represented
    quota = n // len(by_label)
    picked: list[dict[str, str]] = []
    for v in by_label.values():
        picked.extend(v[:quota])
    return picked


def refilter_rows(
    translated: list[dict[str, str]],
    english_by_id: dict[str, dict[str, str]],
    script: str,
    stats: Counter[str] | None = None,
) -> list[dict[str, str]]:
    """Re-apply quality checks to already-translated rows (no re-translation needed)."""
    stats = stats if stats is not None else Counter()
    kept: list[dict[str, str]] = []
    for r in translated:
        source = english_by_id[r["id"].split("-", 1)[1]]["text"]
        reason = check_translation(source, r["text"], script)
        if reason:
            stats[reason] += 1
        else:
            kept.append(r)
    return kept


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
