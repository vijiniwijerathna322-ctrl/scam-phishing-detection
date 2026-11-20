"""Loaders for public English seed datasets, plus the cleaning pipeline.

Loaders read LOCAL files only; download them yourself and check each licence:
  - UCI SMS Spam Collection  -> data/raw/SMSSpamCollection        (tab: label<TAB>text)
  - Mendeley SMS Phishing    -> data/raw/mendeley_sms_phishing.csv (columns LABEL, TEXT, ...)
  - Smishtank                -> data/raw/smishtank.csv             (text column name varies)
"""

from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from smishdet.data.dedupe import cluster_near_duplicates
from smishdet.data.normalize import dedupe_key, normalize
from smishdet.data.pii import scrub
from smishdet.data.script import detect_script
from smishdet.data.split import group_split
from smishdet.domain.schema import Label, Language, Message, ScamType


@dataclass(frozen=True)
class RawRecord:
    text: str
    label: Label
    source: str


_LABELS = {"ham": Label.HAM, "legitimate": Label.HAM, "spam": Label.SPAM, "smishing": Label.SMISHING}


def _parse_label(value: str) -> Label | None:
    return _LABELS.get(value.strip().casefold())


def load_uci(path: Path) -> Iterator[RawRecord]:
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            label, _, text = line.rstrip("\n").partition("\t")
            parsed = _parse_label(label)
            if parsed and text.strip():
                yield RawRecord(text, parsed, "uci")


def load_mendeley(path: Path) -> Iterator[RawRecord]:
    with path.open(encoding="utf-8-sig", errors="replace", newline="") as f:
        for row in csv.DictReader(f):
            row = {k.strip().upper(): v for k, v in row.items() if k}
            parsed = _parse_label(row.get("LABEL", ""))
            text = row.get("TEXT", "")
            if parsed and text.strip():
                yield RawRecord(text, parsed, "mendeley")


def load_smishtank(path: Path, text_column: str = "text") -> Iterator[RawRecord]:
    """Smishtank is smishing-only: every row is labelled smishing."""
    with path.open(encoding="utf-8-sig", errors="replace", newline="") as f:
        for row in csv.DictReader(f):
            text = row.get(text_column, "")
            if text.strip():
                yield RawRecord(text, Label.SMISHING, "smishtank")


@dataclass
class BuildReport:
    read: int = 0
    exact_dupes_dropped: int = 0
    label_conflicts_dropped: int = 0
    label_conflicts_resolved: int = 0
    kept: int = 0
    near_dup_groups: int = 0


def build_english_seed(
    records: Iterator[RawRecord],
    threshold: float = 0.85,
    seed: int = 42,
    ratios: dict[str, float] | None = None,
) -> tuple[list[tuple[Message, str]], BuildReport]:
    """Normalise, scrub PII, drop exact dupes, cluster near-dupes, split by cluster."""
    report = BuildReport()
    by_key: dict[str, list[RawRecord]] = defaultdict(list)
    for r in records:
        report.read += 1
        by_key[dedupe_key(r.text)].append(r)

    unique: list[RawRecord] = []
    for group in by_key.values():
        report.exact_dupes_dropped += len(group) - 1
        labels = {r.label for r in group}
        if labels == {Label.SPAM, Label.SMISHING}:
            # Mendeley re-annotated UCI spam as smishing: take the more specific label
            unique.append(next(r for r in group if r.label == Label.SMISHING))
            report.label_conflicts_resolved += 1
        elif len(labels) > 1:
            # e.g. ham vs spam: ambiguous, drop for manual review
            report.label_conflicts_dropped += len(group)
            report.exact_dupes_dropped -= len(group) - 1
        else:
            unique.append(group[0])

    texts = [scrub(normalize(r.text)) for r in unique]
    clusters = cluster_near_duplicates(texts, threshold=threshold, seed=seed)
    splits = group_split(clusters, ratios, seed)

    out: list[tuple[Message, str]] = []
    for i, (r, text, cl, split) in enumerate(zip(unique, texts, clusters, splits, strict=True)):
        msg = Message(
            id=f"en-{i:06d}",
            text=text,
            label=r.label,
            language=Language.ENGLISH,
            script=detect_script(text),
            source=r.source,
            scam_type=ScamType.NONE,
            template_group=f"c{cl}",
        )
        out.append((msg, split))
    report.kept = len(out)
    report.near_dup_groups = len(set(clusters))
    return out, report


def write_csv(rows: list[tuple[Message, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["id", "text", "label", "language", "script", "source", "scam_type",
            "template_group", "split"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for m, split in rows:
            w.writerow([m.id, m.text, m.label.value, m.language.value, m.script.value,
                        m.source, m.scam_type.value, m.template_group, split])
