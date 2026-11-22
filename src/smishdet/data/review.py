"""Native-speaker review: build a sheet for reviewers, then import their answers.

  python -m smishdet.data.review build   -> data/interim/review_sheet.csv
  python -m smishdet.data.review apply   -> data/processed/reviewed.csv + quality stats

Design choices
- The sheet samples only val/test rows. Reviewed rows therefore form a clean gold
  evaluation set that no model has trained on.
- Sampling is round-robin over template groups and balanced over labels, so the sheet
  covers many different messages instead of many variants of one template.
- Written as UTF-8 with BOM so Excel shows Sinhala/Tamil correctly for non-technical reviewers.
- The reviewer sees the proposed label (needed to check it) so answers may be anchored;
  the guideline tells reviewers to judge the message first.
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from smishdet.data.synthetic import LANGS
from smishdet.data.translate import read_rows

REVIEWER_COLS = ["fluency_1to5", "meaning_kept_yes_no", "label_check", "corrected_text", "notes"]
SHEET_COLS = ["id", "language", "label_proposed", "text", "english_reference", *REVIEWER_COLS]
VALID_LABELS = {"ham", "spam", "smishing"}


def english_reference_id(row_id: str) -> str | None:
    """Id of the English message a translated/synthetic row corresponds to."""
    if row_id.startswith("syn-"):
        rest = row_id[4:]
        for lang in sorted(LANGS, key=len, reverse=True):
            if rest.startswith(lang + "-"):
                return "syn-en-" + rest[len(lang) + 1 :]
        return None
    if "-en-" in row_id and row_id.split("-", 1)[1].startswith("en-"):
        return row_id.split("-", 1)[1]  # "si-en-000123" -> "en-000123"
    return None


def sample_language(
    rows: list[dict[str, str]], quota: int, seed: int
) -> list[dict[str, str]]:
    """Balance labels, then round-robin over template groups within each label."""
    rng = random.Random(seed)
    by_label: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_label[r["label"]][r["template_group"]].append(r)
    for groups in by_label.values():
        for items in groups.values():
            rng.shuffle(items)

    per_label: dict[str, list[dict[str, str]]] = {lab: [] for lab in by_label}
    for lab, groups in by_label.items():
        order = list(groups)
        rng.shuffle(order)
        while any(groups[g] for g in order):
            for g in order:
                if groups[g]:
                    per_label[lab].append(groups[g].pop())

    picked: list[dict[str, str]] = []
    labels = sorted(per_label)
    share = max(1, quota // max(1, len(labels)))
    for lab in labels:
        picked.extend(per_label[lab][:share])
    leftovers = [r for lab in labels for r in per_label[lab][share:]]
    rng.shuffle(leftovers)
    picked.extend(leftovers[: max(0, quota - len(picked))])  # top up if a label ran short
    return picked[:quota]


def build_sheet(
    rows: list[dict[str, str]], quotas: dict[str, int], seed: int = 42
) -> list[dict[str, str]]:
    english = {r["id"]: r["text"] for r in rows}
    held_out = [r for r in rows if r["split"] in ("val", "test")]
    out: list[dict[str, str]] = []
    for lang, quota in quotas.items():
        pool = [r for r in held_out if r["language"] == lang]
        for r in sample_language(pool, quota, seed):
            ref_id = english_reference_id(r["id"])
            out.append({
                "id": r["id"], "language": lang, "label_proposed": r["label"],
                "text": r["text"], "english_reference": english.get(ref_id or "", ""),
                **dict.fromkeys(REVIEWER_COLS, ""),
            })
    random.Random(seed).shuffle(out)  # mix languages so reviewers aren't primed by order
    return out


def apply_review(
    all_rows: list[dict[str, str]], reviewed: list[dict[str, str]]
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Turn answered sheet rows into a gold set plus quality statistics."""
    by_id = {r["id"]: r for r in all_rows}
    gold: list[dict[str, str]] = []
    stats: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    skipped: Counter[str] = Counter()
    for rv in reviewed:
        orig = by_id.get(rv["id"])
        fluency = rv.get("fluency_1to5", "").strip()
        if orig is None:
            skipped["unknown_id"] += 1
            continue
        if not fluency and not rv.get("label_check", "").strip():
            skipped["unanswered"] += 1
            continue
        s = stats[orig["language"]]
        if fluency.isdigit():
            s["fluency"].append(float(fluency))
        meaning = rv.get("meaning_kept_yes_no", "").strip().lower()
        if meaning in ("yes", "no"):
            s["meaning_kept"].append(1.0 if meaning == "yes" else 0.0)
        check = rv.get("label_check", "").strip().lower()
        if check in ("ok", *VALID_LABELS):
            s["label_agree"].append(1.0 if check in ("ok", orig["label"]) else 0.0)

        corrected = rv.get("corrected_text", "").strip()
        label = orig["label"]
        if check in VALID_LABELS:
            label = check
        elif check == "unsure":
            skipped["label_unsure"] += 1
            continue
        unusable = (fluency.isdigit() and int(fluency) <= 2) or meaning == "no"
        if unusable and not corrected:
            skipped["rejected_bad_translation"] += 1
            continue
        gold.append({**orig, "text": corrected or orig["text"], "label": label,
                     "source": "reviewed:" + orig["source"]})

    by_language: dict[str, dict[str, float]] = {}
    for lang, m in stats.items():
        by_language[lang] = {k: round(sum(v) / len(v), 3) for k, v in m.items() if v}
        by_language[lang]["n"] = len(m["fluency"])
    summary: dict[str, Any] = {"kept": len(gold), "skipped": dict(skipped),
                               "by_language": by_language}
    return gold, summary


def write_sheet(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:  # BOM for Excel
        w = csv.DictWriter(f, fieldnames=SHEET_COLS)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--data", type=Path, default=Path("data/processed/multilingual.csv"))
    b.add_argument("--out", type=Path, default=Path("data/interim/review_sheet.csv"))
    for lang, default in (("si", 50), ("ta", 50), ("si-en", 20), ("ta-en", 20)):
        b.add_argument(f"--{lang}", type=int, default=default)
    a = sub.add_parser("apply")
    a.add_argument("--data", type=Path, default=Path("data/processed/multilingual.csv"))
    a.add_argument("--sheet", type=Path, default=Path("data/interim/review_sheet.csv"))
    a.add_argument("--out", type=Path, default=Path("data/processed/reviewed.csv"))
    args = p.parse_args()

    rows = read_rows(args.data)
    if args.cmd == "build":
        quotas = {"si": args.si, "ta": args.ta, "si-en": args.si_en,
                  "ta-en": args.ta_en}
        sheet = build_sheet(rows, quotas)
        write_sheet(sheet, args.out)
        print(f"wrote {len(sheet)} rows to {args.out}")
        print("languages:", dict(Counter(r["language"] for r in sheet)))
        print("labels:", dict(Counter(r["label_proposed"] for r in sheet)))
    else:
        with args.sheet.open(encoding="utf-8-sig", newline="") as f:
            reviewed = list(csv.DictReader(f))
        gold, summary = apply_review(rows, reviewed)
        if gold:
            with args.out.open("w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(gold[0]))
                w.writeheader()
                w.writerows(gold)
        print(summary)


if __name__ == "__main__":
    main()
