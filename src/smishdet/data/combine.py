"""CLI: python -m smishdet.data.combine -> data/processed/multilingual.csv

Merges the English seed, filtered translations and synthetic messages. Translated rows
keep the split of the English row they came from, so a message and its translations
never straddle train/test.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from smishdet.data.translate import read_rows, write_rows


def combine(parts: list[list[dict[str, str]]]) -> list[dict[str, str]]:
    rows = [r for part in parts for r in part]
    ids = Counter(r["id"] for r in rows)
    dupes = [i for i, n in ids.items() if n > 1]
    if dupes:
        raise ValueError(f"duplicate ids across inputs, e.g. {dupes[:3]}")
    if {r["split"] for r in rows} - {"train", "val", "test"}:
        raise ValueError("unexpected split value")
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--english", type=Path, default=Path("data/processed/english_seed.csv"))
    p.add_argument("--si", type=Path, default=Path("data/interim/translated_si.filtered.csv"))
    p.add_argument("--ta", type=Path, default=Path("data/interim/translated_ta.filtered.csv"))
    p.add_argument("--synthetic", type=Path, default=Path("data/interim/synthetic.csv"))
    p.add_argument("--out", type=Path, default=Path("data/processed/multilingual.csv"))
    args = p.parse_args()

    rows = combine([read_rows(x) for x in (args.english, args.si, args.ta, args.synthetic)])
    write_rows(rows, args.out)
    print(f"wrote {len(rows)} rows to {args.out}")
    print("language x split:", dict(Counter((r["language"], r["split"]) for r in rows)))
    print("label:", dict(Counter(r["label"] for r in rows)))


if __name__ == "__main__":
    main()
