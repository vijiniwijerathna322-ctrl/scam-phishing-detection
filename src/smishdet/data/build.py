"""CLI: python -m smishdet.data.build  -> data/processed/english_seed.csv"""

from __future__ import annotations

import argparse
import itertools
from collections import Counter
from pathlib import Path

from smishdet.data import ingest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", type=Path, default=Path("data/raw"))
    p.add_argument("--out", type=Path, default=Path("data/processed/english_seed.csv"))
    p.add_argument("--threshold", type=float, default=0.85)
    args = p.parse_args()

    sources = []
    for name, loader in [
        ("SMSSpamCollection", ingest.load_uci),
        ("mendeley_sms_phishing.csv", ingest.load_mendeley),
        ("smishtank.csv", ingest.load_smishtank),
    ]:
        path = args.raw / name
        if path.exists():
            sources.append(loader(path))
        else:
            print(f"skip (not found): {path}")
    if not sources:
        raise SystemExit("No raw files found. See smishdet/data/ingest.py docstring.")

    rows, report = ingest.build_english_seed(itertools.chain(*sources), args.threshold)
    ingest.write_csv(rows, args.out)

    print(report)
    print("labels:", dict(Counter(m.label.value for m, _ in rows)))
    print("splits:", dict(Counter(s for _, s in rows)))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
