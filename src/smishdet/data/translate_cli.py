"""CLI: python -m smishdet.data.translate_cli --n 600

Writes data/interim/translated_{si,ta}.csv (plus a review sheet) from the English seed.
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

from smishdet.data import translate as tr

SCRIPTS = {"si": "sinhala", "ta": "tamil"}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data/processed/english_seed.csv"))
    p.add_argument("--out", type=Path, default=Path("data/interim"))
    p.add_argument("--n", type=int, default=600, help="pilot sample size; 0 = all rows")
    p.add_argument("--langs", nargs="+", default=["si", "ta"], choices=list(SCRIPTS))
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--refilter", action="store_true",
                   help="re-apply quality filters to existing translated_*.csv (no model needed)")
    args = p.parse_args()

    rows = tr.read_rows(args.data)
    if args.refilter:
        english = {r["id"]: r for r in rows}
        for lang in args.langs:
            src = args.out / f"translated_{lang}.csv"
            stats: Counter[str] = Counter()
            before = tr.read_rows(src)
            kept = tr.refilter_rows(before, english, SCRIPTS[lang], stats)
            tr.write_rows(kept, args.out / f"translated_{lang}.filtered.csv")
            print(f"{lang}: {len(before)} -> {len(kept)} kept; dropped {dict(stats)}")
        return

    if args.n:
        rows = tr.stratified_sample(rows, args.n)
    print(f"translating {len(rows)} messages into {args.langs}")

    for lang in args.langs:
        t0 = time.time()
        fn = tr.translate_batch_fn(lang, batch_size=args.batch_size)
        stats = Counter()
        out = tr.translate_rows(rows, fn, lang, SCRIPTS[lang], stats)
        tr.write_rows(out, args.out / f"translated_{lang}.csv")
        print(f"{lang}: {len(out)} kept; dropped {dict(stats)} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
