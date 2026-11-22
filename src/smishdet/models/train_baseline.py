"""CLI: python -m smishdet.models.train_baseline

Trains on the `train` split, tunes nothing on `test`, and writes the model plus
metrics (overall and per slice) to outputs/tfidf_lr/.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from smishdet.domain.schema import Label
from smishdet.evaluation.metrics import evaluate, evaluate_slices
from smishdet.models.tfidf import TfidfClassifier


def source_type(source: str) -> str:
    """Collapse source names into public / translated / synthetic for slicing."""
    if source.startswith("translated"):
        return "translated"
    return "synthetic" if source == "synthetic" else "public"


def read_split(path: Path) -> dict[str, list[dict[str, str]]]:
    splits: dict[str, list[dict[str, str]]] = {"train": [], "val": [], "test": []}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            row["source_type"] = source_type(row["source"])
            splits[row["split"]].append(row)
    return splits


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data/processed/english_seed.csv"))
    p.add_argument("--config", type=Path, default=Path("configs/model/tfidf_lr.yaml"))
    p.add_argument("--out", type=Path, default=Path("outputs/tfidf_lr"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-languages", nargs="+",
                   help="train only on these languages (e.g. en) to test cross-lingual transfer")
    args = p.parse_args()

    splits = read_split(args.data)
    model = TfidfClassifier.from_config(args.config, seed=args.seed)
    train = splits["train"]
    if args.train_languages:
        train = [r for r in train if r["language"] in args.train_languages]
    print(f"training on {len(train)} rows")
    model.fit([r["text"] for r in train], [Label(r["label"]) for r in train])

    results: dict[str, object] = {"train_rows": len(train),
                                  "train_languages": args.train_languages or "all"}
    # "heldout" pools val+test: per-language test slices alone are tiny and noisy
    evals = {"val": splits["val"], "test": splits["test"],
             "heldout": splits["val"] + splits["test"]}
    for name, rows in evals.items():
        y_true = [Label(r["label"]) for r in rows]
        y_pred = model.predict([r["text"] for r in rows])
        res = evaluate(y_true, y_pred)
        for key in ("language", "script", "source_type"):
            res[f"by_{key}"] = evaluate_slices(y_true, y_pred, [r[key] for r in rows])
        results[name] = res
        print(f"{name}: n={res['n']} macro_f1={res['macro_f1']:.3f}")
        for lab, m in res["per_class"].items():
            print(f"  {lab:9s} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1-score']:.3f}")
        if name == "heldout":
            for lang, s in res["by_language"].items():
                flag = " (small n)" if s["unreliable_small_n"] else ""
                print(f"  lang {lang:6s} n={s['n']:4d} macro_f1={s['macro_f1']:.3f}{flag}")

    args.out.mkdir(parents=True, exist_ok=True)
    model.save(args.out / "model.joblib")
    (args.out / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
