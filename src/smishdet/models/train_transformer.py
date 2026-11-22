"""CLI: python -m smishdet.models.train_transformer --data data/processed/multilingual.csv

Fine-tunes XLM-R. The val split selects the best epoch; the test split is only scored
at the end. Writes best checkpoint, metrics.json and a training log to --out.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import yaml

from smishdet.domain.schema import Label
from smishdet.evaluation.report import evaluate_all, print_results
from smishdet.models.train_baseline import read_split
from smishdet.models.xlmr import XlmrClassifier


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data/processed/multilingual.csv"))
    p.add_argument("--config", type=Path, default=Path("configs/model/xlmr_base.yaml"))
    p.add_argument("--out", type=Path, default=Path("outputs/xlmr_multi_all"))
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-length", type=int, help="override max_length from the model config")
    p.add_argument("--train-languages", nargs="+")
    p.add_argument("--max-train", type=int, default=0, help="debug: random subset of train")
    p.add_argument("--max-eval", type=int, default=0, help="debug: random subset of val/test")
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    splits = read_split(args.data)
    rng = random.Random(args.seed)

    train = splits["train"]
    if args.train_languages:
        train = [r for r in train if r["language"] in args.train_languages]
    if args.max_train:
        train = rng.sample(train, min(args.max_train, len(train)))
    if args.max_eval:
        for name in ("val", "test"):
            splits[name] = rng.sample(splits[name], min(args.max_eval, len(splits[name])))

    args.out.mkdir(parents=True, exist_ok=True)
    log_file = (args.out / "train.log").open("a", encoding="utf-8")

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    log(f"training on {len(train)} rows, val {len(splits['val'])}, test {len(splits['test'])}")
    max_length = args.max_length or int(cfg.get("max_length", 128))
    log(f"max_length {max_length}, epochs {args.epochs}, batch {args.batch_size}, lr {args.lr}")
    model = XlmrClassifier(cfg["pretrained"], max_length=max_length)
    model.fit(
        [r["text"] for r in train],
        [Label(r["label"]) for r in train],
        val=([r["text"] for r in splits["val"]], [Label(r["label"]) for r in splits["val"]]),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        best_dir=args.out / "best",
        log_every=10,
        on_log=log,
    )

    results = evaluate_all(model, splits)
    print_results(results)
    payload = {"train_rows": len(train), "train_languages": args.train_languages or "all",
               "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
               "max_length": max_length, **results}
    (args.out / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log(f"saved to {args.out}")


if __name__ == "__main__":
    main()
