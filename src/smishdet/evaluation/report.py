"""Shared evaluation report used by every training CLI (baseline, XLM-R, ...)."""

from __future__ import annotations

from typing import Any

from smishdet.domain.schema import Label
from smishdet.evaluation.metrics import evaluate, evaluate_slices
from smishdet.models.base import Classifier

SLICES = ("language", "script", "source_type")


def evaluate_all(
    model: Classifier, splits: dict[str, list[dict[str, str]]]
) -> dict[str, dict[str, Any]]:
    """Score val, test and pooled held-out (val+test; per-language test slices alone are tiny).

    Each split is predicted once; held-out reuses those predictions.
    """
    preds: dict[str, list[Label]] = {
        name: model.predict([r["text"] for r in splits[name]]) for name in ("val", "test")
    }
    rows = {"val": splits["val"], "test": splits["test"],
            "heldout": splits["val"] + splits["test"]}
    preds["heldout"] = preds["val"] + preds["test"]

    results: dict[str, dict[str, Any]] = {}
    for name, rs in rows.items():
        y_true = [Label(r["label"]) for r in rs]
        res = evaluate(y_true, preds[name])
        for key in SLICES:
            res[f"by_{key}"] = evaluate_slices(y_true, preds[name], [r[key] for r in rs])
        results[name] = res
    return results


def print_results(results: dict[str, dict[str, Any]]) -> None:
    for name, res in results.items():
        print(f"{name}: n={res['n']} macro_f1={res['macro_f1']:.3f}")
        for lab, m in res["per_class"].items():
            print(f"  {lab:9s} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1-score']:.3f}")
        if name == "heldout":
            for lang, s in res["by_language"].items():
                flag = " (small n)" if s["unreliable_small_n"] else ""
                print(f"  lang {lang:6s} n={s['n']:4d} macro_f1={s['macro_f1']:.3f}{flag}")
