"""Evaluation metrics, including per-slice breakdowns (language, script, source...)."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from sklearn.metrics import classification_report, f1_score

from smishdet.domain.schema import Label


def evaluate(y_true: Sequence[Label], y_pred: Sequence[Label]) -> dict[str, Any]:
    t = [x.value for x in y_true]
    p = [x.value for x in y_pred]
    labels = [lab.value for lab in Label]
    report = classification_report(t, p, labels=labels, output_dict=True, zero_division=0)
    return {
        "n": len(t),
        "macro_f1": float(f1_score(t, p, labels=labels, average="macro", zero_division=0)),
        "per_class": {k: report[k] for k in labels},
    }


def evaluate_slices(
    y_true: Sequence[Label],
    y_pred: Sequence[Label],
    slice_values: Sequence[str],
    min_n: int = 20,
) -> dict[str, dict[str, Any]]:
    """Macro-F1 per slice value. Small slices are reported but flagged as unreliable."""
    idx: dict[str, list[int]] = defaultdict(list)
    for i, v in enumerate(slice_values):
        idx[v].append(i)
    out: dict[str, dict[str, Any]] = {}
    for v, ids in sorted(idx.items()):
        res = evaluate([y_true[i] for i in ids], [y_pred[i] for i in ids])
        res["unreliable_small_n"] = len(ids) < min_n
        out[v] = res
    return out
