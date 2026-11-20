"""Char n-gram TF-IDF + Logistic Regression baseline.

Character n-grams cope with Sinhala/Tamil morphology and the inconsistent
romanisation of Singlish/Tanglish, where word-level vocabularies break down.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from smishdet.data.normalize import normalize
from smishdet.domain.schema import Label


class TfidfClassifier:
    """Implements the `Classifier` protocol from `smishdet.models.base`."""

    def __init__(
        self,
        ngram_range: tuple[int, int] = (2, 5),
        analyzer: str = "char_wb",
        class_weight: str | None = "balanced",
        C: float = 10.0,
        seed: int = 42,
    ) -> None:
        self._pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        analyzer=analyzer,
                        ngram_range=ngram_range,
                        sublinear_tf=True,
                        min_df=2,
                        lowercase=True,
                        preprocessor=normalize,
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        C=C, class_weight=class_weight, max_iter=1000, random_state=seed
                    ),
                ),
            ]
        )

    @classmethod
    def from_config(cls, path: Path, seed: int = 42) -> TfidfClassifier:
        cfg: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls(
            ngram_range=tuple(cfg.get("ngram_range", (2, 5))),
            analyzer=cfg.get("analyzer", "char_wb"),
            class_weight=cfg.get("class_weight", "balanced"),
            C=float(cfg.get("C", 10.0)),
            seed=seed,
        )

    def fit(self, texts: list[str], labels: list[Label]) -> TfidfClassifier:
        self._pipeline.fit(texts, [lab.value for lab in labels])
        return self

    def predict(self, texts: list[str]) -> list[Label]:
        return [Label(v) for v in self._pipeline.predict(texts)]

    def predict_proba(self, texts: list[str]) -> list[dict[Label, float]]:
        probs = self._pipeline.predict_proba(texts)
        classes = [Label(c) for c in self._pipeline.classes_]
        return [dict(zip(classes, map(float, row), strict=True)) for row in probs]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._pipeline, path)

    @classmethod
    def load(cls, path: Path) -> TfidfClassifier:
        obj = cls()
        obj._pipeline = joblib.load(path)
        return obj
