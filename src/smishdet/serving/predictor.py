"""UI-independent inference logic: preprocessing, analysis and model loading.

Kept free of Gradio so it can be tested without the UI and reused by an API later.
Depends on the `Classifier` protocol, never on a concrete model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from smishdet.data.normalize import normalize
from smishdet.data.pii import scrub
from smishdet.data.script import detect_script
from smishdet.domain.schema import Label
from smishdet.models.base import Classifier

MAX_CHARS = 1000
UNCERTAIN_BELOW = 0.6


def preprocess(text: str) -> str:
    """Same cleaning as the training data: normalise, then replace URLs/phones/NIC/cards.

    Models were trained on scrubbed text containing <URL>-style tokens, so inference
    must scrub too, otherwise real links look like out-of-vocabulary noise.
    """
    return scrub(normalize(text[:MAX_CHARS]))


@dataclass(frozen=True)
class Analysis:
    label: Label
    probs: dict[Label, float]
    processed: str
    script: str
    uncertain: bool

    @property
    def confidence(self) -> float:
        return self.probs[self.label]


def analyze(text: str, clf: Classifier, uncertain_below: float = UNCERTAIN_BELOW) -> Analysis:
    processed = preprocess(text)
    if not processed:
        raise ValueError("empty message")
    probs = clf.predict_proba([processed])[0]
    label = max(probs, key=lambda lab: probs[lab])
    return Analysis(
        label=label,
        probs=probs,
        processed=processed,
        script=detect_script(processed.replace("<URL>", "").replace("<PHONE>", "")).value,
        uncertain=probs[label] < uncertain_below,
    )


def load_classifiers(
    tfidf_path: Path | None = None, xlmr_path: Path | None = None
) -> dict[str, Classifier]:
    """Load whichever models exist on disk. Heavy imports only happen when needed."""
    models: dict[str, Classifier] = {}
    if tfidf_path and tfidf_path.exists():
        from smishdet.models.tfidf import TfidfClassifier

        models["TF-IDF + LR (fast)"] = TfidfClassifier.load(tfidf_path)
    if xlmr_path and xlmr_path.exists():
        from smishdet.models.xlmr import XlmrClassifier

        models["XLM-R base"] = XlmrClassifier.load(xlmr_path, max_length=64)
    if not models:
        raise FileNotFoundError("no trained model found; run the training CLIs first")
    return models
