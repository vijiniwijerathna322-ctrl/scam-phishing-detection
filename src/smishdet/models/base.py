"""Model interface. Serving and evaluation depend on this, not on concrete models."""

from __future__ import annotations

from typing import Protocol

from smishdet.domain.schema import Label


class Classifier(Protocol):
    def predict(self, texts: list[str]) -> list[Label]: ...

    def predict_proba(self, texts: list[str]) -> list[dict[Label, float]]: ...
