"""XLM-R sequence classifier with a plain PyTorch training loop.

Plain loop (no Trainer/accelerate) keeps dependencies small and behaviour explicit.
Heavy imports (torch, transformers) happen lazily so the rest of the package and the
CI tests do not need them.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from smishdet.data.normalize import normalize
from smishdet.domain.schema import Label

LABELS = [lab.value for lab in Label]  # fixed order: ham, spam, smishing


class XlmrClassifier:
    """Implements the `Classifier` protocol from `smishdet.models.base`."""

    def __init__(self, pretrained: str = "xlm-roberta-base", max_length: int = 128) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            pretrained,
            num_labels=len(LABELS),
            id2label=dict(enumerate(LABELS)),
            label2id={v: i for i, v in enumerate(LABELS)},
        )

    def _encode(self, texts: list[str]) -> Any:
        return self.tokenizer(
            [normalize(t) for t in texts],
            return_tensors="pt",
            padding=True,  # dynamic padding: only to the longest in the batch
            truncation=True,
            max_length=self.max_length,
        )

    def fit(
        self,
        texts: list[str],
        labels: list[Label],
        val: tuple[list[str], list[Label]] | None = None,
        epochs: int = 3,
        batch_size: int = 16,
        lr: float = 2e-5,
        weight_decay: float = 0.01,
        class_weighted: bool = True,
        seed: int = 42,
        best_dir: Path | None = None,
        log_every: int = 25,
        on_log: Callable[[str], None] = print,
    ) -> XlmrClassifier:
        import torch
        from sklearn.metrics import f1_score
        from transformers import get_linear_schedule_with_warmup

        torch.manual_seed(seed)
        rng = random.Random(seed)
        y = [LABELS.index(lab.value) for lab in labels]

        weights = None
        if class_weighted:
            counts = [max(1, y.count(i)) for i in range(len(LABELS))]
            weights = torch.tensor([len(y) / (len(LABELS) * c) for c in counts])
        loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        steps_per_epoch = (len(texts) + batch_size - 1) // batch_size
        total = steps_per_epoch * epochs
        sched = get_linear_schedule_with_warmup(opt, int(0.06 * total), total)

        best_f1, step, t0 = -1.0, 0, time.time()
        for epoch in range(1, epochs + 1):
            self.model.train()
            order = list(range(len(texts)))
            rng.shuffle(order)
            run_loss = 0.0
            for s in range(0, len(order), batch_size):
                ids = order[s : s + batch_size]
                enc = self._encode([texts[i] for i in ids])
                logits = self.model(**enc).logits
                loss = loss_fn(logits, torch.tensor([y[i] for i in ids]))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                step += 1
                run_loss += loss.item()
                if step % log_every == 0:
                    elapsed = time.time() - t0
                    eta = elapsed / step * (total - step)
                    on_log(f"epoch {epoch}/{epochs} step {step}/{total} "
                           f"loss {run_loss / log_every:.4f} elapsed {elapsed / 60:.1f}m "
                           f"eta {eta / 60:.1f}m")
                    run_loss = 0.0
            if val is not None:
                preds = self.predict(val[0])
                f1 = float(f1_score([v.value for v in val[1]], [p.value for p in preds],
                                    labels=LABELS, average="macro", zero_division=0))
                on_log(f"epoch {epoch} val macro_f1 {f1:.4f}")
                if f1 > best_f1 and best_dir is not None:
                    best_f1 = f1
                    self.save(best_dir)
                    on_log(f"  new best, saved to {best_dir}")
        if best_dir is not None and best_f1 >= 0:
            self.load_weights(best_dir)  # return the best-on-val checkpoint, not the last
        return self

    def predict_proba(self, texts: list[str], batch_size: int = 64) -> list[dict[Label, float]]:
        import torch

        self.model.eval()
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))  # less padding
        out: dict[int, dict[Label, float]] = {}
        for s in range(0, len(order), batch_size):
            ids = order[s : s + batch_size]
            with torch.inference_mode():
                probs = torch.softmax(self.model(**self._encode([texts[i] for i in ids])).logits,
                                      dim=-1)
            for i, row in zip(ids, probs.tolist(), strict=True):
                out[i] = {Label(lab): p for lab, p in zip(LABELS, row, strict=True)}
        return [out[i] for i in range(len(texts))]

    def predict(self, texts: list[str]) -> list[Label]:
        return [max(p, key=lambda lab: p[lab]) for p in self.predict_proba(texts)]

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load_weights(self, path: Path) -> None:
        from transformers import AutoModelForSequenceClassification

        self.model = AutoModelForSequenceClassification.from_pretrained(path)

    @classmethod
    def load(cls, path: Path, max_length: int = 128) -> XlmrClassifier:
        return cls(pretrained=str(path), max_length=max_length)
