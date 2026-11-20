import pytest

pytest.importorskip("sklearn")

from smishdet.domain.schema import Label  # noqa: E402
from smishdet.evaluation.metrics import evaluate, evaluate_slices  # noqa: E402
from smishdet.models.tfidf import TfidfClassifier  # noqa: E402

TEXTS = (
    ["Your account is suspended verify now at <URL>", "Claim your free prize <URL> now"] * 6
    + ["See you at the cafe tomorrow", "Can you send me the notes please"] * 6
)
LABELS = [Label.SMISHING, Label.SMISHING] * 6 + [Label.HAM, Label.HAM] * 6


def test_fit_predict_and_proba():
    model = TfidfClassifier().fit(TEXTS, LABELS)
    preds = model.predict(["Verify your account now <URL>", "See you tomorrow at the cafe"])
    assert preds == [Label.SMISHING, Label.HAM]
    proba = model.predict_proba(["Verify your account now <URL>"])[0]
    assert abs(sum(proba.values()) - 1.0) < 1e-6


def test_handles_sinhala_and_tamil_text():
    model = TfidfClassifier().fit(["ඔබගේ ගිණුම අත්හිටුවා ඇත"] * 3 + ["வணக்கம் நண்பா"] * 3,
                                  [Label.SMISHING] * 3 + [Label.HAM] * 3)
    assert model.predict(["ඔබගේ ගිණුම අත්හිටුවා ඇත"]) == [Label.SMISHING]


def test_save_load_roundtrip(tmp_path):
    model = TfidfClassifier().fit(TEXTS, LABELS)
    model.save(tmp_path / "m.joblib")
    loaded = TfidfClassifier.load(tmp_path / "m.joblib")
    assert loaded.predict(TEXTS[:4]) == model.predict(TEXTS[:4])


def test_evaluate_and_slices():
    y = [Label.HAM, Label.SPAM, Label.SMISHING, Label.HAM]
    assert evaluate(y, y)["macro_f1"] == 1.0
    out = evaluate_slices(y, y, ["en", "en", "si", "si"], min_n=3)
    assert set(out) == {"en", "si"} and out["en"]["unreliable_small_n"] is True
