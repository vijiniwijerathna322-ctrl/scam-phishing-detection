import pytest

from smishdet.domain.schema import Label
from smishdet.serving.predictor import MAX_CHARS, analyze, load_classifiers, preprocess


class FakeClassifier:
    """Records what it was given and returns a fixed distribution."""

    def __init__(self, probs):
        self.probs = probs
        self.seen: list[str] = []

    def predict_proba(self, texts):
        self.seen.extend(texts)
        return [self.probs for _ in texts]

    def predict(self, texts):
        return [max(self.probs, key=lambda lab: self.probs[lab]) for _ in texts]


def test_preprocess_matches_training_cleaning():
    out = preprocess("Pay​ now  http://bit.ly/x call 0771234567")
    assert out == "Pay now <URL> call <PHONE>"


def test_preprocess_truncates_long_input():
    assert len(preprocess("a" * (MAX_CHARS * 3))) == MAX_CHARS


def test_analyze_scrubs_before_predicting_and_picks_argmax():
    clf = FakeClassifier({Label.HAM: 0.05, Label.SPAM: 0.05, Label.SMISHING: 0.9})
    a = analyze("Verify at http://evil.example/login", clf)
    assert clf.seen == ["Verify at <URL>"]  # model never sees the raw link
    assert a.label == Label.SMISHING and a.confidence == 0.9 and not a.uncertain
    assert a.script == "latin"


def test_analyze_flags_low_confidence_and_detects_script():
    clf = FakeClassifier({Label.HAM: 0.4, Label.SPAM: 0.35, Label.SMISHING: 0.25})
    a = analyze("ඔබගේ ගිණුම අත්හිටුවා ඇත <URL>", clf)
    assert a.uncertain and a.script == "sinhala"


def test_analyze_rejects_empty_message():
    clf = FakeClassifier({Label.HAM: 1.0, Label.SPAM: 0.0, Label.SMISHING: 0.0})
    with pytest.raises(ValueError):
        analyze("   ​ ", clf)


def test_load_classifiers_requires_a_model(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_classifiers(tmp_path / "missing.joblib", tmp_path / "missing_dir")
