import csv
import json

import pytest

from smishdet.release import USER_PLACEHOLDER, dataset_card, stage_release, tfidf_card, xlmr_card

COLS = ["id", "text", "label", "language", "script", "source", "scam_type",
        "template_group", "split"]
METRICS = {
    "epochs": 2, "max_length": 64, "train_rows": 100,
    "val": {"macro_f1": 0.906}, "test": {"macro_f1": 0.892}, "heldout": {"macro_f1": 0.902},
}


def _write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)


def _row(i, split, source="synthetic"):
    return dict(id=f"syn-{i}", text="hi <URL>", label="ham", language="en", script="latin",
                source=source, scam_type="none", template_group=f"g{i}", split=split)


@pytest.fixture
def artifacts(tmp_path):
    syn = tmp_path / "synthetic.csv"
    _write_csv(syn, [_row(1, "train"), _row(2, "val"), _row(3, "test")])
    tfidf = tmp_path / "tfidf"
    tfidf.mkdir()
    (tfidf / "model.joblib").write_bytes(b"x")
    (tfidf / "metrics.json").write_text(json.dumps(METRICS), encoding="utf-8")
    run = tmp_path / "xlmr"
    best = run / "best"
    best.mkdir(parents=True)
    (best / "model.safetensors").write_bytes(b"w")
    (run / "metrics.json").write_text(json.dumps(METRICS), encoding="utf-8")
    return syn, tfidf, best, tmp_path / "release"


def test_stage_builds_three_repos_with_split_files(artifacts):
    syn, tfidf, best, out = artifacts
    paths = stage_release(syn, tfidf, best, out)
    assert (paths["dataset"] / "data" / "train.csv").exists()
    assert (paths["dataset"] / "README.md").exists()
    assert (paths["tfidf"] / "model.joblib").exists()
    assert (paths["xlmr"] / "model.safetensors").exists()
    # the training run's metrics.json must not leak into the model repo as a stray file
    assert not (paths["xlmr"] / "metrics.json").exists()


def test_stage_refuses_non_synthetic_rows(artifacts):
    syn, tfidf, best, out = artifacts
    _write_csv(syn, [_row(1, "train", source="uci")])
    with pytest.raises(ValueError, match="synthetic"):
        stage_release(syn, tfidf, best, out)


def test_cards_use_real_numbers_and_state_caveats():
    xc = xlmr_card(METRICS)
    assert "0.892" in xc and "0.906" in xc and "cc-by-nc-4.0" in xc
    assert "not a security product" in xc.lower()
    assert "pickle" in tfidf_card(METRICS).lower()
    dc = dataset_card([_row(1, "train")])
    assert "Not reviewed by native speakers" in dc and "cc-by-4.0" in dc
    assert USER_PLACEHOLDER in xc  # replaced only at upload time
