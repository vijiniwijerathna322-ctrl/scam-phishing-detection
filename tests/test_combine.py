import pytest

pytest.importorskip("sklearn")

from smishdet.data.combine import combine  # noqa: E402
from smishdet.models.train_baseline import source_type  # noqa: E402


def _r(i, split="train"):
    return {"id": i, "split": split}


def test_combine_concatenates_and_rejects_duplicate_ids():
    assert len(combine([[_r("a")], [_r("b")]])) == 2
    with pytest.raises(ValueError, match="duplicate"):
        combine([[_r("a")], [_r("a")]])


def test_combine_rejects_bad_split():
    with pytest.raises(ValueError, match="split"):
        combine([[_r("a", "dev")]])


def test_source_type():
    assert source_type("uci") == "public"
    assert source_type("mendeley") == "public"
    assert source_type("translated:uci") == "translated"
    assert source_type("synthetic") == "synthetic"
