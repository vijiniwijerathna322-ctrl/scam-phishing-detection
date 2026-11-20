from smishdet.data.dedupe import cluster_near_duplicates
from smishdet.data.ingest import RawRecord, build_english_seed, load_mendeley, load_uci
from smishdet.data.normalize import normalize
from smishdet.data.split import group_split
from smishdet.domain.schema import Label


def test_normalize_strips_zero_width_and_whitespace():
    assert normalize("a​  b\n c") == "a b c"


def test_near_duplicates_cluster_together():
    texts = [
        "Your account is suspended. Verify now at http://x.co/aaa",
        "Your account is suspended. Verify now at http://x.co/aab",
        "Meet me at the cafe at five tomorrow evening",
    ]
    c = cluster_near_duplicates(texts, threshold=0.7)
    assert c[0] == c[1] != c[2]


def test_group_split_keeps_groups_whole():
    groups = [i // 4 for i in range(100)]
    splits = group_split(groups, seed=1)
    seen: dict[int, str] = {}
    for g, s in zip(groups, splits, strict=True):
        assert seen.setdefault(g, s) == s
    assert {"train", "val", "test"} <= set(splits)


def test_build_drops_exact_dupes_and_label_conflicts():
    recs = [
        RawRecord("Win a prize now!", Label.SPAM, "a"),
        RawRecord("win a prize now", Label.SPAM, "b"),  # exact dupe after key
        RawRecord("See you at 5", Label.HAM, "a"),
        RawRecord("See you at 5", Label.SPAM, "b"),  # conflict
    ]
    rows, report = build_english_seed(iter(recs))
    assert report.kept == 1
    assert report.exact_dupes_dropped == 1
    assert report.label_conflicts_dropped == 2


def test_spam_vs_smishing_conflict_prefers_smishing():
    recs = [
        RawRecord("Claim your prize now", Label.SPAM, "uci"),
        RawRecord("Claim your prize now", Label.SMISHING, "mendeley"),
    ]
    rows, report = build_english_seed(iter(recs))
    assert rows[0][0].label == Label.SMISHING
    assert report.label_conflicts_resolved == 1 and report.label_conflicts_dropped == 0


def test_build_scrubs_pii():
    rows, _ = build_english_seed(iter([RawRecord("Call 0771234567 http://a.b/c", Label.SPAM, "a")]))
    assert "0771234567" not in rows[0][0].text


def test_loaders(tmp_path):
    uci = tmp_path / "uci"
    uci.write_text("ham\tHello there\nspam\tWIN cash\n", encoding="utf-8")
    assert [r.label for r in load_uci(uci)] == [Label.HAM, Label.SPAM]

    mend = tmp_path / "m.csv"
    mend.write_text("LABEL,TEXT,URL\nSmishing,Click now,yes\nham,Hi,no\n", encoding="utf-8")
    assert [r.label for r in load_mendeley(mend)] == [Label.SMISHING, Label.HAM]
