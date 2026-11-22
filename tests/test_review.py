from smishdet.data.review import (
    REVIEWER_COLS,
    apply_review,
    build_sheet,
    english_reference_id,
    sample_language,
)


def _row(i, lang="si", label="smishing", group="g1", split="test", source="translated:uci"):
    return {"id": i, "text": f"text {i}", "label": label, "language": lang, "script": "sinhala",
            "source": source, "scam_type": "none", "template_group": group, "split": split}


def test_english_reference_id():
    assert english_reference_id("si-en-000123") == "en-000123"
    assert english_reference_id("ta-en-000045") == "en-000045"
    assert english_reference_id("syn-si-traffic1-003") == "syn-en-traffic1-003"
    assert english_reference_id("syn-si-en-traffic1-003") == "syn-en-traffic1-003"
    assert english_reference_id("syn-ta-en-bank2-010") == "syn-en-bank2-010"
    assert english_reference_id("en-000123") is None


def test_sheet_uses_only_heldout_and_adds_english_reference():
    rows = [_row("en-1", lang="en", split="test", source="uci"),
            _row("si-en-1", split="test"),
            _row("si-en-2", split="train")]
    sheet = build_sheet(rows, {"si": 5})
    assert [r["id"] for r in sheet] == ["si-en-1"]  # train row excluded
    assert sheet[0]["english_reference"] == "text en-1"
    assert all(sheet[0][c] == "" for c in REVIEWER_COLS)


def test_sample_balances_labels_and_spreads_templates():
    rows = [_row(f"a{i}", label="ham", group="g1") for i in range(10)]
    rows += [_row(f"b{i}", label="ham", group="g2") for i in range(10)]
    rows += [_row(f"c{i}", label="smishing", group="g3") for i in range(10)]
    picked = sample_language(rows, 6, seed=1)
    assert len(picked) == 6
    assert sum(r["label"] == "ham" for r in picked) == 3
    ham_groups = {r["template_group"] for r in picked if r["label"] == "ham"}
    assert ham_groups == {"g1", "g2"}  # not 3 variants of one template


def test_apply_review_corrects_rejects_and_summarises():
    all_rows = [_row(f"si-en-{i}") for i in range(1, 6)]

    def rv(i, **kw):
        return {"id": f"si-en-{i}", "fluency_1to5": "", "meaning_kept_yes_no": "",
                "label_check": "", "corrected_text": "", **kw}

    reviewed = [
        rv(1, fluency_1to5="5", meaning_kept_yes_no="yes", label_check="ok"),
        rv(2, fluency_1to5="2", meaning_kept_yes_no="no", corrected_text="better text"),
        rv(3, fluency_1to5="1", meaning_kept_yes_no="no"),  # bad, no fix -> rejected
        rv(4, fluency_1to5="4", meaning_kept_yes_no="yes", label_check="spam"),
        rv(5),  # unanswered
    ]
    gold, summary = apply_review(all_rows, reviewed)
    by = {g["id"]: g for g in gold}
    assert set(by) == {"si-en-1", "si-en-2", "si-en-4"}
    assert by["si-en-2"]["text"] == "better text"
    assert by["si-en-4"]["label"] == "spam"
    assert by["si-en-1"]["source"] == "reviewed:translated:uci"
    assert summary["skipped"] == {"rejected_bad_translation": 1, "unanswered": 1}
    assert summary["by_language"]["si"]["fluency"] == round((5 + 2 + 1 + 4) / 4, 3)
