from collections import Counter

from smishdet.data.translate import (
    check_translation,
    mask_placeholders,
    refilter_rows,
    script_fraction,
    stratified_sample,
    translate_rows,
    unmask_placeholders,
)


def test_mask_roundtrip():
    masked, orig = mask_placeholders("Pay at <URL> or call <PHONE>")
    assert masked == "Pay at [1] or call [2]"
    text, ok = unmask_placeholders("ගෙවන්න [1] හෝ [2]", orig)
    assert ok and text == "ගෙවන්න <URL> හෝ <PHONE>"


def test_unmask_detects_lost_marker():
    _, ok = unmask_placeholders("no marker here", ["<URL>"])
    assert not ok


def _row(i, text, label="ham"):
    return {"id": f"en-{i}", "text": text, "label": label, "language": "en", "script": "latin",
            "source": "uci", "scam_type": "none", "template_group": f"c{i}", "split": "train"}


def test_translate_rows_metadata_and_drops():
    rows = [_row(1, "Hello <URL>"), _row(2, "Hello <URL> again")]
    def fake(texts):
        return ["ආයුබෝවන් [1]", "ආයුබෝවන්"]  # second loses its marker

    out = translate_rows(rows, fake, "si", "sinhala")
    assert len(out) == 1
    r = out[0]
    assert r["text"] == "ආයුබෝවන් <URL>" and r["language"] == "si"
    assert r["source"] == "translated:uci" and r["split"] == "train" and r["id"] == "si-en-1"


def test_check_translation_reasons():
    long_src = "Txt CHAT to 86688 now to claim your free ringtone and join the club today ok"
    si_ok = "දැන් ඔබේ නොමිලේ ringtone එක ලබාගන්න 86688 වෙත CHAT යවන්න සහ සමාජයට එක්වන්න"
    assert check_translation(long_src, si_ok, "sinhala") is None
    assert check_translation("hi", "", "sinhala") == "empty"
    untranslated = "Free Msg: Ringtone"
    assert check_translation(untranslated, untranslated, "sinhala") == "wrong_script"
    assert check_translation(long_src, "ඔබට ආයුබෝවන්", "sinhala") == "too_short"
    assert check_translation("Txt CHAT to 86688", "CHAT යවන්න", "sinhala") == "number_lost"


def test_script_fraction_ignores_placeholders():
    assert script_fraction("<URL> ආයුබෝවන්", "sinhala") == 1.0
    assert script_fraction("வணக்கம் hello", "tamil") < 0.6


def test_refilter_rows():
    english = {"en-1": _row(1, "Free Msg: Ringtone")}
    bad = {**_row(1, "Free Msg: Ringtone"), "id": "si-en-1"}
    stats = Counter()
    assert refilter_rows([bad], english, "sinhala", stats) == []
    assert stats["wrong_script"] == 1


def test_stratified_sample_balances_labels():
    rows = [_row(i, "x", "ham") for i in range(50)]
    rows += [_row(100 + i, "y", "spam") for i in range(10)]
    picked = stratified_sample(rows, 12)
    assert sum(r["label"] == "spam" for r in picked) == 6
