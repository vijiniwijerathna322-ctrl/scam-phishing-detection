import re

import pytest

from smishdet.data.script import detect_script
from smishdet.data.synthetic import LANGS, generate, render, split_messages
from smishdet.data.synthetic_templates import DAYS, DEADLINES, META, TEMPLATES
from smishdet.domain.schema import Label, Script


def test_every_language_has_every_template():
    for lang in LANGS:
        assert set(TEMPLATES[lang]) == set(META), lang


def test_language_pools_are_parallel():
    assert len({len(v) for v in DEADLINES.values()}) == 1
    assert len({len(v) for v in DAYS.values()}) == 1


def test_label_counts_after_expansion():
    labels = [m.label for m in META.values()]
    assert labels.count(Label.SPAM) >= 8 and labels.count(Label.HAM) >= 14


def test_noise_zero_is_clean_and_noise_keeps_placeholders():
    clean = [m.text for m in generate(per_template=4, noise=0.0)]
    assert clean == [m.text for m in generate(per_template=4)]
    noisy = list(generate(per_template=8, noise=1.0))
    assert [m.text for m in noisy] != [m.text for m in generate(per_template=8)]
    clean_by_id = {m.id: m.text for m in generate(per_template=8)}
    for m in noisy:
        original = clean_by_id.get(m.id)
        if original:  # same variant, check placeholders survive and are not altered
            assert original.count("<URL>") == m.text.count("<URL>")
            assert original.count("<PHONE>") == m.text.count("<PHONE>")
        assert "<url>" not in m.text and "<phone>" not in m.text  # never lower-cased


def test_noise_is_deterministic():
    a = [m.text for m in generate(per_template=6, seed=3, noise=0.5)]
    b = [m.text for m in generate(per_template=6, seed=3, noise=0.5)]
    assert a == b


def test_noise_does_not_depend_on_label():
    """Share of noisy variants should be similar across labels (no label shortcut)."""
    clean = {m.id: m.text for m in generate(per_template=20)}
    rates = {}
    for label in (Label.HAM, Label.SPAM, Label.SMISHING):
        msgs = [m for m in generate(per_template=20, noise=0.3) if m.label == label]
        changed = sum(m.text != clean.get(m.id, m.text) for m in msgs)
        rates[label] = changed / len(msgs)
    assert max(rates.values()) - min(rates.values()) < 0.15, rates


@pytest.mark.parametrize("lang", list(LANGS))
def test_no_unfilled_slots_or_stray_braces(lang):
    for key in META:
        for v in range(5):
            text = render(key, lang, v)
            assert "{" not in text and "}" not in text, (lang, key, text)


def test_script_matches_language():
    for m in generate(per_template=2):
        found = detect_script(re.sub(r"<\w+>", "", m.text))
        if m.script == Script.LATIN:
            assert found == Script.LATIN, m.text
        else:
            # Sinhala/Tamil messages may contain Latin brand names, but must contain the script
            assert m.script.value in {"sinhala", "tamil"}
            assert any(0x0D80 <= ord(c) <= 0x0DFF or 0x0B80 <= ord(c) <= 0x0BFF for c in m.text)


def test_variants_share_choices_across_languages():
    en = render("bank1", "en", 3)
    si = render("bank1", "si", 3)
    bank = en.split(":")[0]
    assert si.startswith(bank)


def test_deterministic_and_unique_ids():
    a = list(generate(per_template=5, seed=1))
    b = list(generate(per_template=5, seed=1))
    assert [m.text for m in a] == [m.text for m in b]
    assert len({m.id for m in a}) == len(a)


def test_all_classes_in_every_split_and_groups_whole():
    msgs = list(generate(per_template=10))
    splits = split_messages(msgs)
    for label in Label:
        assert {s for m, s in zip(msgs, splits, strict=True) if m.label == label} >= {
            "train", "val", "test"}
    seen: dict[str, str] = {}
    for m, s in zip(msgs, splits, strict=True):
        assert seen.setdefault(m.template_group or "", s) == s


def test_smishing_has_scam_type_and_ham_does_not():
    for m in generate(per_template=1):
        assert (m.scam_type.value != "none") == (m.label == Label.SMISHING)
