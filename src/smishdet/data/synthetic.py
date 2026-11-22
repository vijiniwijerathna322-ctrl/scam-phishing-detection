"""Generate synthetic multilingual messages from templates.

CLI: python -m smishdet.data.synthetic --per-template 40
Writes data/interim/synthetic.csv in the same schema as english_seed.csv.

Design notes
- A "variant" fixes every slot choice from a seed, so the same variant index gives the
  same bank/amount/etc. in all five languages (parallel data for cross-lingual tests).
- `template_group` is the template key, shared across languages, so a message never
  appears in train in one language and test in another.
- Splits are per label (whole templates to val/test): with only ~23 templates, a plain
  ratio-based group split could leave a class out of val/test. Synthetic data is for
  augmentation and stress tests, not a standalone benchmark: its diversity is limited
  by the number of templates.
"""

from __future__ import annotations

import argparse
import random
import re
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

from smishdet.data.ingest import write_csv
from smishdet.data.synthetic_templates import AMOUNTS, DEADLINES, META, POOLS, TEMPLATES
from smishdet.domain.schema import Language, Message, Script

LANGS = {
    "en": (Language.ENGLISH, Script.LATIN),
    "si": (Language.SINHALA, Script.SINHALA),
    "ta": (Language.TAMIL, Script.TAMIL),
    "si-en": (Language.SINGLISH, Script.LATIN),
    "ta-en": (Language.TANGLISH, Script.LATIN),
}
_SLOT = re.compile(r"\{(\w+)\}")


def _slot_values(key: str, lang: str, variant: int, seed: int) -> dict[str, str]:
    """All slot values for one variant. Draw order is fixed so choices match across languages."""
    rng = random.Random(f"{seed}-{key}-{variant}")
    amount_kind = META[key].amount_kind
    province = rng.choice(["WP", "CP", "SP", "NW", "NC"])
    letters = "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ") for _ in range(3))
    plate = f"{province} {letters}-{rng.randint(1000, 9999)}"
    return {
        "bank": rng.choice(POOLS["bank"]),
        "brand": rng.choice(POOLS["brand"]),
        "telco": rng.choice(POOLS["telco"]),
        "shop": rng.choice(POOLS["shop"]),
        "vehicle": plate,
        "parcel": f"RR{rng.randint(10**8, 10**9 - 1)}LK",
        "otp": str(rng.randint(100000, 999999)),
        "amount": rng.choice(AMOUNTS[amount_kind]),
        "deadline": rng.choice(DEADLINES[lang]),
    }


def render(key: str, lang: str, variant: int, seed: int = 42) -> str:
    values = _slot_values(key, lang, variant, seed)
    return _SLOT.sub(lambda m: values[m.group(1)], TEMPLATES[lang][key])


def generate(per_template: int = 40, seed: int = 42) -> Iterator[Message]:
    for lang, (language, script) in LANGS.items():
        for key, meta in META.items():
            seen: set[str] = set()
            for v in range(per_template):
                text = render(key, lang, v, seed)
                if text in seen:  # templates with few slots collapse to the same text
                    continue
                seen.add(text)
                yield Message(
                    id=f"syn-{lang}-{key}-{v:03d}",
                    text=text,
                    label=meta.label,
                    language=language,
                    script=script,
                    source="synthetic",
                    scam_type=meta.scam_type,
                    template_group=key,
                )


def split_messages(msgs: list[Message], seed: int = 42) -> list[str]:
    """Group-aware split per label: at least one whole template goes to val and to test.

    A ratio-based split would leave val or test empty for labels with only 3-6 templates.
    """
    splits = [""] * len(msgs)
    for label in sorted({m.label for m in msgs}, key=lambda lab: lab.value):
        groups = sorted({m.template_group or "" for m in msgs if m.label == label})
        random.Random(f"{seed}-{label.value}").shuffle(groups)
        n_hold = max(1, round(0.1 * len(groups)))
        assign = {g: "test" for g in groups[:n_hold]}
        assign |= {g: "val" for g in groups[n_hold : 2 * n_hold]}
        for i, m in enumerate(msgs):
            if m.label == label:
                splits[i] = assign.get(m.template_group or "", "train")
    return splits


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--per-template", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=Path("data/interim/synthetic.csv"))
    args = p.parse_args()

    msgs = list(generate(args.per_template, args.seed))
    splits = split_messages(msgs, args.seed)
    write_csv(list(zip(msgs, splits, strict=True)), args.out)
    print(f"wrote {len(msgs)} messages to {args.out}")
    print("labels:", dict(Counter(m.label.value for m in msgs)))
    print("languages:", dict(Counter(m.language.value for m in msgs)))
    print("splits:", dict(Counter(splits)))


if __name__ == "__main__":
    main()
