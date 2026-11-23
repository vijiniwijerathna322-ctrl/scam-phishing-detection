"""Stage and upload a Hugging Face release (private by default).

  python -m smishdet.release stage                 # build release/ locally, no network
  python -m smishdet.release upload --user NAME    # needs `hf auth login` done by you

Scope (decided with the project owner): synthetic dataset + TF-IDF and XLM-R models.
UCI/Mendeley rows and NLLB translations are NOT released: the Mendeley licence is
unconfirmed and the licence status of NLLB outputs is unclear.

Cards are generated from metrics.json so the numbers cannot drift from the real results.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

USER_PLACEHOLDER = "{user}"
REPOS = {"dataset": "sl-smish-synthetic", "tfidf": "sl-smish-tfidf", "xlmr": "sl-smish-xlmr"}

WARNING = (
    "**Research artifact, not a security product.** Do not use it to block or flag real "
    "users' messages without human review. Non-English evaluation data are synthetic "
    "templates and machine translations, not real Sinhala/Tamil SMS."
)


def _f1(metrics: dict[str, Any], split: str) -> str:
    return f"{metrics[split]['macro_f1']:.3f}"


def dataset_card(rows: list[dict[str, str]]) -> str:
    labels = Counter(r["label"] for r in rows)
    langs = Counter(r["language"] for r in rows)
    splits = Counter(r["split"] for r in rows)
    return f"""---
license: cc-by-4.0
pretty_name: Sri Lankan scam SMS (synthetic, multilingual)
language: [en, si, ta]
task_categories: [text-classification]
tags: [smishing, phishing, sms, sinhala, tamil, code-mixed, synthetic]
size_categories: [1K<n<10K]
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train.csv
  - split: validation
    path: data/val.csv
  - split: test
    path: data/test.csv
---

# Sri Lankan scam SMS (synthetic, multilingual)

{WARNING}

Template-generated SMS messages in English, Sinhala, Tamil, Singlish (Sinhala in Latin
letters) and Tanglish (Tamil in Latin letters), modelled on public Sri Lanka CERT and
TRCSL scam alerts. The same message exists in all five varieties (parallel data).

## Contents ({len(rows)} rows)
- Labels: {dict(labels)}
- Language varieties: {dict(langs)}
- Splits: {dict(splits)} (assigned per template, so no template appears in two splits)
- Smishing types: fake traffic fines, parcel/customs fees, bank card suspension or
  suspicious transaction, prizes/lotteries, government aid, jobs and loans, OTP/NIC/SIM
  requests. Ham includes hard cases such as legitimate bank OTPs and debit alerts.

Columns: `id, text, label, language, script, source, scam_type, template_group, split`.
Links and phone numbers are the placeholders `<URL>` and `<PHONE>`; no real links or
numbers are included. Messages name real Sri Lankan institutions (banks, telcos, Sri Lanka
Police and Post) as the public alerts do; they are fabricated and do not come from them.

## Limitations (read before use)
- **Only 23 templates.** Diversity is low, especially for spam ({labels.get('spam', 0)} rows)
  and ham. Many rows differ only in slot values (bank, amount, deadline).
- **Not reviewed by native speakers.** Sinhala, Tamil, Singlish and Tanglish texts were
  written without native-speaker review and may be unnatural or wrong.
- Models trained on this data look far better on it than on real messages, because test
  templates resemble training templates. Use it for augmentation and stress tests, not as a
  benchmark of real-world performance.
- Not a sample of real scams: real messages are messier, longer and change quickly.

## Intended use
Research on scam/phishing detection in low-resource languages, and data augmentation.
Out of scope: production filtering, profiling people, any use that treats it as ground truth
for real messages.

Project code and the other datasets' status: see the GitHub repository
`{USER_PLACEHOLDER}/scam-phishing-detection`.
"""


def tfidf_card(metrics: dict[str, Any]) -> str:
    return f"""---
license: cc-by-nc-4.0
library_name: sklearn
pipeline_tag: text-classification
language: [en, si, ta]
tags: [smishing, phishing, sms, sinhala, tamil, tf-idf]
---

# TF-IDF + logistic regression baseline for scam SMS

{WARNING}

Character n-gram (2-5) TF-IDF with class-balanced logistic regression. Labels: `ham`,
`spam`, `smishing`. Small (about 3 MB) and fast; useful as a baseline.

## Training data
UCI SMS Spam Collection (Almeida & Hidalgo, CC BY 4.0), Mendeley SMS Phishing dataset
(Mishra & Soni, 2022), machine translations of those into Sinhala and Tamil (NLLB-200,
CC-BY-NC 4.0), and synthetic templates ({USER_PLACEHOLDER}/{REPOS['dataset']}). Because of
the translated data, this model is released as **CC BY-NC 4.0**.

## Results (macro-F1, multilingual split)
| val | test | held-out (val+test) |
|---|---|---|
| {_f1(metrics, 'val')} | **{_f1(metrics, 'test')}** | {_f1(metrics, 'heldout')} |

Compare models on the **test** column. Non-English held-out rows are machine translations
and templates, so per-language scores say little about real messages.

## Usage
```python
import joblib
model = joblib.load("model.joblib")   # sklearn Pipeline
model.predict(["Your card is suspended, verify now <URL>"])
```
Replace links/phone numbers with `<URL>` / `<PHONE>` first, as in training.
**Security:** `joblib` files are pickles and can run code when loaded. Load only files you trust.

## Limitations
Trained on old English SMS plus machine-translated and synthetic text; not tested on real
Sinhala or Tamil SMS. Not a security product.
"""


def xlmr_card(metrics: dict[str, Any]) -> str:
    return f"""---
license: cc-by-nc-4.0
library_name: transformers
pipeline_tag: text-classification
base_model: FacebookAI/xlm-roberta-base
language: [en, si, ta]
tags: [smishing, phishing, sms, sinhala, tamil, code-mixed]
---

# XLM-R base fine-tuned for scam SMS (en / si / ta / code-mixed)

{WARNING}

`xlm-roberta-base` fine-tuned for 3 labels: `ham`, `spam`, `smishing`. Fine-tuned for
{metrics['epochs']} epochs (best epoch chosen on val), max length {metrics['max_length']}, on
{metrics['train_rows']} rows: UCI + Mendeley English SMS, NLLB-200 translations into
Sinhala/Tamil, and synthetic templates ({USER_PLACEHOLDER}/{REPOS['dataset']}).
Released as **CC BY-NC 4.0** because of the translated training data.

## Results (macro-F1, multilingual split)
| val (used to pick the epoch) | test | held-out (val+test) |
|---|---|---|
| {_f1(metrics, 'val')} | **{_f1(metrics, 'test')}** | {_f1(metrics, 'heldout')} |

A TF-IDF baseline scores about the same on **test** (see {USER_PLACEHOLDER}/{REPOS['tfidf']}), so
this model shows no clear gain yet. Val and held-out favour this model because val chose the
epoch. Non-English held-out rows are machine translations and templates, not real SMS.

## Usage
```python
from transformers import pipeline
clf = pipeline("text-classification", model="{USER_PLACEHOLDER}/{REPOS['xlmr']}", top_k=None)
clf("Your card is suspended, verify now <URL>")
```
Replace links/phone numbers with `<URL>` / `<PHONE>` first, as in training.

## Limitations
Not evaluated on real Sinhala/Tamil SMS; high confidence on template-like text is expected
and does not carry over to real messages. Not a security product.
"""


def stage_release(
    synthetic_csv: Path, tfidf_dir: Path, xlmr_best: Path, out: Path
) -> dict[str, Path]:
    """Copy exactly the files to be published into out/{dataset,tfidf,xlmr}."""
    if out.exists():
        shutil.rmtree(out)
    paths = {k: out / k for k in REPOS}

    with synthetic_csv.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if any(r["source"] != "synthetic" for r in rows):
        raise ValueError("dataset release must contain synthetic rows only")
    (paths["dataset"] / "data").mkdir(parents=True)
    for split in ("train", "val", "test"):
        with (paths["dataset"] / "data" / f"{split}.csv").open(
            "w", encoding="utf-8", newline=""
        ) as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(r for r in rows if r["split"] == split)
    (paths["dataset"] / "README.md").write_text(dataset_card(rows), encoding="utf-8")

    paths["tfidf"].mkdir(parents=True)
    shutil.copy(tfidf_dir / "model.joblib", paths["tfidf"] / "model.joblib")
    tfidf_metrics = json.loads((tfidf_dir / "metrics.json").read_text(encoding="utf-8"))
    (paths["tfidf"] / "README.md").write_text(tfidf_card(tfidf_metrics), encoding="utf-8")

    shutil.copytree(xlmr_best, paths["xlmr"])
    xlmr_metrics = json.loads((xlmr_best.parent / "metrics.json").read_text(encoding="utf-8"))
    (paths["xlmr"] / "README.md").write_text(xlmr_card(xlmr_metrics), encoding="utf-8")
    return paths


def upload(paths: dict[str, Path], user: str, private: bool = True) -> list[str]:
    """Create repos (private unless told otherwise) and upload. Needs a prior `hf auth login`."""
    from huggingface_hub import HfApi

    api = HfApi()
    urls: list[str] = []
    for key, path in paths.items():
        readme = path / "README.md"
        readme.write_text(readme.read_text(encoding="utf-8").replace(USER_PLACEHOLDER, user),
                          encoding="utf-8")
        repo_id = f"{user}/{REPOS[key]}"
        repo_type = "dataset" if key == "dataset" else "model"
        api.create_repo(repo_id, repo_type=repo_type, private=private, exist_ok=True)
        api.upload_folder(folder_path=str(path), repo_id=repo_id, repo_type=repo_type,
                          commit_message="Initial release")
        urls.append(f"https://huggingface.co/{'datasets/' if key == 'dataset' else ''}{repo_id}")
    return urls


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("stage", "upload"):
        s = sub.add_parser(name)
        s.add_argument("--out", type=Path, default=Path("release"))
        s.add_argument("--synthetic", type=Path, default=Path("data/interim/synthetic.csv"))
        s.add_argument("--tfidf", type=Path, default=Path("outputs/tfidf_multi_all"))
        s.add_argument("--xlmr", type=Path, default=Path("outputs/xlmr_multi_all/best"))
    up = sub.choices["upload"]
    up.add_argument("--user", required=True, help="your Hugging Face username")
    up.add_argument("--public", action="store_true", help="default is private")
    args = p.parse_args()

    paths = stage_release(args.synthetic, args.tfidf, args.xlmr, args.out)
    for key, path in paths.items():
        files = sorted(x for x in path.rglob("*") if x.is_file())
        size = sum(x.stat().st_size for x in files) / 1e6
        print(f"{key:8s} -> {REPOS[key]}: {len(files)} files, {size:.1f} MB")
    if args.cmd == "upload":
        for url in upload(paths, args.user, private=not args.public):
            print("uploaded:", url)


if __name__ == "__main__":
    main()
