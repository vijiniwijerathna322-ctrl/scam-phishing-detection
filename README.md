# Scam & Phishing Detection in Sinhala, Tamil and Code-Mixed SMS

Builds a Sinhala / Tamil / code-mixed (Singlish, Tanglish) smishing dataset and classifiers.
No public Sinhala or Tamil smishing dataset was found when this project started, so the
dataset itself is a main contribution. See [PLAN.md](PLAN.md) for research, architecture
and roadmap, [docs/datasheet.md](docs/datasheet.md) for the dataset description and
[docs/annotation_guidelines.md](docs/annotation_guidelines.md) for reviewers.

## Status

| Stage | State |
|---|---|
| English seed (UCI + Mendeley, deduplicated, PII-scrubbed) | done, 6,012 messages |
| Machine translation to Sinhala / Tamil (NLLB-200, quality-filtered) | pilot done: 445 si, 505 ta |
| Synthetic messages from SLCERT/TRCSL scam themes (5 language varieties) | done, 3,967 messages from 38 templates (v2, with optional surface noise) |
| Baselines: TF-IDF + LR, XLM-R fine-tune | done |
| Native-speaker review (gold set) | **not started: needs reviewers** |
| Full translation (all 6,012), more synthetic templates, MuRIL/SinBERT | todo |
| Demo (FastAPI/Gradio), model + dataset release | todo |

## Results so far (read the caveats)

Macro-F1 on the multilingual split. The table uses **dataset v1** (8,732 rows, 1,770
synthetic messages); the models were trained on it:

| Model | val | test | held-out (val+test) |
|---|---|---|---|
| TF-IDF char n-grams + LR | 0.867 | **0.896** | 0.883 |
| XLM-R base, 2 epochs | 0.906 | 0.892 | 0.902 |

**Dataset v2** (10,929 rows, 3,967 synthetic messages from 38 templates, with noise) changes
which templates are held out, so its scores are not comparable with v1. TF-IDF retrained on v2:
val 0.818, test 0.955, held-out 0.880. The val-test gap (0.82 vs 0.96) shows how much scores
swing with the few held-out templates, so treat any single number as rough. Held-out slices are
now larger: code-mixed 225-231 rows (was 36), synthetic 1,138 rows scoring 0.855. XLM-R has
not been retrained on v2 (about 4 hours on CPU).

- **Use the test column to compare models.** XLM-R's epoch was chosen on val, so val and
  held-out favour it. On test the two models are indistinguishable.
- Sinhala/Tamil/code-mixed held-out data are machine translations and synthetic templates,
  **not real SMS**. Per-language scores say little about real-world Sinhala/Tamil messages.
  In v1 the code-mixed slices had only 36 messages each (now over 200, from few templates).
- TF-IDF trained on English only scores 0.72 (si) and 0.75 (ta) on the translated data, but
  much of that is shared tokens (`<URL>`, numbers, brand names), not true transfer.
- Real and synthetic data must be reported separately (see `source_type` slices).

## Setup (PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pytest
```

Training needs PyTorch and Transformers: `pip install -e ".[ml]"`, with Python 3.11 or 3.12
(PyTorch wheels may be missing for the newest Python). XLM-R fine-tuning on CPU takes about
4 hours for 2 epochs; a free GPU (Colab/Kaggle) is far faster.

## Pipeline

Every command is run from the repo root with `$env:PYTHONPATH="src"` (or after `pip install -e .`).

```powershell
python -m smishdet.data.build                 # English seed from data/raw/ (UCI, Mendeley)
python -m smishdet.data.translate_cli --n 600 # NLLB-200 pilot translation to si and ta
python -m smishdet.data.translate_cli --refilter  # re-apply quality filters, no model needed
python -m smishdet.data.synthetic             # template-based synthetic messages
python -m smishdet.data.combine               # merge into data/processed/multilingual.csv
python -m smishdet.models.train_baseline --data data/processed/multilingual.csv --out outputs/tfidf_multi_all
python -m smishdet.models.train_transformer --epochs 2 --max-length 64
python -m smishdet.data.review build          # sheet for native-speaker reviewers
python -m smishdet.data.review apply          # import answers, build the gold set
python -m smishdet.serving.app                # Gradio demo at http://127.0.0.1:7860
```

The demo (`pip install -e ".[serve]"`) loads whichever trained models exist in `outputs/`
(TF-IDF, XLM-R), scrubs links/phone numbers exactly as in training, and flags low-confidence
results. It listens on localhost only and does not store messages. It is a research demo:
its examples come from the same templates the models trained on, so high confidence there
says nothing about real-world messages.

Raw datasets are not included: download UCI SMS Spam Collection and Mendeley SMS Phishing
yourself into `data/raw/` and check their licences first (see the `ingest.py` docstring).

## Layout

`src/smishdet/{domain,data,models,evaluation,serving,utils}` (dependencies point inward),
`configs/` (YAML), `data/` (gitignored), `docs/` (guidelines, datasheet), `tests/`.
Project rules for contributors and AI assistants are in [CLAUDE.md](CLAUDE.md).

## Data, privacy and licences

- **Never commit raw messages.** Scrub PII (`smishdet.data.pii.scrub`) before any storage
  or sharing. `data/` is gitignored.
- Splits are by `template_group` (near-duplicate clusters and template keys), never random.
  Translations inherit the split of their English source message.
- NLLB-200 is CC-BY-NC 4.0, so translated data is non-commercial. Synthetic texts name real
  Sri Lankan institutions, as the public alerts do, and contain only `<URL>` and `<PHONE>`
  placeholders. Sinhala/Tamil/Singlish/Tanglish templates are **not yet native-reviewed**.
