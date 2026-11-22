# Scam & Phishing Detection in Sinhala, Tamil and Code-Mixed SMS — Project Plan

## 1. Research Findings (summary)

**Gap confirmed.** A 2026 systematic review of SMS phishing ([arXiv 2604.11429](https://arxiv.org/pdf/2604.11429)) says public datasets are overwhelmingly English. No public Sinhala or Tamil smishing dataset turned up. One ACL 2026 paper ([nlpaics 1.7](https://aclanthology.org/2026.nlpaics-1.7.pdf)) uses a proprietary multilingual SMS set; I did not verify whether it includes Sinhala/Tamil. Read it before finalising scope.

| Resource | Use in this project | Caveat |
|---|---|---|
| [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) | English ham/spam seed, baseline | Old (2012), no smishing class |
| [Mendeley SMS Phishing](https://data.mendeley.com/datasets/f45bkkt8pr) (5,971 msgs, ham/spam/smishing) | English 3-class seed, translation source | ~4,753 messages overlap UCI: deduplicate |
| [Smishtank](https://arxiv.org/abs/2402.18430) (1,090 smishing) | Smishing templates/URL patterns | Smishing only; check license |
| [SinBERT](https://huggingface.co/NLPC-UOM/SinBERT-large) | Sinhala encoder | [BERTifying Sinhala](https://arxiv.org/pdf/2208.07864) found XLM-R-large beat it on classification |
| XLM-R, MuRIL, IndicBERT | Tamil/multilingual encoders | MuRIL often best on Tamil ([DravidianLangTech 2026](https://aclanthology.org/2026.dravidianlangtech-1.22/)); tasks differ, so benchmark yourself |
| [Sinhala-English code-mixed, 10k comments](https://dl.lib.uom.lk/items/6339c67b-d7a7-4b20-b812-14cfd7cf4599) | Code-mixing style, language-ID | Social-media comments, not SMS |
| [TamilMixSentiment](https://arxiv.org/pdf/2006.00206) (15.7k) | Tanglish style reference | Sentiment, not scam |
| SinMix2Mono (2026, [EAMT](https://aclanthology.org/2026.eamt-1.10/)) | Romanized Sinhala to script normalisation | Verify availability |

**Real-world scam themes in Sri Lanka** (SLCERT/TRCSL alerts, e.g. [traffic fine](https://www.adaderana.lk/news/123052/sri-lanka-cert-issues-public-warning-over-fake-traffic-fine-sms-and-whatsapp-scam), [postal/customs](https://www.sundaytimes.lk/240421/news/thousands-fall-prey-to-massive-sms-fraud-using-postal-dept-as-a-front-554749.html), [card suspension, prizes](https://adaderana.lk/news/105027/sri-lanka-cert-alerts-on-spike-in-scams-through-sms-and-phone-calls)): fake police fines, unpaid customs/delivery fees, bank/credit-card suspension, prize/lottery, government cash aid, job/loan offers, OTP/NIC harvesting. Use these as the label taxonomy and seed templates.

**Open questions to resolve early**
1. Licensing/ethics for any real SMS (consent, PII). Plan assumes synthetic + public-sourced only unless donors consent.
2. Whether to ask SLCERT/TRCSL/telcos for collaboration (high value, slow).

## 2. Goals and Deliverables
1. **Dataset** `sl-smish` (Sinhala, Tamil, English, Singlish, Tanglish, mixed), labels: `ham`, `spam`, `smishing` + scam-type subcategory.
2. **Classifier(s)** on Hugging Face Hub with model card.
3. **Evaluation report** (per-language, per-script, robustness).
4. **Demo** (Gradio/FastAPI) and reproducible pipeline.

Non-goals: production telecom integration, URL takedown.

## 3. Architecture

```
 Sources ──► Ingestion ──► Cleaning/PII scrub ──► Normalisation ──► Annotation ──► Versioned Dataset (HF Hub)
 (public EN sets,            (regex + NER,          (Unicode NFC,     (Label Studio,     │
  synthetic, crowd,           dedupe)               script/lang ID,   2 annotators,      ▼
  consented donors)                                 translit)         adjudication)   Training pipeline
                                                                                         │ baselines: TF-IDF+LR
                                                                                         │ encoders: XLM-R, MuRIL, SinBERT
                                                                                         ▼
                                                              Evaluation ──► Model registry (HF Hub) ──► Inference API ──► Demo UI
```

**Layers (clean architecture, dependencies point inward)**
- `domain`: Message, Label, LanguageTag, taxonomy (pure Python, no I/O).
- `data`: loaders, cleaners, normalisers, splitters, dataset builders.
- `models`: model factory, training, calibration, inference wrapper (common `Classifier` interface).
- `evaluation`: metrics, slicing, robustness tests.
- `serving`: FastAPI + Gradio; thin, depends on `models` interface only.

**Design principles applied**: single responsibility per module; strategy pattern for models (swap encoder via config); factory for tokenizers/models; dependency inversion (serving depends on an interface); config over code (Hydra/YAML); pure functions for transforms; immutability of raw data.

## 4. Data Strategy (the core of the project)

**Sourcing, in priority order**
1. Public English datasets (UCI, Mendeley, Smishtank), deduplicated, translated to Sinhala/Tamil via NLLB-200 / IndicTrans2 / LLM, then human-corrected (sample at least 20% reviewed, report error rate).
2. Template-based synthetic generation from SLCERT/TRCSL alert themes: slots for brands (banks, Sri Lanka Post, SL Police, Dialog/Mobitel), amounts, fake URLs. Use an LLM for paraphrase diversity; mark `source=synthetic`.
3. Code-mixed: generate by romanising (Singlish/Tanglish transliteration) and token-level mixing; validate with native speakers since romanisation is non-standard.
4. Crowd/community donation form (opt-in, anonymised) and public screenshots (OCR) only where licensing allows.
5. Hard ham: legit bank alerts, OTPs, delivery notices, telco promos, so the model doesn't learn "mentions bank = scam".

**Schema** (per row): `id, text, label, scam_type, language {si,ta,en,si-en,ta-en,mixed}, script {sinhala,tamil,latin,mixed}, source, has_url, has_phone, split, annotator_agreement, license`.

**Quality**: written annotation guideline; 2 annotators plus adjudicator; Cohen's/Krippendorff's kappa target >= 0.75; near-duplicate removal (MinHash) **before** splitting to prevent leakage; split by template/campaign group, not randomly. Report synthetic vs real performance separately.

**Ethics/Legal**: scrub phone numbers, names, NICs, account numbers; Sri Lanka PDPA 2022 awareness; dataset card with intended use, limitations, bias; gated access if real messages included; CC BY-NC or CC BY 4.0 depending on source licences.

## 5. Modelling Plan
1. **Baselines**: char n-gram TF-IDF + Logistic Regression/SVM (char n-grams handle Sinhala/Tamil morphology and romanisation noise).
2. **Transformers** fine-tuned: XLM-R-base/large, MuRIL (Tamil), SinBERT (Sinhala), possibly IndicBERT; optionally LaBSE/multilingual-e5 embeddings + LR as cheap alternative.
3. **Techniques**: class-weighted loss, handcrafted features (URL presence/shortener, phone, urgency words) concatenated; transliteration augmentation; adversarial augmentation (typos, zero-width chars, mixed scripts); optional distillation to a small model for on-device use.
4. **Zero/few-shot LLM baseline** for comparison.
5. **Calibration** (temperature scaling) and threshold selection for a low false-positive operating point.

## 6. Evaluation
- Metrics: macro-F1 (primary), per-class precision/recall, PR-AUC for smishing, false-positive rate on hard ham, ECE.
- Slices: language, script, source (real vs synthetic), scam type, message length.
- Tests: cross-lingual transfer (train EN, test SI/TA), cross-source generalisation, adversarial robustness, unseen-campaign holdout.
- Error analysis notebook; confidence intervals via bootstrap; 3+ seeds.

## 7. Tech Stack
- **Language/tools**: Python 3.11, `uv` or Poetry, Ruff + Black + mypy, pre-commit, pytest.
- **ML**: PyTorch, Hugging Face `transformers`, `datasets`, `evaluate`, `peft` (optional), scikit-learn, `sentencepiece`.
- **NLP/Indic**: `indic-nlp-library`, `sinling`/`sinhala-nlp` tools (verify maintenance), `indic-transliteration`, `fastText` lid / `langid` + custom script detector via Unicode ranges (Sinhala U+0D80–0DFF, Tamil U+0B80–0BFF).
- **Annotation**: Label Studio.
- **Experiment tracking**: Weights & Biases or MLflow; Hydra configs; DVC for data versioning (or HF Hub datasets with revisions).
- **Serving**: FastAPI + Gradio/HF Spaces; ONNX/quantised model for CPU.
- **CI**: GitHub Actions (lint, tests, small-sample smoke training).
- **Compute**: Colab/Kaggle GPU or local; base-size models fit on a 16 GB GPU.

## 8. Repository Layout
```
ScamAndPhishingDetection/
├─ PLAN.md
├─ pyproject.toml
├─ configs/            # hydra: data/, model/, train/, eval/
├─ data/               # raw/ (read-only, gitignored), interim/, processed/
├─ docs/               # annotation_guidelines.md, datasheet.md, model_card.md
├─ notebooks/
├─ src/smishdet/
│  ├─ domain/  data/  models/  evaluation/  serving/  utils/
├─ tests/
└─ .github/workflows/
```

## 9. Standards and Best Practices
- **Reproducibility**: fixed seeds, pinned deps, config-driven runs, data versions in model cards.
- **Documentation**: Datasheets for Datasets, Model Cards, annotation guideline versioning.
- **Code**: type hints, docstrings, small pure functions, no notebook-only logic, >= 80% coverage on `data/` transforms.
- **Data hygiene**: raw immutable; group-aware splits; leakage tests in CI (assert zero near-dup overlap across splits).
- **Security/privacy**: no secrets in repo, PII scrubber tested, gated release.
- **Responsible AI**: report FP risk (blocking legit bank SMS), publish limitations, human-in-the-loop recommendation.
- **Git**: trunk-based, conventional commits, PR templates.

## 10. Roadmap (approx. 12 weeks, intermediate level)
| Weeks | Milestone |
|---|---|
| 1 | Read key papers (incl. nlpaics 1.7), finalise taxonomy, repo scaffold, ethics/licence decisions |
| 2–3 | Collect and clean English seeds; build language/script ID; dedupe |
| 3–5 | Synthetic + translated Sinhala/Tamil; romanised/code-mixed generation; annotation guideline |
| 5–7 | Annotation rounds, agreement check, hard-ham set, dataset v0.1 on HF Hub (gated) |
| 7–9 | Baselines + transformer fine-tuning, ablations |
| 9–10 | Evaluation slices, robustness, calibration |
| 10–11 | Demo, model + dataset cards, paper-style write-up |
| 12 | Release v1.0, optional workshop submission (e.g. DravidianLangTech, LoResMT) |

## 11. Risks and Mitigations
| Risk | Mitigation |
|---|---|
| Synthetic data not representative of real scams | Real-sample validation set from SLCERT-style public alerts/consented donors; report separately |
| Poor MT quality for SMS slang | Human review sample; back-translation check |
| Romanisation inconsistency | Multiple transliteration schemes; native-speaker check |
| Label noise | Guidelines, adjudication, confident-learning pass |
| Privacy of real SMS | Scrubbing, consent, gated release |
| Overfitting to templates | Template-group splits, unseen-campaign test |

## 11b. Phase 2 (extension, after the SMS dataset and classifier are done)
Multi-channel: WhatsApp and Facebook messages. Plan: add a `channel` field (`sms`, `whatsapp`, `facebook`) to the schema, new scam types (fake job/loan pages, giveaways, Marketplace, account-hijack), longer `max_length`, emoji handling, and a separate evaluation slice per channel. Data only from synthetic messages modelled on SLCERT/TRCSL alerts and consented, PII-scrubbed donations. No scraping of private chats or Facebook (privacy law and platform terms). Decide the exact API/data route when Phase 2 starts.

## 12. Immediate Next Steps
1. Read the nlpaics 1.7 paper and the SMS phishing review for taxonomy and methods.
2. Scaffold repo (`uv init`, pre-commit, package layout).
3. Download UCI + Mendeley + Smishtank, dedupe, and profile.
4. Draft annotation guideline and scam-type taxonomy from SLCERT alerts.
5. Prototype TF-IDF baseline on English as the first end-to-end pipeline.
