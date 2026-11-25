# Datasheet: Sinhala / Tamil / code-mixed smishing dataset (v0.1, draft)

Follows the "Datasheets for Datasets" outline. **Not released.** This describes the
working dataset in `data/processed/multilingual.csv`. Update before any release.

## Motivation
- **Why:** no public Sinhala or Tamil smishing dataset was found; existing public SMS
  datasets are almost entirely English.
- **Intended use:** research on scam/phishing SMS detection for low-resource South Asian
  languages, model benchmarking, and data augmentation.
- **Not for:** blocking real users' messages without human review, profiling individuals,
  or any production decision. It is not a benchmark of real Sinhala/Tamil SMS yet.

## Composition (v0.2, 10,929 rows)
| Part | Rows | What it is |
|---|---|---|
| English seed | 6,012 | UCI SMS Spam Collection + Mendeley SMS Phishing, deduplicated |
| Translated si | 445 | Machine-translated (NLLB-200) from English seed, quality-filtered |
| Translated ta | 505 | Same, Tamil |
| Synthetic | 3,967 | 38 templates in en, si, ta, si-en (Singlish), ta-en (Tanglish), 30% of variants with surface noise |

Labels: `ham`, `spam`, `smishing` (+ `scam_type` for synthetic smishing: traffic fine,
customs/delivery, bank card, prize/lottery, government aid, job/loan, OTP/NIC harvesting,
other utility-disconnection scams).
Overall: 6,696 ham, 1,466 spam, 2,767 smishing.
(v0.1 had 8,732 rows with 23 templates; the models and the Hugging Face staging in this
repo were built on v0.1 unless stated.)

Columns: `id, text, label, language, script, source, scam_type, template_group, split`.
`source` is `uci`, `mendeley`, `translated:<origin>` or `synthetic`.

Splits (train/val/test) are assigned by group: near-duplicate cluster for English, and
template key for synthetic. A translation shares the split of its English source.

## Collection process
- English: public datasets, downloaded from their official pages. Smishtank was not
  available (its page returned 404).
- Translation: `facebook/nllb-200-distilled-600M`, with `<URL>`/`<PHONE>` placeholders masked
  during translation. Rows are dropped if placeholders, numbers or length are lost, or the
  output is not mostly in the target script.
- Synthetic: templates modelled on public SLCERT/TRCSL alerts (fake traffic fines, customs
  fees, card suspension, prizes, aid, jobs/loans, OTP/NIC requests) plus hard ham (legitimate
  bank OTPs and alerts, bills, deliveries, personal messages). No private messages were used.

## Preprocessing
Unicode NFC, zero-width character removal (except ZWJ, needed in Sinhala), whitespace
collapse, PII scrubbing (URLs, Sri Lankan phone numbers, NIC, card numbers),
exact-duplicate removal, MinHash near-duplicate clustering. Spam vs smishing label
conflicts between UCI and Mendeley are resolved to smishing (286 texts); other conflicts
would be dropped (none occurred).

## Known limitations and biases
- **No real Sinhala/Tamil SMS yet.** All non-English rows are machine translations or
  templates. Per-language results are indicative only.
- **Translation quality is uneven.** Pilot inspection showed truncation and untranslated
  SMS slang; filters removed 13-19% of rows but cannot confirm correctness.
- **Native-speaker review has not happened.** Sinhala/Tamil/Singlish/Tanglish templates were
  written without review. The review workflow and guideline exist (`docs/annotation_guidelines.md`).
- **Low diversity in synthetic data.** 38 templates only (16 smishing, 8 spam, 14 ham).
  Many rows differ only in slot values or light noise. Held-out evaluation depends on just
  2-3 held-out templates per label, so scores swing a lot between val and test.
- **English seed bias.** UCI/Mendeley are older, UK/Indian-English and not Sri Lankan; the
  UCI and Mendeley sets overlap (about 4,753 common messages, deduplicated here).
- **Class imbalance** (smishing is 25% overall, 75% of synthetic data).
- Brand names (banks, telcos, Sri Lanka Police/Post) are real institutions used as in public
  alerts. No live links or real phone numbers are included.

## Licensing (check before release)
- UCI and Mendeley: verify each dataset's current licence and attribution requirements.
- NLLB-200 outputs: model is CC-BY-NC 4.0, so translated rows are treated as non-commercial.
- Synthetic rows and code: choose a licence deliberately; a non-commercial licence is the
  safe default while translated data is included.

## Distribution and maintenance
Not distributed. Plan: a gated release on the Hugging Face Hub with this datasheet, after
native-speaker review. Contact and versioning to be filled in at release.
