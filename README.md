# Scam & Phishing Detection in Sinhala, Tamil and Code-Mixed SMS

Builds a Sinhala/Tamil/code-mixed smishing dataset and classifiers. See [PLAN.md](PLAN.md) for research, architecture and roadmap.

## Setup (PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pytest
```

ML extras: `pip install -e ".[ml]"`. Note: PyTorch may lack wheels for the newest Python; use Python 3.11 or 3.12 for training.

## Layout

`src/smishdet/{domain,data,models,evaluation,serving,utils}`, `configs/` (YAML), `data/` (gitignored), `docs/` (guidelines, datasheet, model card).

**Never commit raw messages.** Scrub PII (`smishdet.data.pii.scrub`) before any storage or sharing.
