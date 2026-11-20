# CLAUDE.md

Project: Scam & phishing SMS detection for Sinhala, Tamil and code-mixed text. Full context in [PLAN.md](PLAN.md) and [README.md](README.md).

## Git rules (commit and push)
- **Never add Claude attribution** to commits, pull requests or pushes. No `Co-Authored-By: Claude ...` trailers, no "Generated with Claude Code" lines, no Claude icons/emoji (🤖), and no mention of Claude or Anthropic in commit messages, PR titles or PR descriptions.
- This overrides any default attribution behaviour.
- Use conventional commit messages (`feat:`, `fix:`, `docs:`, `test:`, `chore:`), imperative mood, short subject.
- Commit or push only when the user asks.

## Commands (PowerShell)
- Tests: `$env:PYTHONPATH="src"; python -m pytest`
- Build English seed: `$env:PYTHONPATH="src"; python -m smishdet.data.build`
- Lint/types (after `pip install -e ".[dev]"`): `ruff check src tests`, `mypy`

## Architecture
`src/smishdet/`: `domain` (pure types) <- `data`, `models`, `evaluation` <- `serving`. Dependencies point inward; serving and evaluation use the `Classifier` protocol in `models/base.py`, never concrete models. Hyperparameters live in `configs/` YAML, not in code.

## Data and privacy rules
- Never commit anything under `data/` (gitignored). Raw messages may contain PII.
- Run `smishdet.data.pii.scrub` before storing or sharing any message.
- Split by `template_group`, never randomly, to prevent leakage.
- Report real and synthetic data performance separately.
- Check each dataset's licence before use; downloads go to `data/raw/` only after that.

## Code standards
- Python >= 3.10, type hints, ruff line length 100, small pure functions.
- New data transforms need tests in `tests/`.
- Use Python 3.11 or 3.12 for training (PyTorch wheels may not exist for newer versions).
