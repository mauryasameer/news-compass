# NewsCompass

[![Version](https://img.shields.io/badge/version-0.1.2-blue)](CHANGELOG.md)
[![CI](https://github.com/mauryasameer/news-compass/actions/workflows/ci.yml/badge.svg)](https://github.com/mauryasameer/news-compass/actions)
[![Python](https://img.shields.io/badge/python-3.12-3776AB)](https://www.python.org)
[![License](https://img.shields.io/badge/license-MIT-informational)](LICENSE)

Four pluggable news recommenders (user-based CF, item-based CF, content-based, ALS) plus
a configurable weighted hybrid, built on [meerax](https://github.com/mauryasameer/the-forge).
Adds a GenAI narrative layer via `meerax.llm` and renders a self-contained HTML report
(with real Precision@K/MAP@K/RMSE evaluation) via `meerax.report`. Trained on the CI&T
Deskdrop news-portal interaction dataset (1,895 users, 2,987 articles).

## Setup

```bash
pip install -r requirements.txt
# optional, for the ALS strategy:
pip install -r requirements-als.txt
```

## Data

```bash
python scripts/fetch_data.py   # requires a Kaggle API token at ~/.kaggle/kaggle.json
```

Downloads and unzips the Deskdrop dataset into `src/data/` as
`consumer_transanctions.csv` and `platform_content.csv` — the default paths `src/app.py`
reads from.

## Usage

```bash
python -m src.app --strategy item_cf --user-id 17 --top-k 10
python -m src.app --strategy hybrid --hybrid item_cf:0.5,als:0.5 --user-id 17 --top-k 10
python -m src.app --strategy content --seed-title "..." --top-k 10
python -m src.app --eval --strategy item_cf
```

`--strategy` accepts `user_cf`, `item_cf`, `content`, `als`, or `hybrid` (default
`item_cf`). `--llm-provider` accepts `ollama` (default), `claude`, or `openai`. Reports
are written to `reports/news_compass_report.html` by default (`--output` to override).

## Docker

```bash
docker compose build
docker compose run news-compass python -m src.app --strategy item_cf --user-id 17
```

## Governance

See [GOVERNANCE.md](./GOVERNANCE.md) for intended use, the explainability boundary,
fairness limitations, LLM controls, audit trail, and regulatory framing.

## License

Distributed under the MIT License. See `LICENSE` for more information.
