# Changelog

All notable changes to this project will be documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.1.2] - 2026-09-17

### Changed
- `implicit` 0.7.2 → 0.7.3 (upstream bug fixes: nan checks in BPR GPU path, an overflow
  warning in nearest-neighbors, a memory-access fix). Verified against the real package,
  not just CI's green check (the shared CI never installs `requirements-als.txt`, so it
  can't validate this on its own): real ALS unit tests pass, full suite 49/49, and an
  end-to-end `--strategy als` CLI run against real data succeeds.

## [0.1.1] - 2026-09-17

### Fixed
- `scikit-learn`, `gensim`, and `scipy` were unpinned floors (`>=`) in `requirements.txt`
  — a PROJECT_GUIDELINES violation for anything deployed. Pinned exact to the versions
  actually tested throughout this build (`scikit-learn==1.9.0`, `gensim==4.4.0`,
  `scipy==1.18.1`).

## [0.1.0] - 2026-09-15

### Added
- Four recommender strategies behind one `RecommenderProvider` interface — user-CF,
  item-CF, content-based TF-IDF, and ALS — plus a configurable weighted hybrid blend
  (`--hybrid "item_cf:0.5,als:0.5"`) that composes any subset of them.
- GenAI narrative layer (`narrative_service.py`) generating a plain-English, per-article
  explanation for each recommendation via `meerax.llm` (Ollama/Claude/OpenAI), with
  `temperature=0` for reproducibility and graceful per-item fallback on LLM failure.
- Self-contained HTML report (`report_service.py`) via `meerax.report.builder`, with a
  permanent governance banner, an escaped Run Info audit block (strategy, user id,
  timestamp), and an optional Evaluation section (`--eval`) with real Precision@K/MAP@K
  (`meerax.eval.recommender`) and RMSE (`meerax.eval.timeseries`) against the full
  interaction history.
- `scripts/fetch_data.py` — Kaggle "Articles sharing/reading from CI&T Deskdrop" dataset
  fetch and transform into the column schema this app reads, fails loudly without
  credentials.
- Docker packaging (`Dockerfile`, `docker-compose.yml`).
- `GOVERNANCE.md` — intended use, explainability boundary, fairness scope (popularity
  bias, cold-start), LLM controls, audit trail, regulatory framing.
- CI, dependency pins, and initial repo layout.
