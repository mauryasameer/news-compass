# GOVERNANCE.md

## Intended Use

NewsCompass supports a content-platform reviewer's or product manager's decisions —
surfacing candidate article recommendations for a given user alongside the scores and
reasoning behind them. It does not autonomously publish, push notifications for, or place
recommendations in front of end users. Every recommendation is decision support, to be
reviewed before any action is taken.

## Explainability Boundary

Cosine similarity (user-CF/item-CF), TF-IDF similarity (content-based), and ALS latent
factors are all directly inspectable numeric scores — not opaque model outputs. The
narrative section attached to each recommendation is an LLM-generated plain-English
commentary on that score and title, not an independently verified claim and not the
ranking signal itself — this boundary is stated in the report itself (a permanent
banner), not only here.

## Fairness

The dataset carries no individual-level demographic data (inputs are interaction
type/timestamp, item id, and article text) — a disparate-impact fairness analysis in the
protected-attribute sense does not apply. Two real, documented limitations instead:

- **Popularity bias**: collaborative-filtering strategies (user-CF, item-CF, ALS) tend to
  over-recommend already-popular articles, since they're the ones with the most
  interaction signal. This is a known property of CF, not a solved problem here.
- **Cold-start**: a user or article with no interaction history has no CF signal at all;
  every provider falls back to plain popularity ranking (`popularity_rank()`) rather than
  producing an empty or crashing result, but that fallback carries no personalization.

## LLM Controls

The narrative call passes `temperature=0` to the configured LLM provider, so a given
recommendation's narrative is reproducible rather than randomly sampled. The inputs
reaching the prompt are the recommended article's title and its numeric relevance score.
The title is not user-typed at request time, but it is not curated or trusted input
either: article titles are third-party-authored free text scraped from the open web (real
news article titles from arbitrary sources), and they reach `NARRATIVE_PROMPT` verbatim in
`narrative_service.py`. So the prompt-injection surface is not "no free text reaches the
model" — it is assessed as low-to-moderate: there is no tool-use and no mechanism for the
LLM's output to trigger further action beyond being displayed as commentary text in a
report, so the worst case of a hostile or adversarially-crafted title is a weird-looking
narrative sentence, not code or script execution. (The HTML report output itself is
separately protected by `html.escape()` on all rendered content — a different,
already-verified control, not a property of the LLM call.)

## Audit Trail

Every generated report embeds a "Run Info" block recording the strategy used, the user id
(when applicable), and a timestamp — so any given report is traceable to what produced it.
When run with `--eval`, the report also includes the real Precision@K/MAP@K/RMSE numbers
for that strategy against the full interaction history.

**RMSE caveat**: the RMSE figure pairs each held-out interaction's imputed rating (scale
~1.2-51, depending on how rare that interaction type is) against the provider's raw
similarity/relevance score for that item (a scale that varies by provider and, for the
CF providers, with how many items the user has rated — anywhere from single digits to the
hundreds). These are two incommensurable scales, not a calibrated prediction-vs-actual
pair, so the resulting number (384.18 on the real local dataset, for example) is not a
normalized prediction-error metric in the usual RMSE sense. Read it only as a rough
magnitude indicator for a single strategy run, and do not compare RMSE across strategies
with different score scales — content-based TF-IDF scores are bounded 0-1, ALS scores are
unbounded, and CF similarity-weighted sums scale with rated-item count.

## Regulatory Framing

NewsCompass is a content-recommendation tool, not a financial-services, healthcare, or HR
system — none of SR 11-7, the EU AI Act's high-risk categories, or GDPR's
special-category-data provisions apply to this domain the way they do for a
compliance-evaluation or fraud/credit model. Noted here for completeness: this project
does not currently operate under a specific regulatory framework, and that absence is
itself worth stating rather than leaving unaddressed.
