from __future__ import annotations

import html
from datetime import UTC, datetime

from meerax.report.builder import ReportBuilder, ReportSection

from src.services.eval_service import EvalResult

GOVERNANCE_BANNER = (
    "GOVERNANCE NOTICE: This report supports a content-platform reviewer's decisions; it "
    "does not autonomously place or auto-publish recommendations. Per-article narratives "
    "are LLM-generated commentary on the numeric ranking above them, not independently "
    "verified claims and not the ranking signal itself."
)


def _run_info_html(strategy_name: str, user_id: int | None, timestamp: str) -> str:
    user_part = f"user_id: {user_id} | " if user_id is not None else ""
    return (
        f"<p class='meta'>Run Info &mdash; strategy: {html.escape(strategy_name)} | "
        f"{user_part}generated: {html.escape(timestamp)}</p>"
    )


def build_report(
    title: str,
    recommendations: list[dict],
    narratives: dict[int, str],
    strategy_name: str,
    user_id: int | None,
    eval_result: EvalResult | None = None,
) -> ReportBuilder:
    timestamp = datetime.now(UTC).isoformat(timespec="seconds")
    # ReportBuilder.to_html() does not escape the report title itself (only section
    # content is escaped by callers) — every call site here passes a hardcoded literal
    # today, but escape defensively so this stays safe if title is ever wired to
    # data-derived input later.
    report = ReportBuilder(html.escape(title), subtitle=GOVERNANCE_BANNER)

    for rec in recommendations:
        narrative = narratives.get(rec["item_id"], "")
        content = html.escape(narrative) if narrative else ""
        report.add_section(
            ReportSection(
                title=html.escape(rec["title"]),
                content=content,
                metrics={"score": rec["score"]},
            )
        )

    if eval_result is not None:
        rmse_caveat = (
            "Note: RMSE compares imputed-rating scale against raw provider-score scale — "
            "it is not a normalized prediction-error metric. Treat it as a rough magnitude "
            "indicator only, and do not compare it across strategies with different score "
            "scales."
        )
        report.add_section(
            ReportSection(
                title="Evaluation",
                content=html.escape(rmse_caveat),
                metrics={
                    **eval_result.recommender_metrics.to_dict(),
                    "rmse": eval_result.rmse,
                },
            )
        )

    report.add_section(
        ReportSection(title="Run Info", content=_run_info_html(strategy_name, user_id, timestamp))
    )
    return report
