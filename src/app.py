from __future__ import annotations

import argparse
import sys

from meerax.llm.claude import ClaudeProvider
from meerax.llm.ollama import OllamaProvider
from meerax.llm.openai_provider import OpenAIProvider

from src.services.data_service import load_interactions
from src.services.eval_service import run_full_evaluation
from src.services.narrative_service import generate_narratives
from src.services.recommend_service import build_provider, recommend_for_user, set_articles_if_needed
from src.services.report_service import build_report

LLM_PROVIDERS: dict[str, type] = {
    "ollama": OllamaProvider,
    "claude": ClaudeProvider,
    "openai": OpenAIProvider,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="news-compass")
    parser.add_argument("--consumer-csv", default="src/data/consumer_transanctions.csv")
    parser.add_argument("--content-csv", default="src/data/platform_content.csv")
    parser.add_argument(
        "--strategy", choices=["user_cf", "item_cf", "content", "als", "hybrid"], default="item_cf"
    )
    parser.add_argument("--hybrid", default=None, help='e.g. "item_cf:0.5,als:0.5"')
    parser.add_argument("--user-id", type=int, default=None)
    parser.add_argument("--seed-title", default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--llm-provider", choices=list(LLM_PROVIDERS.keys()), default="ollama")
    parser.add_argument("--output", default="reports/news_compass_report.html")
    parser.add_argument("--eval", action="store_true", help="run full evaluation and include it in the report")
    args = parser.parse_args(argv)

    if args.top_k <= 0:
        print(f"error: --top-k must be positive, got {args.top_k}", file=sys.stderr)
        return 1

    try:
        data = load_interactions(args.consumer_csv, args.content_csv)
    except FileNotFoundError as exc:
        print(f"error: could not read data file: {exc}", file=sys.stderr)
        return 1

    try:
        provider = build_provider(args.strategy, args.hybrid)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    provider.fit(data.ratings)

    if args.strategy == "content" and args.seed_title:
        from src.providers.content_provider import ContentBasedProvider

        assert isinstance(provider, ContentBasedProvider)
        provider.set_articles(data.articles)
        raw = provider.recommend_similar(args.seed_title, k=args.top_k)
        articles_by_id = data.articles.set_index("item_id")
        recommendations = [
            {
                "item_id": item_id,
                "title": articles_by_id.loc[item_id]["title"] if item_id in articles_by_id.index else f"item {item_id}",
                "item_url": articles_by_id.loc[item_id]["item_url"] if item_id in articles_by_id.index else "",
                "score": score,
            }
            for item_id, score in raw
        ]
        user_id = None
    else:
        set_articles_if_needed(provider, data.articles)
        user_id = args.user_id if args.user_id is not None else next(iter(data.user_id_map.values()))
        recommendations = recommend_for_user(data, provider, user_id=user_id, k=args.top_k)

    llm = LLM_PROVIDERS[args.llm_provider]()
    narratives = generate_narratives(recommendations, llm)

    eval_result = run_full_evaluation(data.ratings, provider, k=args.top_k) if args.eval else None

    report = build_report(
        "NewsCompass — Recommendation Report",
        recommendations,
        narratives,
        strategy_name=args.strategy,
        user_id=user_id,
        eval_result=eval_result,
    )
    report.save(args.output)
    print(f"report written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
