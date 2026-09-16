from __future__ import annotations

from src.core.interfaces import RecommenderProvider
from src.providers.hybrid_provider import HybridProvider
from src.providers.item_cf_provider import ItemCFProvider
from src.providers.user_cf_provider import UserCFProvider
from src.services.data_service import ProcessedData

PROVIDERS: dict[str, type[RecommenderProvider]] = {
    "user_cf": UserCFProvider,
    "item_cf": ItemCFProvider,
}


def _parse_hybrid_spec(spec: str) -> dict[str, float]:
    weights: dict[str, float] = {}
    for pair in spec.split(","):
        name, weight = pair.split(":")
        weights[name.strip()] = float(weight)
    return weights


def build_provider(strategy: str, hybrid_spec: str | None) -> RecommenderProvider:
    if strategy == "hybrid":
        if not hybrid_spec:
            raise ValueError("--hybrid spec is required when --strategy hybrid is used")
        weights = _parse_hybrid_spec(hybrid_spec)
        members: dict[str, RecommenderProvider] = {}
        for name in weights:
            if name == "content":
                from src.providers.content_provider import ContentBasedProvider

                members[name] = ContentBasedProvider()
            elif name == "als":
                from src.providers.als_provider import ALSProvider

                members[name] = ALSProvider()
            elif name in PROVIDERS:
                members[name] = PROVIDERS[name]()
            else:
                raise ValueError(f"unknown hybrid member: {name}")
        return HybridProvider(providers=members, weights=weights)

    if strategy == "content":
        from src.providers.content_provider import ContentBasedProvider

        return ContentBasedProvider()

    if strategy == "als":
        from src.providers.als_provider import ALSProvider

        return ALSProvider()

    if strategy in PROVIDERS:
        return PROVIDERS[strategy]()

    raise ValueError(f"unknown strategy: {strategy}")


def recommend_for_user(
    data: ProcessedData,
    provider: RecommenderProvider,
    user_id: int,
    k: int,
) -> list[dict]:
    raw = provider.recommend(user_id=user_id, k=k)
    articles_by_id = data.articles.set_index("item_id")
    results = []
    for item_id, score in raw:
        row = articles_by_id.loc[item_id] if item_id in articles_by_id.index else None
        results.append(
            {
                "item_id": item_id,
                "title": row["title"] if row is not None else f"item {item_id}",
                "item_url": row["item_url"] if row is not None else "",
                "score": score,
            }
        )
    return results


def set_articles_if_needed(provider: RecommenderProvider, articles) -> None:
    """ContentBasedProvider needs set_articles() before use. HybridProvider has no way
    to know about that call, so this walks into any hybrid's members and forwards it —
    without this, a ContentBasedProvider nested inside a HybridProvider would raise
    RuntimeError the first time the hybrid asked it for recommendations."""
    if hasattr(provider, "set_articles"):
        provider.set_articles(articles)
    if isinstance(provider, HybridProvider):
        for member in provider.members.values():
            set_articles_if_needed(member, articles)
