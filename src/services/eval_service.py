from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from meerax.eval.recommender import RecommenderMetrics, evaluate_recommender
from meerax.eval.timeseries import evaluate_forecast

from src.core.interfaces import RecommenderProvider


@dataclass
class EvalResult:
    recommender_metrics: RecommenderMetrics
    rmse: float


def run_full_evaluation(ratings: pd.DataFrame, provider: RecommenderProvider, k: int) -> EvalResult:
    """Precision@K/MAP@K (holding out each user's rated items as 'relevant') and RMSE
    (predicted score vs. actual rating for each user's held-out items)."""
    recommended: list[list[int]] = []
    relevant: list[set[int]] = []
    y_true: list[float] = []
    y_pred: list[float] = []

    for user_id in sorted(ratings["user_id"].unique()):
        user_items = ratings[ratings.user_id == user_id]
        relevant_items = set(user_items["item_id"])
        ranked = provider.recommend(user_id=int(user_id), k=k, exclude_seen=False)
        recommended.append([item_id for item_id, _ in ranked])
        relevant.append(relevant_items)

        scores_by_item = dict(ranked)
        for row in user_items.itertuples():
            if row.item_id in scores_by_item:
                y_true.append(row.rating)
                y_pred.append(scores_by_item[row.item_id])

    recommender_metrics = evaluate_recommender(recommended, relevant, k=k)
    rmse = (
        evaluate_forecast(np.array(y_true), np.array(y_pred)).rmse if y_true else 0.0
    )
    return EvalResult(recommender_metrics=recommender_metrics, rmse=rmse)
