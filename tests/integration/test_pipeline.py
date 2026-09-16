import pandas as pd
from meerax.llm.base import LLMResponse

from src.services.data_service import load_interactions
from src.services.eval_service import run_full_evaluation
from src.services.narrative_service import generate_narratives
from src.services.recommend_service import build_provider, recommend_for_user
from src.services.report_service import build_report


class _StubLLM:
    def generate(self, prompt, system=None, **kwargs):
        return LLMResponse(content="stub narrative", model="stub", input_tokens=1, output_tokens=1)

    def chat(self, messages, system=None, **kwargs):
        return self.generate(messages[-1]["content"])


def _make_synthetic_csvs(tmp_path):
    consumer_path = tmp_path / "consumer.csv"
    pd.DataFrame(
        {
            "event_timestamp": range(20),
            "interaction_type": (["content_watched"] * 15 + ["content_liked"] * 5),
            "item_id": [i % 5 for i in range(20)],
            "consumer_id": [i % 4 for i in range(20)],
            "consumer_session_id": [1] * 20,
            "consumer_device_info": [""] * 20,
            "consumer_location": [""] * 20,
            "country": [""] * 20,
        }
    ).to_csv(consumer_path, index=False)

    content_path = tmp_path / "content.csv"
    pd.DataFrame(
        {
            "event_timestamp": range(5),
            "interaction_type": ["content_present"] * 5,
            "item_id": range(5),
            "producer_id": [1] * 5,
            "producer_session_id": [1] * 5,
            "producer_device_info": [""] * 5,
            "producer_location": [""] * 5,
            "producer_country": [""] * 5,
            "item_type": ["HTML"] * 5,
            "item_url": [f"http://x/{i}" for i in range(5)],
            "title": [f"Article {i}" for i in range(5)],
            "text_description": [f"description of article {i} topic" for i in range(5)],
            "language": ["en"] * 5,
        }
    ).to_csv(content_path, index=False)

    return str(consumer_path), str(content_path)


def test_full_pipeline_produces_nonempty_report(tmp_path):
    consumer_csv, content_csv = _make_synthetic_csvs(tmp_path)
    data = load_interactions(consumer_csv, content_csv)

    provider = build_provider("item_cf", hybrid_spec=None)
    provider.fit(data.ratings)

    user_id = next(iter(data.user_id_map.values()))
    recommendations = recommend_for_user(data, provider, user_id=user_id, k=3)
    narratives = generate_narratives(recommendations, _StubLLM())
    eval_result = run_full_evaluation(data.ratings, provider, k=3)

    report = build_report(
        "NewsCompass Test Report",
        recommendations,
        narratives,
        strategy_name="item_cf",
        user_id=user_id,
        eval_result=eval_result,
    )

    output_path = tmp_path / "report.html"
    report.save(str(output_path))

    assert output_path.exists()
    assert output_path.stat().st_size > 0
