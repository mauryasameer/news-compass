from meerax.llm.base import LLMResponse

from src.services.narrative_service import generate_narratives


class _StubLLM:
    def __init__(self, content: str = "stub narrative") -> None:
        self._content = content

    def generate(self, prompt, system=None, **kwargs):
        assert kwargs.get("temperature") == 0.0
        return LLMResponse(content=self._content, model="stub", input_tokens=1, output_tokens=1)

    def chat(self, messages, system=None, **kwargs):
        return self.generate(messages[-1]["content"])


class _FailingLLM:
    def generate(self, prompt, system=None, **kwargs):
        raise RuntimeError("boom")

    def chat(self, messages, system=None, **kwargs):
        return self.generate(messages[-1]["content"])


def test_generates_one_narrative_per_recommendation():
    recs = [
        {"item_id": 1, "title": "A", "score": 0.9},
        {"item_id": 2, "title": "B", "score": 0.5},
    ]
    result = generate_narratives(recs, _StubLLM())
    assert result[1] == "stub narrative"
    assert result[2] == "stub narrative"


def test_llm_failure_falls_back_to_placeholder():
    recs = [{"item_id": 1, "title": "A", "score": 0.9}]
    result = generate_narratives(recs, _FailingLLM())
    assert result[1] == "narrative unavailable"


def test_empty_recommendations_returns_empty_dict():
    assert generate_narratives([], _StubLLM()) == {}
