"""Unit tests for the eval response cache -- no API calls."""

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.globals import set_llm_cache
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration

from kdigo_guideline_rag.eval import cache
from kdigo_guideline_rag.eval.judge import AnswerVerdict, judge_answer


@pytest.fixture
def store(tmp_path: Path) -> Iterator[cache.ResponseStore]:
    yield cache.enable(tmp_path / "responses.sqlite")
    set_llm_cache(None)
    cache._store = None


def test_identical_chat_call_replays_and_a_changed_prompt_misses(store: cache.ResponseStore) -> None:
    model = FakeListChatModel(responses=["first", "second", "third"])
    assert model.invoke("grade this").content == "first"
    assert model.invoke("grade this").content == "first"  # replayed, not the next fake response
    assert model.invoke("grade that").content == "second"
    assert cache.llm_counts() == (1, 3)


def test_tool_calls_survive_the_round_trip(store: cache.ResponseStore) -> None:
    message = AIMessage(
        content="", tool_calls=[{"name": "GradeResult", "args": {"relevant": True}, "id": "t1"}]
    )
    llm_cache = cache.LLMCache(store)
    llm_cache.update("prompt", "llm", [ChatGeneration(message=message)])
    (replayed,) = llm_cache.lookup("prompt", "llm")
    assert replayed.message.tool_calls == message.tool_calls


class FakeJudgeClient:
    """Stands in for ``anthropic.Anthropic``; counts parse calls."""

    def __init__(self) -> None:
        self.calls = 0
        verdict = AnswerVerdict(
            unsupported_claims=[],
            transferred_claims=[],
            addresses_question=True,
            declined=False,
            wrongly_declined=False,
            outside_knowledge=False,
            asks_user_for_context=False,
            note="Supported.",
        )
        self.beta = SimpleNamespace(messages=SimpleNamespace(parse=self._parse))
        self._response = SimpleNamespace(stop_reason="end_turn", parsed_output=verdict)

    def _parse(self, **_: object) -> SimpleNamespace:
        self.calls += 1
        return self._response


ROW = {
    "id": "q1",
    "question": "What is the SBP target?",
    "category": "recommendation",
    "scope": None,
    "context": "[KDIGO 2021 Blood Pressure in CKD, Chapter 3, Recommendation 3.1.1]\nSBP < 120 mm Hg (2B).",
    "answer": "Target SBP < 120 mm Hg [KDIGO 2021 Blood Pressure in CKD, Chapter 3, Recommendation 3.1.1].",
    "cited_passages": [["KDIGO 2021 Blood Pressure in CKD, Chapter 3, Recommendation 3.1.1", "SBP < 120"]],
}


def test_judge_replays_an_identical_request_and_rejudges_a_changed_answer(store: cache.ResponseStore) -> None:
    client = FakeJudgeClient()
    first, second = judge_answer(ROW, client), judge_answer(ROW, client)
    assert client.calls == 1 and (first["cached"], second["cached"]) == (False, True)
    assert second["verdict"] == first["verdict"] and second["verdict"]["materially_faithful"]
    judge_answer({**ROW, "answer": ROW["answer"] + " Also check potassium."}, client)
    assert client.calls == 2


def test_judge_calls_fresh_when_caching_is_off() -> None:
    client = FakeJudgeClient()
    judge_answer(ROW, client)
    judge_answer(ROW, client)
    assert client.calls == 2
    assert cache.replay_note(0, 2) == "cache: off (--no-cache)"
