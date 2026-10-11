"""Unit tests for node functions that don't require live API calls."""

from unittest.mock import patch

from langchain_core.documents import Document

from kdigo_guideline_rag.nodes import GradeResult, grade


class _FakeGrader:
    def __init__(self, results):
        self._results = results

    def batch(self, inputs, return_exceptions=False):
        assert return_exceptions is True
        return self._results


class _FakeLLM:
    def __init__(self, results):
        self._results = results

    def with_structured_output(self, model):
        return _FakeGrader(self._results)


def _docs(n):
    return [Document(page_content=f"chunk {i}") for i in range(n)]


def test_grade_treats_exception_as_not_relevant():
    """A malformed structured-output call for one chunk (observed in
    production: Claude nesting the whole JSON payload as a string inside a
    single field instead of proper top-level args) must not crash the whole
    grading pass -- that chunk is dropped, the rest of the batch still
    grades normally.
    """
    docs = _docs(3)
    results = [
        GradeResult(relevant=True, reasoning="on topic"),
        ValueError("malformed tool call"),
        GradeResult(relevant=False, reasoning="off topic"),
    ]
    with patch("kdigo_guideline_rag.nodes.get_llm", return_value=_FakeLLM(results)):
        out = grade({"question": "q", "documents": docs})

    assert out["graded_documents"] == [docs[0]]


def test_grade_returns_empty_list_for_no_documents():
    with patch("kdigo_guideline_rag.nodes.get_llm", return_value=_FakeLLM([])):
        out = grade({"question": "q", "documents": []})

    assert out["graded_documents"] == []


def test_cite_names_figure_number_for_figure_chunks() -> None:
    from langchain_core.documents import Document

    from kdigo_guideline_rag.nodes import _cite

    doc = Document(
        page_content="Figure 48| Circumstances for referral\n\n• eGFR <30",
        metadata={"source_guideline": "KDIGO 2024 CKD", "chapter": "Chapter 5", "label_type": "Figure"},
    )
    assert _cite(doc) == "KDIGO 2024 CKD, Chapter 5, Figure 48"


def test_generate_without_relevant_chunks_returns_fixed_response_without_llm():
    """Structural guarantee: with nothing to cite, no model call can volunteer outside knowledge."""
    from kdigo_guideline_rag.nodes import generate

    with patch("kdigo_guideline_rag.nodes.get_llm", side_effect=AssertionError("LLM must not be called")):
        out = generate({"question": "ESA hemoglobin target?", "graded_documents": []}, {"configurable": {}})
    assert "don't address this question" in out["answer"]
    assert "KDIGO 2021 Blood Pressure in CKD" in out["answer"]


def test_generate_fixed_response_names_only_the_filtered_guideline():
    from kdigo_guideline_rag.nodes import generate

    out = generate(
        {"question": "q", "graded_documents": []}, {"configurable": {"guideline_filter": "aki_only"}}
    )
    assert "AKI/AKD" in out["answer"] and "Glomerular" not in out["answer"]


def test_generate_answers_original_question_not_the_rewrite():
    from kdigo_guideline_rag.nodes import generate

    captured = {}

    class _CapturingLLM:
        def invoke(self, messages):
            captured["user"] = messages[-1][1]
            return type("R", (), {"text": "answer"})()

    state = {
        "question": "rewritten retrieval-friendly query",
        "original_question": "what the user actually asked",
        "graded_documents": _docs(1),
    }
    with (
        patch("kdigo_guideline_rag.nodes.get_llm", return_value=_CapturingLLM()),
        patch("kdigo_guideline_rag.nodes.reference_documents", return_value=()),
    ):
        generate(state, {"configurable": {}})
    assert "what the user actually asked" in captured["user"]
    assert "rewritten retrieval-friendly query" not in captured["user"]


def test_generate_appends_reference_passages_once_and_records_them():
    """Grading tables ride along with every answer (so grade explanations are
    grounded), are not duplicated if retrieval already found them, and are
    recorded as part of what the answer is judged against."""
    from kdigo_guideline_rag.nodes import generate

    ref = Document(
        page_content="Table 47| KDIGO nomenclature", metadata={"source_guideline": "KDIGO 2024 CKD"}
    )

    class _LLM:
        def invoke(self, messages):
            return type("R", (), {"text": "answer"})()

    with (
        patch("kdigo_guideline_rag.nodes.get_llm", return_value=_LLM()),
        patch("kdigo_guideline_rag.nodes.reference_documents", return_value=(ref,)),
    ):
        out = generate({"question": "q", "graded_documents": _docs(2) + [ref]}, {"configurable": {}})
        out2 = generate({"question": "q", "graded_documents": _docs(2)}, {"configurable": {}})
    assert [d.page_content for d in out["generation_documents"]].count(ref.page_content) == 1
    assert out2["generation_documents"][-1] is ref


def test_prepare_records_the_users_message_and_resets_turn_state():
    from langchain_core.messages import HumanMessage

    from kdigo_guideline_rag.nodes import prepare

    out = prepare({"question": "Should they get an SGLT2i?", "rewrite_count": 1})
    assert out["original_question"] == out["standalone_question"] == "Should they get an SGLT2i?"
    assert out["rewrite_count"] == 0
    assert out["messages"] == [HumanMessage(content="Should they get an SGLT2i?")]


def test_condense_passes_first_turn_through_without_llm():
    from langchain_core.messages import HumanMessage

    from kdigo_guideline_rag.nodes import condense

    state = {"original_question": "What is CKD?", "messages": [HumanMessage(content="What is CKD?")]}
    with patch("kdigo_guideline_rag.nodes.get_llm", side_effect=AssertionError("no LLM on a first turn")):
        out = condense(state)
    assert out == {"standalone_question": "What is CKD?", "question": "What is CKD?"}


def test_condense_rewrites_follow_up_from_history_excluding_current_turn():
    from langchain_core.messages import AIMessage, HumanMessage

    from kdigo_guideline_rag.nodes import condense

    seen = {}

    class _LLM:
        def invoke(self, messages):
            seen["user"] = messages[-1][1]
            return type(
                "R", (), {"text": "Should a patient with eGFR 38 and A2 albuminuria get an SGLT2i?"}
            )()

    state = {
        "original_question": "Should they get an SGLT2i?",
        "messages": [
            HumanMessage(content="My patient has eGFR 38 and A2 albuminuria."),
            AIMessage(content="x" * 1000),
            HumanMessage(content="Should they get an SGLT2i?"),
        ],
    }
    with patch("kdigo_guideline_rag.nodes.get_llm", return_value=_LLM()):
        out = condense(state)
    assert (
        out["question"]
        == out["standalone_question"]
        == "Should a patient with eGFR 38 and A2 albuminuria get an SGLT2i?"
    )
    history, latest = seen["user"].split("Latest message:")
    assert "User: My patient has eGFR 38" in history
    assert "Should they get an SGLT2i?" not in history  # current turn isn't history
    assert "x" * 401 not in history  # earlier answers truncated
    assert latest.strip() == "Should they get an SGLT2i?"


def test_condense_history_window_keeps_only_recent_turns():
    from langchain_core.messages import AIMessage, HumanMessage

    from kdigo_guideline_rag.nodes import CONDENSE_HISTORY_TURNS, _format_history

    msgs = []
    for i in range(CONDENSE_HISTORY_TURNS + 2):
        msgs += [HumanMessage(content=f"q{i}"), AIMessage(content=f"a{i}")]
    text = _format_history(msgs)
    assert "User: q0" not in text and f"User: q{CONDENSE_HISTORY_TURNS + 1}" in text


def test_generate_answers_the_standalone_question_and_appends_ai_message():
    from langchain_core.messages import AIMessage

    from kdigo_guideline_rag.nodes import generate

    captured = {}

    class _LLM:
        def invoke(self, messages):
            captured["user"] = messages[-1][1]
            return type("R", (), {"text": "cited answer"})()

    state = {
        "question": "retrieval rewrite",
        "original_question": "Should they get an SGLT2i?",
        "standalone_question": "Should a patient with eGFR 38 get an SGLT2i?",
        "graded_documents": _docs(1),
    }
    with (
        patch("kdigo_guideline_rag.nodes.get_llm", return_value=_LLM()),
        patch("kdigo_guideline_rag.nodes.reference_documents", return_value=()),
    ):
        out = generate(state, {"configurable": {}})
    assert "Should a patient with eGFR 38 get an SGLT2i?" in captured["user"]
    assert "retrieval rewrite" not in captured["user"]
    assert out["messages"] == [AIMessage(content="cited answer")]
