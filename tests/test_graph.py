"""Graph routing logic (unit) and full end-to-end invocation (integration)."""

import os

import pytest

from kdigo_guideline_rag.nodes import MAX_REWRITES, route_after_grade


def _state(graded_documents, rewrite_count):
    return {"graded_documents": graded_documents, "rewrite_count": rewrite_count}


def test_routes_to_generate_when_chunks_survived_grading():
    assert route_after_grade(_state(graded_documents=["a chunk"], rewrite_count=0)) == "generate"


def test_routes_to_rewrite_when_no_relevant_chunks_and_retries_remain():
    assert route_after_grade(_state(graded_documents=[], rewrite_count=0)) == "rewrite"


def test_routes_to_generate_after_rewrite_budget_exhausted():
    """No relevant chunks even after a rewrite -- must not loop forever.
    Generates anyway so the model can say explicitly it found nothing.
    """
    assert route_after_grade(_state(graded_documents=[], rewrite_count=MAX_REWRITES)) == "generate"


HAS_LIVE_KEYS = bool(os.environ.get("ANTHROPIC_API_KEY")) and bool(os.environ.get("VOYAGE_API_KEY"))


@pytest.mark.integration
@pytest.mark.skipif(not HAS_LIVE_KEYS, reason="requires ANTHROPIC_API_KEY and VOYAGE_API_KEY")
def test_end_to_end_query_produces_cited_answer():
    from kdigo_guideline_rag.graph import graph

    result = graph.invoke({"question": "Should a patient with G3b A2 CKD be on an SGLT2 inhibitor?"})

    assert result["answer"]
    assert "KDIGO" in result["answer"]
    assert result["rewrite_count"] <= MAX_REWRITES


@pytest.mark.integration
@pytest.mark.skipif(not HAS_LIVE_KEYS, reason="requires ANTHROPIC_API_KEY and VOYAGE_API_KEY")
def test_user_context_persists_across_invocations_for_same_user():
    from kdigo_guideline_rag.graph import graph

    config = {"configurable": {"user_id": "pytest-user"}}
    graph.invoke({"question": "As a nephrology fellow, what is the CKD definition?"}, config=config)
    second = graph.invoke({"question": "What about referral criteria?"}, config=config)

    assert "nephrology fellow" in second["user_context"]
