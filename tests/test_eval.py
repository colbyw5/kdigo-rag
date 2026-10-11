"""Unit tests for eval retrieval metrics -- no API calls."""

from langchain_core.documents import Document

from kdigo_guideline_rag.eval.questions import (
    AKI,
    BP,
    CKD,
    EVAL_QUESTIONS,
    EvalQuestion,
    any_of,
    phrase,
    rec,
    table,
)
from kdigo_guideline_rag.eval.retrieval import matches, score
from kdigo_guideline_rag.eval.runner import RunResult, retrieval_metrics


def _docs(n: int) -> list[Document]:
    return [Document(page_content=f"chunk {i}") for i in range(n)]


def test_retrieval_metrics_reads_final_state() -> None:
    state = {"documents": _docs(5), "graded_documents": _docs(3), "rewrite_count": 1}
    assert retrieval_metrics(state) == (5, 3, 1)


def test_retrieval_metrics_defaults_when_fields_missing() -> None:
    assert retrieval_metrics({}) == (0, 0, 0)


def test_precision_handles_zero_retrieved() -> None:
    r = RunResult(passed=None, answer="", citations=[], retrieved=0, relevant=0, rewrites=0)
    assert r.precision == 0.0


def test_precision_ratio() -> None:
    r = RunResult(passed=True, answer="", citations=[], retrieved=5, relevant=4, rewrites=0)
    assert r.precision == 0.8


def _doc(text: str, guideline: str, label_type: str | None = None, number: str | None = None) -> Document:
    meta = {"source_guideline": guideline, "label_type": label_type, "recommendation_number": number}
    return Document(page_content=text, metadata={k: v for k, v in meta.items() if v is not None})


def test_rec_needle_requires_guideline_and_number() -> None:
    """Numbers collide across guidelines: CKD Rec 3.7.1 is SGLT2i, AKI Rec 3.7.1 is rasburicase."""
    needle = rec(AKI, "3.7.1")
    assert matches(
        needle,
        "Recommendation 3.7.1: rasburicase",
        {"source_guideline": AKI, "label_type": "Recommendation", "recommendation_number": "3.7.1"},
    )
    assert not matches(
        needle,
        "Recommendation 3.7.1: SGLT2i",
        {"source_guideline": CKD, "label_type": "Recommendation", "recommendation_number": "3.7.1"},
    )


def test_phrase_needle_ignores_spacing_ligatures_and_punctuation() -> None:
    needle = phrase(CKD, "less certainty around the benefit")
    assert matches(needle, "there is less certainty around the bene fi t of", {"source_guideline": CKD})


def test_table_needle_ignores_table_of_contents_mention() -> None:
    needle = table(CKD, "Table 2| GFR categories in CKD")
    assert not matches(needle, "S137 Table 2. GFR categories in CKD S137 Table 3.", {"source_guideline": CKD})
    assert matches(
        needle, "Table 2| GFR categories in CKD\n\n| G1 |", {"source_guideline": CKD, "label_type": "Table"}
    )


def test_score_hit_recall_and_reciprocal_rank() -> None:
    q = EvalQuestion(
        id="t",
        question="?",
        category="synthesis",
        split="dev",
        needles=[rec(CKD, "3.4.1"), rec(AKI, "5.1.1")],
    )
    docs = [
        _doc("noise", CKD),
        _doc("Recommendation 3.4.1", CKD, "Recommendation", "3.4.1"),
        _doc("noise", AKI),
    ]
    s = score(q, docs)

    assert s.hit and s.recall == 0.5 and s.reciprocal_rank == 0.5
    assert s.missing == [rec(AKI, "5.1.1")]


def test_eval_set_ids_unique_and_only_out_of_scope_lacks_needles() -> None:
    assert len({q.id for q in EVAL_QUESTIONS}) == len(EVAL_QUESTIONS)
    for q in EVAL_QUESTIONS:
        assert bool(q.needles) == (q.category != "out_of_scope"), q.id
        assert (q.scope is not None) == (q.category == "out_of_scope"), q.id


def test_any_of_satisfied_by_any_option() -> None:
    needle = any_of(
        table(CKD, "Table 2| GFR categories"), table(BP, "G3b Moderately to severely decreased 30-44")
    )
    assert matches(
        needle,
        "| G3b | Moderately to severely decreased | 30-44 |",
        {"source_guideline": BP, "label_type": "Table"},
    )
    assert not matches(
        needle,
        "| G3b | Moderately to severely decreased | 30-44 |",
        {"source_guideline": AKI, "label_type": "Table"},
    )


def test_conversation_context_check_requires_and_forbids_terms() -> None:
    from kdigo_guideline_rag.eval.conversations import CONVERSATIONS
    from kdigo_guideline_rag.eval.multiturn import context_check

    switch = next(c for c in CONVERSATIONS if c.kind == "switch")
    assert context_check(switch, "What defines Stage 2 AKI?") == []
    assert any(
        "leaked" in p
        for p in context_check(
            switch, "For a G3b A2 patient on an SGLT2 inhibitor, what defines Stage 2 AKI?"
        )
    )
    carry = next(c for c in CONVERSATIONS if c.id == "conv_bp_children")
    assert context_check(carry, "What about in children?") != []  # lost the BP topic
    assert context_check(carry, "What blood pressure target is recommended for children with CKD?") == []


def test_conversations_are_well_formed() -> None:
    from kdigo_guideline_rag.eval.conversations import CONVERSATIONS

    assert len({c.id for c in CONVERSATIONS}) == len(CONVERSATIONS)
    for c in CONVERSATIONS:
        assert len(c.turns) >= 2, c.id
        assert bool(c.needles) == (c.kind != "out_of_scope"), c.id
