"""Unit tests for the Streamlit client's display logic -- no server, no Streamlit."""

from app.ui_helpers import GUIDELINE_FILTERS, pipeline_summary, source_label, sources_from_state
from kdigo_guideline_rag.configuration import GUIDELINE_FILTER_MAP, Configuration

REC = {
    "page_content": "Recommendation 3.4.1: We suggest ... SBP < 120 mm Hg (2B).",
    "metadata": {
        "source_guideline": "KDIGO 2024 CKD Evaluation and Management",
        "chapter": "Chapter 3",
        "label_type": "Recommendation",
        "recommendation_number": "3.4.1",
        "page": 44,
    },
}
FIGURE_SERIALIZED = {
    "lc": 1,
    "type": "constructor",
    "id": ["langchain", "schema", "document", "Document"],
    "kwargs": {
        "page_content": "Figure 48| Circumstances for referral\n\n• eGFR <30",
        "metadata": {"source_guideline": "KDIGO 2024 CKD Evaluation and Management", "label_type": "Figure"},
    },
}
REFERENCE = {
    "page_content": "Table 47| KDIGO nomenclature",
    "metadata": {
        "source_guideline": "KDIGO 2024 CKD Evaluation and Management",
        "label_type": "Table",
        "reference": True,
    },
}


def test_ui_filters_match_graph_configuration() -> None:
    """The UI env can't import the graph package, so its copy must not drift."""
    from typing import get_args

    allowed = set(get_args(Configuration.__annotations__["guideline_filter"]))
    assert set(GUIDELINE_FILTERS.values()) == allowed
    assert set(GUIDELINE_FILTERS.values()) - {"all"} == set(GUIDELINE_FILTER_MAP)


def test_source_label_for_recommendation_and_figure() -> None:
    assert source_label(REC["page_content"], REC["metadata"]) == (
        "KDIGO 2024 CKD Evaluation and Management, Chapter 3, Recommendation 3.4.1"
    )
    kw = FIGURE_SERIALIZED["kwargs"]
    assert source_label(kw["page_content"], kw["metadata"]).endswith("Figure 48")


def test_sources_handle_both_serializations_and_flag_references() -> None:
    sources = sources_from_state({"generation_documents": [REC, FIGURE_SERIALIZED, REFERENCE]})
    assert [s.reference for s in sources] == [False, False, True]
    assert sources[0].page == 44
    assert sources[1].text.startswith("Figure 48|")


def test_not_covered_answer_has_no_sources() -> None:
    assert sources_from_state({"answer": "not covered", "generation_documents": []}) == []


def test_interpreted_as_only_for_condensed_follow_ups() -> None:
    from app.ui_helpers import interpreted_as

    assert (
        interpreted_as({"original_question": "What is CKD?", "standalone_question": "What is CKD?"}) is None
    )
    follow_up = {
        "original_question": "What about in children?",
        "standalone_question": "What BP target for children with CKD?",
    }
    assert interpreted_as(follow_up) == "What BP target for children with CKD?"


def test_pipeline_summary_shows_rewrite_only_when_it_happened() -> None:
    same = pipeline_summary(
        {
            "question": "q",
            "original_question": "q",
            "standalone_question": "q",
            "documents": [1, 2],
            "graded_documents": [1],
        }
    )
    assert (
        same["rewritten retrieval query"] is None
        and same["interpreted as"] is None
        and same["retrieved"] == 2
    )
    rewritten = pipeline_summary(
        {"question": "better q", "original_question": "q", "standalone_question": "q"}
    )
    assert rewritten["rewritten retrieval query"] == "better q"
