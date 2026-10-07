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
