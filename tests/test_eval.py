"""Unit tests for eval retrieval metrics -- no API calls."""

from langchain_core.documents import Document

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
