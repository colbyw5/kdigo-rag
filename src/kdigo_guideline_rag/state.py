"""Graph state schema for the CKD guideline RAG pipeline."""

from typing import TypedDict

from langchain_core.documents import Document


class GraphState(TypedDict):
    """State passed between graph nodes.

    Attributes:
        question: The current user question (may be rewritten mid-graph).
        original_question: The question as originally asked by the user.
        documents: Chunks retrieved from the vector store for the current question.
        graded_documents: Documents that passed the relevance grading step.
        rewrite_count: Number of times the query has been rewritten, capped at 1.
        answer: The final synthesized, cited answer.
    """

    question: str
    original_question: str
    documents: list[Document]
    graded_documents: list[Document]
    rewrite_count: int
    answer: str
