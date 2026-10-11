"""Graph state schema for the CKD guideline RAG pipeline."""

from typing import Annotated, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class GraphState(TypedDict):
    """State passed between graph nodes.

    Attributes:
        question: The working retrieval query -- the standalone question,
            possibly rewritten mid-graph to improve retrieval.
        original_question: This turn's message exactly as the user typed it.
        standalone_question: What the user is asking, self-contained: equal to
            ``original_question`` on a first turn, or a follow-up condensed
            with the conversation's context ("Should they get an SGLT2i?" ->
            "Should a patient with eGFR 38 and A2 albuminuria get an SGLT2i?").
            Retrieval, grading, and the answer all use this.
        messages: Conversation history (user questions + assistant answers),
            appended each turn; persisted per thread by the checkpointer.
        documents: Chunks retrieved from the vector store for the current question.
        graded_documents: Documents that passed the relevance grading step.
        rewrite_count: Number of times the query has been rewritten, capped at 1.
        answer: The final synthesized, cited answer.
        generation_documents: Exactly the passages the generate prompt received
            (graded chunks plus any always-included reference passages) -- what
            the eval judges the answer against.
        user_context: Known facts about the user (role, expertise level,
            recurring interests) recalled from memory, formatted for the
            generate prompt.
    """

    question: str
    original_question: str
    standalone_question: str
    messages: Annotated[list[AnyMessage], add_messages]
    documents: list[Document]
    graded_documents: list[Document]
    rewrite_count: int
    answer: str
    generation_documents: list[Document]
    user_context: str
