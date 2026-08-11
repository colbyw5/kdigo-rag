"""Graph node functions: retrieve, grade, generate, rewrite."""

from functools import lru_cache

from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from kdigo_guideline_rag.configuration import GUIDELINE_FILTER_MAP, Configuration
from kdigo_guideline_rag.ingest.embed import get_vectorstore
from kdigo_guideline_rag.prompts import (
    GENERATE_SYSTEM_PROMPT,
    GENERATE_USER_PROMPT,
    GRADE_SYSTEM_PROMPT,
    GRADE_USER_PROMPT,
    REWRITE_SYSTEM_PROMPT,
    REWRITE_USER_PROMPT,
)
from kdigo_guideline_rag.state import GraphState

MODEL = "claude-sonnet-5"
TOP_K = 5
MAX_REWRITES = 1


class GradeResult(BaseModel):
    """Structured relevance grade for one retrieved passage."""

    relevant: bool = Field(description="Whether the chunk is relevant to the question")
    reasoning: str = Field(description="Brief explanation of the relevance decision")


@lru_cache(maxsize=1)
def get_llm() -> ChatAnthropic:
    """Shared, cached chat model -- reused by nodes.py and memory.py."""
    return ChatAnthropic(model=MODEL)


def _cite(doc: Document) -> str:
    m = doc.metadata
    parts = [m.get("source_guideline", "Unknown guideline")]
    if m.get("chapter"):
        parts.append(m["chapter"])
    if m.get("label_type") and m.get("recommendation_number"):
        parts.append(f"{m['label_type']} {m['recommendation_number']}")
    return ", ".join(parts)


def prepare(state: GraphState) -> dict:
    """Initialize bookkeeping fields so callers only need to supply ``question``."""
    return {"original_question": state["question"], "rewrite_count": 0}


def retrieve(state: GraphState, config: RunnableConfig) -> dict:
    """Vector search over the KDIGO guideline collection for the current question."""
    cfg = Configuration.from_runnable_config(config)
    vectorstore = get_vectorstore(reset=False)
    filter_ = None
    if cfg.guideline_filter != "all":
        filter_ = {"source_guideline": GUIDELINE_FILTER_MAP[cfg.guideline_filter]}
    documents = vectorstore.similarity_search(state["question"], k=TOP_K, filter=filter_)
    return {"documents": documents}


def grade(state: GraphState) -> dict:
    """LLM-grade each retrieved chunk for relevance to the question, independently."""
    grader = get_llm().with_structured_output(GradeResult)
    inputs = [
        [
            ("system", GRADE_SYSTEM_PROMPT),
            ("user", GRADE_USER_PROMPT.format(question=state["question"], passage=doc.page_content)),
        ]
        for doc in state["documents"]
    ]
    results: list[GradeResult] = grader.batch(inputs) if inputs else []
    relevant = [doc for doc, result in zip(state["documents"], results, strict=True) if result.relevant]
    return {"graded_documents": relevant}


def generate(state: GraphState) -> dict:
    """Synthesize a cited answer from the graded, relevant chunks."""
    context = "\n\n".join(f"[{_cite(doc)}]\n{doc.page_content}" for doc in state["graded_documents"])
    messages = [
        ("system", GENERATE_SYSTEM_PROMPT),
        (
            "user",
            GENERATE_USER_PROMPT.format(
                user_context=state.get("user_context") or "(none learned yet)",
                question=state["question"],
                context=context,
            ),
        ),
    ]
    response = get_llm().invoke(messages)
    return {"answer": response.text}


def rewrite(state: GraphState) -> dict:
    """Rewrite the question to improve retrieval, incrementing the rewrite count."""
    messages = [
        ("system", REWRITE_SYSTEM_PROMPT),
        ("user", REWRITE_USER_PROMPT.format(question=state["question"])),
    ]
    response = get_llm().invoke(messages)
    return {"question": response.text, "rewrite_count": state["rewrite_count"] + 1}


def route_after_grade(state: GraphState) -> str:
    """Conditional edge: generate if any chunk survived grading, else rewrite (once)."""
    if state["graded_documents"]:
        return "generate"
    if state["rewrite_count"] < MAX_REWRITES:
        return "rewrite"
    # No relevant chunks even after a rewrite -- generate anyway so the model
    # can say explicitly that the guidelines don't cover the question.
    return "generate"
