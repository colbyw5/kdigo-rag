"""Graph node functions: retrieve, grade, generate, rewrite."""

import logging
import re
from functools import lru_cache

from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from kdigo_guideline_rag.configuration import GUIDELINE_FILTER_MAP, REFERENCE_PASSAGES, Configuration
from kdigo_guideline_rag.ingest.embed import get_vectorstore
from kdigo_guideline_rag.ingest.text_layer import normalize
from kdigo_guideline_rag.prompts import (
    CONDENSE_SYSTEM_PROMPT,
    CONDENSE_USER_PROMPT,
    GENERATE_SYSTEM_PROMPT,
    GENERATE_USER_PROMPT,
    GRADE_SYSTEM_PROMPT,
    GRADE_USER_PROMPT,
    NOT_COVERED_RESPONSE,
    REWRITE_SYSTEM_PROMPT,
    REWRITE_USER_PROMPT,
)
from kdigo_guideline_rag.state import GraphState

logger = logging.getLogger(__name__)

MODEL = "claude-sonnet-5"
TOP_K = 5
MAX_REWRITES = 1
# Conversation turns the condense step sees, and how much of each earlier answer
# (the first lines are enough to know what a follow-up refers to).
CONDENSE_HISTORY_TURNS = 3
CONDENSE_ANSWER_CHARS = 400


class GradeResult(BaseModel):
    """Structured relevance grade for one retrieved passage."""

    relevant: bool = Field(description="Whether the chunk is relevant to the question")
    reasoning: str = Field(description="Brief explanation of the relevance decision")


@lru_cache(maxsize=1)
def get_llm() -> ChatAnthropic:
    """Shared, cached chat model -- reused by nodes.py and memory.py."""
    # Each question makes ~7 calls (memory, 5 grades, generate); with the SDK's
    # default 2 retries, one dropped connection killed whole eval runs.
    # timeout bounds a stalled connection (one hung 9.5 h); retries then recover it.
    return ChatAnthropic(model=MODEL, max_retries=6, timeout=120)


FIGURE_LABEL_PATTERN = re.compile(r"^\s*((?:Figure|Fig\.?)\s*\d+[A-Za-z]?)", re.IGNORECASE)


def _cite(doc: Document) -> str:
    m = doc.metadata
    parts = [m.get("source_guideline", "Unknown guideline")]
    if m.get("chapter"):
        parts.append(m["chapter"])
    if m.get("label_type") and m.get("recommendation_number"):
        parts.append(f"{m['label_type']} {m['recommendation_number']}")
    elif m.get("label_type") == "Figure" and (figure := FIGURE_LABEL_PATTERN.match(doc.page_content)):
        parts.append(figure.group(1))
    return ", ".join(parts)


@lru_cache(maxsize=1)
def reference_documents() -> tuple[Document, ...]:
    """Load the always-included reference passages (each organization's grading table) from the index.

    Raises if any is missing -- a silently absent reference would quietly
    turn grounded grade explanations back into unsupported ones.
    """
    vectorstore = get_vectorstore(reset=False)
    docs = []
    for guideline, caption in REFERENCE_PASSAGES:
        found = vectorstore.get(
            where={"$and": [{"source_guideline": guideline}, {"label_type": "Table"}]},
            include=["documents", "metadatas"],
        )
        match = next(
            (
                # Flagged so clients can show them apart from retrieved passages.
                Document(page_content=t, metadata={**m, "reference": True})
                for t, m in zip(found["documents"], found["metadatas"], strict=True)
                # Normalized match: captions vary ("Table 47| ..."; ADA's uncaptioned
                # table) and Docling pads markdown cells with spaces.
                if normalize(caption) in normalize(t[:400])
            ),
            None,
        )
        if match is None:
            raise LookupError(f"Reference passage not in index: {guideline!r} / {caption!r}")
        docs.append(match)
    return tuple(docs)


def format_context(docs: list[Document]) -> str:
    """Render graded passages exactly as the generate prompt receives them.

    Shared with the eval judge, so answers are judged against the same text
    (and citation headers) the generator actually saw.
    """
    return "\n\n".join(f"[{_cite(doc)}]\n{doc.page_content}" for doc in docs)


def prepare(state: GraphState) -> dict:
    """Start a turn: record the user's message and reset per-turn bookkeeping.

    Callers only supply ``question``. Per-turn fields are reset here because a
    thread's checkpointed state carries over from the previous turn.
    """
    return {
        "original_question": state["question"],
        "standalone_question": state["question"],
        "rewrite_count": 0,
        "messages": [HumanMessage(content=state["question"])],
    }


def _format_history(messages: list[AnyMessage]) -> str:
    lines = []
    for m in messages[-2 * CONDENSE_HISTORY_TURNS :]:
        if isinstance(m, HumanMessage):
            lines.append(f"User: {m.content}")
        elif isinstance(m, AIMessage):
            text = m.content if isinstance(m.content, str) else str(m.content)
            lines.append(
                f"Assistant: {text[:CONDENSE_ANSWER_CHARS]}{'...' if len(text) > CONDENSE_ANSWER_CHARS else ''}"
            )
    return "\n".join(lines)


def condense(state: GraphState) -> dict:
    """Make a follow-up self-contained using the conversation so far.

    First turns pass through with no LLM call. Follow-ups ("Should they get an
    SGLT2i?") are rewritten into a standalone question carrying only the
    context they depend on; that question drives retrieval, grading, and the
    answer. The answer itself still sees only this turn's passages (not
    earlier answers), so every answer stays grounded in its own citations.
    """
    history = state["messages"][:-1]  # everything before this turn's question
    if not history:
        return {"standalone_question": state["original_question"], "question": state["original_question"]}
    messages = [
        ("system", CONDENSE_SYSTEM_PROMPT),
        (
            "user",
            CONDENSE_USER_PROMPT.format(
                history=_format_history(history), question=state["original_question"]
            ),
        ),
    ]
    standalone = get_llm().invoke(messages).text.strip() or state["original_question"]
    return {"standalone_question": standalone, "question": standalone}


def retrieve(state: GraphState, config: RunnableConfig) -> dict:
    """Vector search over the guideline collection for the current question."""
    cfg = Configuration.from_runnable_config(config)
    vectorstore = get_vectorstore(reset=False)
    filter_ = None
    if cfg.guideline_filter != "all":
        filter_ = {"source_guideline": GUIDELINE_FILTER_MAP[cfg.guideline_filter]}
    documents = vectorstore.similarity_search(state["question"], k=TOP_K, filter=filter_)
    return {"documents": documents}


def grade(state: GraphState) -> dict:
    """LLM-grade each retrieved chunk for relevance to the question, independently.

    Each chunk is graded in isolation (``return_exceptions=True``): Claude
    occasionally returns a malformed tool call for one chunk in a batch (the
    whole JSON payload nested as a string inside a single field instead of
    proper top-level args), which would otherwise raise a pydantic
    ``ValidationError`` and crash the entire graph run over one bad chunk.
    A chunk that fails to grade is treated as not relevant rather than
    retried -- conservative, and consistent with how a chunk that's
    genuinely irrelevant is already handled.
    """
    grader = get_llm().with_structured_output(GradeResult)
    inputs = [
        [
            ("system", GRADE_SYSTEM_PROMPT),
            ("user", GRADE_USER_PROMPT.format(question=state["question"], passage=doc.page_content)),
        ]
        for doc in state["documents"]
    ]
    if not inputs:
        return {"graded_documents": []}

    results = grader.batch(inputs, return_exceptions=True)
    relevant = []
    for doc, result in zip(state["documents"], results, strict=True):
        if isinstance(result, Exception):
            logger.warning("Grading failed for a retrieved chunk, treating as not relevant: %r", result)
            continue
        if result.relevant:
            relevant.append(doc)
    return {"graded_documents": relevant}


def _searched_guidelines(cfg: Configuration) -> str:
    if cfg.guideline_filter != "all":
        return GUIDELINE_FILTER_MAP[cfg.guideline_filter]
    names = list(GUIDELINE_FILTER_MAP.values())
    return ", ".join(names[:-1]) + ", and " + names[-1]


def generate(state: GraphState, config: RunnableConfig) -> dict:
    """Synthesize a cited answer from the graded, relevant chunks.

    With no relevant chunks there is nothing to cite, so no LLM is called: a
    fixed response names the guidelines searched. Prompting the model to
    decline proved unreliable -- given an empty context it stated ESA
    hemoglobin targets and tolvaptan dosing from its own knowledge
    (DECISIONS.md 2026-10-09) -- so this is enforced structurally, not by
    prompt. Answers the user's original question; a rewritten one exists only
    to improve retrieval.
    """
    if not state["graded_documents"]:
        cfg = Configuration.from_runnable_config(config)
        answer = NOT_COVERED_RESPONSE.format(searched=_searched_guidelines(cfg))
        return {"answer": answer, "generation_documents": [], "messages": [AIMessage(content=answer)]}
    graded = state["graded_documents"]
    docs = graded + [
        ref for ref in reference_documents() if ref.page_content not in {d.page_content for d in graded}
    ]
    context = format_context(docs)
    messages = [
        ("system", GENERATE_SYSTEM_PROMPT),
        (
            "user",
            GENERATE_USER_PROMPT.format(
                user_context=state.get("user_context") or "(none learned yet)",
                question=state.get("standalone_question")
                or state.get("original_question")
                or state["question"],
                context=context,
            ),
        ),
    ]
    response = get_llm().invoke(messages)
    return {
        "answer": response.text,
        "generation_documents": docs,
        "messages": [AIMessage(content=response.text)],
    }


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
