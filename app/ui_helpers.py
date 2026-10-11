"""Pure display logic for the Streamlit client -- no Streamlit import, so it's
unit-testable in the main environment.

The client only sees the graph's final state as JSON from the LangGraph API
server; documents arrive either as plain ``{"page_content", "metadata"}``
dicts or as LangChain's serialized constructor form, depending on the
server's serializer. Both are handled here.
"""

import re
from dataclasses import dataclass

# Mirrors the graph's Configuration.guideline_filter values; the UI env
# deliberately doesn't install the RAG package to import them.
GUIDELINE_FILTERS = {
    "All guidelines": "all",
    "CKD 2024": "ckd_only",
    "AKI/AKD 2026 (public review draft)": "aki_only",
    "Glomerular Diseases 2021": "glomerular_only",
    "Blood Pressure in CKD 2021": "bp_only",
}

FIGURE_OR_TABLE = re.compile(r"^\s*((?:Figure|Fig\.?|Table)\s*\d+[A-Za-z]?)", re.IGNORECASE)


@dataclass(frozen=True)
class Source:
    """One passage the answer was built from, ready to display."""

    label: str
    text: str
    page: int | None
    reference: bool


def _fields(doc: dict) -> tuple[str, dict]:
    """``(page_content, metadata)`` from either serialized Document shape."""
    if "kwargs" in doc:  # LangChain serialized constructor: {"lc": 1, "type": "constructor", ...}
        doc = doc["kwargs"]
    return doc.get("page_content", ""), doc.get("metadata", {}) or {}


def source_label(text: str, metadata: dict) -> str:
    """Citation-style label: guideline, chapter, then rec/practice point or figure/table."""
    parts = [metadata.get("source_guideline", "Unknown guideline")]
    if metadata.get("chapter"):
        parts.append(metadata["chapter"])
    if metadata.get("label_type") and metadata.get("recommendation_number"):
        parts.append(f"{metadata['label_type']} {metadata['recommendation_number']}")
    elif metadata.get("label_type") in ("Figure", "Table") and (m := FIGURE_OR_TABLE.match(text)):
        parts.append(m.group(1))
    return ", ".join(parts)


def sources_from_state(state: dict) -> list[Source]:
    """Passages the generator received, in order (empty for a 'not covered' answer)."""
    sources = []
    for doc in state.get("generation_documents") or []:
        text, meta = _fields(doc)
        sources.append(
            Source(
                label=source_label(text, meta),
                text=text,
                page=meta.get("page"),
                reference=bool(meta.get("reference")),
            )
        )
    return sources


def pipeline_summary(state: dict) -> dict[str, object]:
    """What the pipeline did for this question -- for the debug panel."""
    question = state.get("original_question") or state.get("question", "")
    rewritten = state.get("question") if state.get("question") != question else None
    return {
        "retrieved": len(state.get("documents") or []),
        "graded relevant": len(state.get("graded_documents") or []),
        "rewritten query": rewritten,
        "what's known about you": state.get("user_context") or "(nothing yet)",
    }
