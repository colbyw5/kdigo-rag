"""Configurable fields for LangGraph assistants."""

from dataclasses import dataclass, fields
from typing import Any, Literal

# Maps a guideline_filter value to the source_guideline metadata string
# used in the Chroma collection (see the guideline profiles in ingest/profiles/).
GUIDELINE_FILTER_MAP = {
    "ckd_only": "KDIGO 2024 CKD Evaluation and Management",
    "aki_only": "KDIGO 2026 AKI/AKD (Public Review Draft)",
    "glomerular_only": "KDIGO 2021 Glomerular Diseases",
    "bp_only": "KDIGO 2021 Blood Pressure in CKD",
}

# Passages always given to the generator alongside retrieved chunks, as
# (source_guideline, caption prefix) -- verbatim corpus text, so explaining an
# evidence grade ("1B") is grounded and citable instead of model knowledge.
# Chosen by reading the extracted text, not just matching captions: the CKD
# guideline's certainty table (Table 45) extracts with merged cells that read
# as "B = Low", so the BP guideline's clean copy is used instead.
REFERENCE_PASSAGES = (
    ("KDIGO 2024 CKD Evaluation and Management", "Table 47| KDIGO nomenclature and description for grading"),
    ("KDIGO 2021 Blood Pressure in CKD", "Table 3| Classification for certainty and quality of the evidence"),
)


@dataclass
class Configuration:
    """Runtime-configurable options for the CKD guideline RAG assistant.

    Attributes:
        guideline_filter: Restrict retrieval to a specific guideline, or "all".
        clinical_focus: Disease area the graph is pointed at.
    """

    guideline_filter: Literal["all", "ckd_only", "aki_only", "glomerular_only", "bp_only"] = "all"
    clinical_focus: str = "kidney"

    @classmethod
    def from_runnable_config(cls, config: dict[str, Any] | None) -> "Configuration":
        """Build a Configuration from a LangGraph ``RunnableConfig``.

        Ignores any ``configurable`` keys not defined on this dataclass
        (e.g. platform-injected ``thread_id``).
        """
        configurable = (config or {}).get("configurable", {})
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in configurable.items() if k in known})
