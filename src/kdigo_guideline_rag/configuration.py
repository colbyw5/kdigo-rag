"""Configurable fields for LangGraph assistants."""

from dataclasses import dataclass, fields
from typing import Any, Literal

# Maps a guideline_filter value to the source_guideline metadata string
# used in the Chroma collection (see ingest/sources.py -- KDIGO_SOURCES).
GUIDELINE_FILTER_MAP = {
    "ckd_only": "KDIGO 2024 CKD Evaluation and Management",
    "aki_only": "KDIGO 2026 AKI/AKD (Public Review Draft)",
    "glomerular_only": "KDIGO 2021 Glomerular Diseases",
    "bp_only": "KDIGO 2021 Blood Pressure in CKD",
}


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
