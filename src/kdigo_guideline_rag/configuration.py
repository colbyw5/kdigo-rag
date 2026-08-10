"""Configurable fields for LangGraph assistants."""

from dataclasses import dataclass
from typing import Literal


@dataclass
class Configuration:
    """Runtime-configurable options for the CKD guideline RAG assistant.

    Attributes:
        guideline_filter: Restrict retrieval to a specific guideline, or "all".
        clinical_focus: Disease area the graph is pointed at.
    """

    guideline_filter: Literal[
        "all", "ckd_only", "aki_only", "glomerular_only", "bp_only"
    ] = "all"
    clinical_focus: str = "kidney"
