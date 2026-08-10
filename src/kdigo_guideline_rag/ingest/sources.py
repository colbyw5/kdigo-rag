"""Per-guideline configuration for the generic ingest pipeline.

Each :class:`GuidelineSource` describes how to extract recommendation-level
chunks from one guideline PDF. The KDIGO documents share a common
``Recommendation X.Y(.Z)?:`` / ``Practice Point X.Y.Z:`` convention, but the
fields here exist so a future, differently formatted guideline (e.g. one
with no numbered-recommendation convention at all, or one where Docling
fails to emit real markdown headings) can be onboarded by adding a new
``GuidelineSource`` entry rather than changing the chunking logic.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[3] / "data"

# Matches "Recommendation 1.1.2.1:" and "Practice Point 1.1.3.1:" across all
# four KDIGO guidelines, regardless of numbering depth (BP uses X.Y, CKD and
# Glomerular use X.Y.Z, AKI mixes both). Confirmed by direct inspection of
# all four PDFs -- see DECISIONS.md.
KDIGO_RECOMMENDATION_PATTERN = re.compile(
    r"(?P<label>Recommendation|Practice Point)\s+(?P<number>\d+(?:\.\d+)*)\s*:"
)

# Fallback for numbered subsections Docling tags as plain text/list items
# instead of section_header -- observed in *every* KDIGO doc so far (e.g.
# AKI's "1.4. AKI and AKD resolution", CKD's "1.1 Detection and evaluation
# of CKD"), so this is a default, not a per-document opt-in.
NUMBERED_HEADING_PATTERN = re.compile(
    r"^(?P<number>\d+(?:\.\d+)*)\.?\s+(?P<title>[A-Z][^\n]*)$"
)


@dataclass
class GuidelineSource:
    """Configuration for one guideline document in the ingest pipeline.

    Attributes:
        name: Value stored as the ``source_guideline`` chunk metadata field.
        pdf_path: Path to the source PDF.
        recommendation_pattern: Regex identifying recommendation/practice-point
            boundaries. ``None`` disables recommendation-level chunking and
            falls back to heading/fixed-window chunking only.
        heading_fallback_pattern: Regex for numbered subsections that Docling
            tags as plain text/list items instead of section headers. Defaults
            to :data:`NUMBERED_HEADING_PATTERN`; set ``None`` to disable.
        noise_patterns: Strings/regexes to strip from extracted text before
            chunking (e.g. watermarks like "DRAFT" bleeding into the text).
        narrative_chunk_size: Target token count for fixed-window chunks used
            on narrative text with no recommendation anchor.
        narrative_chunk_overlap: Overlap (tokens) between narrative chunks.
    """

    name: str
    pdf_path: Path
    recommendation_pattern: re.Pattern | None = KDIGO_RECOMMENDATION_PATTERN
    heading_fallback_pattern: re.Pattern | None = NUMBERED_HEADING_PATTERN
    noise_patterns: list[re.Pattern] = field(default_factory=list)
    narrative_chunk_size: int = 500
    narrative_chunk_overlap: int = 50


KDIGO_SOURCES: list[GuidelineSource] = [
    GuidelineSource(
        name="KDIGO 2024 CKD Evaluation and Management",
        pdf_path=DATA_DIR / "KDIGO-2024-CKD-Guideline.pdf",
    ),
    GuidelineSource(
        name="KDIGO 2026 AKI/AKD (Public Review Draft)",
        pdf_path=DATA_DIR
        / "KDIGO-2026-AKI-AKD-Guideline-Public-Review-Draft-March-2026.pdf",
        noise_patterns=[re.compile(r"\bDRAFT\b")],
    ),
    GuidelineSource(
        name="KDIGO 2021 Glomerular Diseases",
        pdf_path=DATA_DIR
        / "KDIGO-2021-Glomerular-Diseases-Guideline_English_LN-2024-Update.pdf",
    ),
    GuidelineSource(
        name="KDIGO 2021 Blood Pressure in CKD",
        pdf_path=DATA_DIR / "KDIGO-2021-Blood-Pressure-in-CKD-Guideline.pdf",
    ),
]
