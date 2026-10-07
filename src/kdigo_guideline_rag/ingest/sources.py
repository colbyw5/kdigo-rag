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
from typing import Literal

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
NUMBERED_HEADING_PATTERN = re.compile(r"^(?P<number>\d+(?:\.\d+)*)\.?\s+(?P<title>[A-Z][^\n]*)$")

# Matches a "References"/"Bibliography"-type section heading, optionally
# prefixed by chapter/section numbering (e.g. "12. References"). Generic
# vocabulary shared by essentially all clinical guideline and academic
# documents -- not tied to KDIGO's formatting -- so this is a default for
# every source, not a per-document opt-in.
REFERENCE_HEADING_PATTERN = re.compile(
    r"^(?:\d+(?:\.\d+)*\.?\s+)?"
    r"(references?|bibliography|works\s+cited|literature\s+cited|reference\s+list)\s*:?\s*$",
    re.IGNORECASE,
)

# Symbol-font glyphs that Docling decodes to the wrong code points in PDFs
# typeset by Kidney International (KDIGO's journal) -- confirmed by context
# across the CKD, Glomerular and BP guidelines ("ACR ‡ 30 mg/g" for >= 30,
# "target ( £ 120 mm Hg)" for <= 120, "n ¼ 2646" for n = 2646); absent from
# the differently typeset AKI draft. A property of one publisher's font
# encoding, not of PDFs in general ("¼" really is one-quarter elsewhere), so
# it's opt-in per source rather than a global default.
KI_SYMBOL_FONT_MAP = {
    "\x0f": "•",
    "‡": "≥",  # occasionally a genuine footnote dagger instead; rare, and harmless vs. garbling every >=
    "£": "≤",
    "\x00": "−",
    "\x06": "±",
    "\x0e": "°",
    "¼": "=",
}


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
        reference_heading_pattern: Regex identifying a references/bibliography
            section heading; content between it and the next real heading is
            dropped rather than chunked. Defaults to
            :data:`REFERENCE_HEADING_PATTERN`; set ``None`` to disable.
        noise_patterns: Strings/regexes to strip from extracted text before
            chunking (e.g. watermarks like "DRAFT" bleeding into the text).
        char_replacements: Single-character fixes applied to all extracted
            text (headings, narrative, tables) before anything else -- for
            mis-decoded font glyphs, e.g. :data:`KI_SYMBOL_FONT_MAP`.
        repair_split_ligatures: Rejoin ligature fragments ("bene fi ts" ->
            "benefits") using word frequencies; see ``text_repair.py``.
        repair_missing_spaces: Re-insert spaces in long no-space runs
            ("WesuggestthatadultswithhighBP...").
        narrative_chunk_size: Hard upper bound (words) for one narrative
            item; longer items are split into overlapping windows.
        narrative_chunk_overlap: Overlap (words) between those split windows.
        narrative_packing: How narrative segments are combined into chunks.
            ``"absorb"`` (default): segments of at least
            ``narrative_standalone_words`` stay their own chunk; only shorter
            fragments are attached to a neighbor -- keeps self-contained
            bullets/paragraphs precisely embedded. ``"greedy"``: pack every
            consecutive segment into windows of up to
            ``narrative_target_words`` -- more context per chunk, but dilutes
            short standalone passages (measured: see DECISIONS.md).
        narrative_standalone_words: ``"absorb"`` mode -- minimum words for a
            segment to stand alone as its own chunk.
        narrative_target_words: ``"greedy"`` mode -- window size in words.
        narrative_min_words: ``"greedy"`` mode -- a window shorter than this
            at a section boundary is merged into the preceding window of the
            same section (if the result stays within ``narrative_chunk_size``).
        near_duplicate_threshold: Word-sequence similarity at or above which
            a narrative item counts as a near-duplicate of a later one.
        near_duplicate_min_words: Items shorter than this are only deduped on
            exact (normalized) match -- similarity scores are unstable on
            very short text.
        figure_min_words_captioned: A picture with a figure caption is
            indexed if its own text (excluding the caption) has at least this
            many words -- low, because a captioned figure is deliberate
            content (e.g. a 12-word potassium-threshold chart).
        figure_min_words_uncaptioned: An uncaptioned picture needs at least
            this many words -- filters logos and mastheads ("INTERNATIONAL
            SOCIETY OF NEPHROLOGY" is <= 13 words), keeps uncaptioned
            algorithms (Docling often detaches captions from pictures).
    """

    name: str
    pdf_path: Path
    recommendation_pattern: re.Pattern | None = KDIGO_RECOMMENDATION_PATTERN
    heading_fallback_pattern: re.Pattern | None = NUMBERED_HEADING_PATTERN
    reference_heading_pattern: re.Pattern | None = REFERENCE_HEADING_PATTERN
    noise_patterns: list[re.Pattern] = field(default_factory=list)
    char_replacements: dict[str, str] = field(default_factory=dict)
    repair_split_ligatures: bool = True
    repair_missing_spaces: bool = True
    narrative_chunk_size: int = 500
    narrative_chunk_overlap: int = 50
    narrative_packing: Literal["absorb", "greedy"] = "absorb"
    narrative_standalone_words: int = 30
    narrative_target_words: int = 250
    narrative_min_words: int = 50
    near_duplicate_threshold: float = 0.95
    near_duplicate_min_words: int = 15
    figure_min_words_captioned: int = 5
    figure_min_words_uncaptioned: int = 20


KDIGO_SOURCES: list[GuidelineSource] = [
    GuidelineSource(
        name="KDIGO 2024 CKD Evaluation and Management",
        pdf_path=DATA_DIR / "KDIGO-2024-CKD-Guideline.pdf",
        char_replacements=KI_SYMBOL_FONT_MAP,
    ),
    GuidelineSource(
        name="KDIGO 2026 AKI/AKD (Public Review Draft)",
        pdf_path=DATA_DIR / "KDIGO-2026-AKI-AKD-Guideline-Public-Review-Draft-March-2026.pdf",
        noise_patterns=[re.compile(r"\bDRAFT\b")],
    ),
    GuidelineSource(
        name="KDIGO 2021 Glomerular Diseases",
        pdf_path=DATA_DIR / "KDIGO-2021-Glomerular-Diseases-Guideline_English_LN-2024-Update.pdf",
        char_replacements=KI_SYMBOL_FONT_MAP,
    ),
    GuidelineSource(
        name="KDIGO 2021 Blood Pressure in CKD",
        pdf_path=DATA_DIR / "KDIGO-2021-Blood-Pressure-in-CKD-Guideline.pdf",
        char_replacements=KI_SYMBOL_FONT_MAP,
    ),
]
