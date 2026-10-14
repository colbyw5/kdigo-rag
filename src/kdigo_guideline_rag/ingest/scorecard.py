"""Parse-quality scorecard: format-independent signals that a guideline was chunked correctly.

The point is to fail loudly on a new document instead of silently indexing
broken text. Every check uses only the chunks, the source's own patterns,
and the PDF text layer -- nothing about any particular publisher:

- **completeness**: gaps in recommendation numbering (11.6b, 11.6d -> 11.6c
  missing), and recommendations the text layer has but the chunks don't;
- **precision**: recommendations cut off (no grade / sentence end), and
  recommendation text that isn't verbatim in the PDF;
- **cleanliness**: leftover soft hyphens / control / replacement characters,
  and tiny narrative fragments.

A numbering gap the text layer *also* lacks (e.g. a draft that skips a
number) is reported as ``unconfirmed_gaps`` -- informational, not a failure.
"""

import re
import unicodedata
from collections import defaultdict
from dataclasses import asdict, dataclass, field

from kdigo_guideline_rag.ingest.chunker import Chunk
from kdigo_guideline_rag.ingest.sources import GuidelineSource
from kdigo_guideline_rag.ingest.text_layer import is_complete, is_verbatim, text_layer_recommendations

ARTIFACT_CHARS = {"soft_hyphen": "­", "replacement_char": "�", "zero_width": "\u200b"}
TINY_WORDS = 5
# A symbol right before a number that isn't a normal math/punctuation mark --
# the signature of a mis-decoded font glyph ("ACR \u2021 30 mg/g" for >= 30). Copyright,
# superscripts, daggers, Greek letters and full-width digits are legitimate.
GLYPH_SUSPECT = re.compile(
    r"([^A-Za-z0-9_\s<>\u2264\u2265=\u00b1%.,;:()\[\]/+\-\u2013\u2014\u2212\u00b0\u00b7\u2022\u00d7\u00a9\u00b2\u00b3\u2020\u0391-\u03c9\uff10-\uff19'\"\u2018\u2019\u201c\u201d~\u223c\u2248$*#&|])\s?\d"
)


def _norm(text: str) -> str:
    # NFKC first: PyMuPDF returns typeset ligatures as single code points
    # ("\ufb01" for "fi"), which a bare alphanumeric filter would delete.
    return re.sub(r"[^a-z0-9]", "", unicodedata.normalize("NFKC", text).lower())


def _count(items: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        counts[item] = counts.get(item, 0) + 1
    return {k: v for k, v in counts.items() if v >= 3}


def numbering_gaps(numbers: set[str]) -> list[str]:
    """Numbers missing from otherwise-consecutive sequences ("3.7.1, 3.7.3" -> "3.7.2"; "11.6b, 11.6d" -> "11.6c").

    Recommendations and practice points share a numbering space in KDIGO, so
    gaps are computed over all labels together.
    """
    by_parent: dict[tuple[int, ...], set[int]] = defaultdict(set)
    letters: dict[str, set[str]] = defaultdict(set)
    for number in numbers:
        m = re.fullmatch(r"(\d+(?:\.\d+)*)([a-z]?)", number)
        if not m:
            continue
        base, letter = m.groups()
        parts = tuple(int(p) for p in base.split("."))
        by_parent[parts[:-1]].add(parts[-1])
        if letter:
            letters[base].add(letter)
    gaps = []
    for parent, seen in by_parent.items():
        prefix = ".".join(map(str, parent))
        gaps += [f"{prefix}.{k}" if prefix else str(k) for k in range(1, max(seen)) if k not in seen]
    for base, seen in letters.items():
        gaps += [f"{base}{chr(c)}" for c in range(ord("a"), ord(max(seen))) if chr(c) not in seen]
    return sorted(gaps)


@dataclass
class Scorecard:
    """Quality signals for one source's chunks. ``passed`` gates indexing."""

    source: str
    extractor: str
    recommendations: int
    text_layer_recommendations: int
    missing_vs_text_layer: list[str] = field(default_factory=list)
    numbering_gaps: list[str] = field(default_factory=list)
    unconfirmed_gaps: list[str] = field(default_factory=list)
    truncated: list[str] = field(default_factory=list)
    not_verbatim: list[str] = field(default_factory=list)
    artifacts: dict[str, int] = field(default_factory=dict)
    narrative_chunks: int = 0
    tiny_fragments: int = 0
    tables: int = 0
    figures: int = 0
    glyph_suspects: dict[str, int] = field(default_factory=dict)
    expects_recommendations: bool = True
    examples: dict[str, list[str]] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """No confirmed-missing recommendations, at most ~2% cut off or non-verbatim, no artifacts."""
        tolerance = max(1, round(0.02 * self.recommendations))
        return (
            (self.recommendations > 0 or not self.expects_recommendations)
            and not self.missing_vs_text_layer
            and len(self.truncated) <= tolerance
            and len(self.not_verbatim) <= tolerance
            and not any(self.artifacts.values())
        )

    @property
    def problems(self) -> float:
        """Ranking key across candidate strategies (lower is better)."""
        return (
            3 * len(self.missing_vs_text_layer)
            + 2 * len(self.truncated)
            + 2 * len(self.not_verbatim)
            + sum(self.artifacts.values())
            + self.tiny_fragments / 50
            + (100 if self.recommendations == 0 and self.expects_recommendations else 0)
        )

    def summary(self) -> dict:
        """JSON-friendly view, including ``passed`` and ``problems``."""
        return {**asdict(self), "passed": self.passed, "problems": round(self.problems, 2)}


def score(chunks: list[Chunk], source: GuidelineSource) -> Scorecard:
    """Score one source's chunks against its own patterns and the PDF text layer."""
    recs = [c for c in chunks if c.label_type and c.recommendation_number]
    narrative = [c for c in chunks if c.label_type is None]
    numbers = {c.recommendation_number for c in recs}

    tl = text_layer_recommendations(source)
    tl_numbers = {number for _, number in tl}

    gaps = numbering_gaps(numbers)
    truncated = [c for c in recs if not is_complete(c.text, source)]
    not_verbatim = [c for c in recs if not is_verbatim(c.text, source)]
    all_text = " ".join(c.text for c in chunks)
    return Scorecard(
        source=source.name,
        extractor=source.recommendation_extractor,
        recommendations=len(recs),
        text_layer_recommendations=len(tl),
        missing_vs_text_layer=sorted(tl_numbers - numbers),
        numbering_gaps=[g for g in gaps if g in tl_numbers],
        unconfirmed_gaps=[g for g in gaps if g not in tl_numbers],
        truncated=[f"{c.label_type} {c.recommendation_number}" for c in truncated],
        not_verbatim=[f"{c.label_type} {c.recommendation_number}" for c in not_verbatim],
        artifacts={name: all_text.count(ch) for name, ch in ARTIFACT_CHARS.items()},
        narrative_chunks=len(narrative),
        tiny_fragments=sum(len(c.text.split()) < TINY_WORDS for c in narrative),
        tables=sum(c.label_type == "Table" for c in chunks),
        expects_recommendations=source.recommendation_pattern is not None,
        glyph_suspects=dict(
            sorted(
                _count(GLYPH_SUSPECT.findall(all_text) + re.findall(r"(\$)[ \t]\d", all_text)).items(),
                key=lambda kv: -kv[1],
            )[:8]
        ),
        figures=sum(c.label_type == "Figure" for c in chunks),
        examples={
            "truncated": [c.text[-120:] for c in truncated[:3]],
            "not_verbatim": [c.text[:120] for c in not_verbatim[:3]],
        },
    )
