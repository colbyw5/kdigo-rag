"""Recommendation extraction from the PDF's text layer (PyMuPDF).

Docling's layout model gives the best structure for most guidelines, but its
reading order can break recommendation boxes in multi-column layouts (ADA
Standards of Care: recommendations cut mid-word, continuations scattered,
some never starting an item). The text layer keeps each box contiguous.

Everything here is verbatim PDF text: recommendations are *located* with the
source's own ``recommendation_pattern`` and *bounded* by the next
recommendation or the source's grade marker -- never rewritten.
"""

import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from kdigo_guideline_rag.ingest.sources import GuidelineSource, noise_pattern

MAX_RECOMMENDATION_WORDS = 160
SENTENCE_END = re.compile(r"[.!?)\]](?=\s|$)")
# A recommendation can't end on a list lead-in -- "can be established by:"
# or an enumerator "(i)" / "(a)" / "(2)" means its list continues.
CONTINUES = re.compile(r"(:|\((?:[ivx]+|[a-z]|\d{1,2})\))\s*$")


@dataclass(frozen=True)
class TextLayerRecommendation:
    """One recommendation as it appears in the PDF text layer."""

    label: str | None
    number: str
    text: str
    page: int


@lru_cache(maxsize=16)
def _pages(pdf_path: Path) -> tuple[str, ...]:
    import fitz  # PyMuPDF

    with fitz.open(pdf_path) as doc:
        return tuple(page.get_text("text") for page in doc)


def page_texts(source: GuidelineSource) -> list[str]:
    """Per-page text-layer text, with soft hyphens and the source's noise removed."""
    out = []
    for text in _pages(Path(source.pdf_path)):
        text = re.sub(r"\xad\s*", "", text)  # soft hyphen (+ its line break) never belongs in text
        text = re.sub(r"[\u200b\u200c\u200d\u2060\ufeff]", "", text)  # zero-width characters
        for pattern in source.noise_patterns:
            text = pattern.sub(" ", text)
        out.append(text)
    return out


def normalize(text: str) -> str:
    """Comparison form: NFKC (PyMuPDF returns ligatures as single code points),
    lowercase, letters and digits only -- immune to spacing, glyph and ligature noise."""
    return re.sub(r"[^a-z0-9]", "", unicodedata.normalize("NFKC", text).lower())


@lru_cache(maxsize=16)
def _pdf_normalized(pdf_path: Path, noise: tuple[str, ...]) -> str:
    source = GuidelineSource(name="", pdf_path=pdf_path, noise_patterns=[noise_pattern(p) for p in noise])
    return normalize(" ".join(page_texts(source)))


def is_verbatim(text: str, source: GuidelineSource) -> bool:
    """Whether ``text`` appears contiguously in the PDF's text layer (normalized)."""
    return normalize(text) in _pdf_normalized(
        Path(source.pdf_path), tuple(p.pattern for p in source.noise_patterns)
    )


def is_valid_start(text: str, m: re.Match, source: GuidelineSource) -> bool:
    """Whether a recommendation-pattern match in ``text`` is a real start: at
    the start of the text or a line, after a sentence end, or right after one of
    the source's own grade markers ("...insulin. A 9.2 For most adults" in ADA).
    Rejects mid-sentence mentions ("as in Recommendation 3.7.1:", "HbA1c 7.0 In
    trials..."). Uses only the source's patterns -- no format is assumed here.

    Only *bare-number* patterns (ADA's "11.7a") need this context check. A
    *labelled* pattern ("Practice Point 1.2.1:") delimits itself and is
    accepted anywhere -- requiring a sentence end before it rejected 23 real
    KDIGO AKI practice points whose section heading Docling had merged into
    the same item ("1.2. Staging of AKI Practice Point 1.2.1:").
    """
    if m.groupdict().get("label"):
        return True
    stripped = text[max(0, m.start() - 60) : m.start()].rstrip(" \t")
    if not stripped or stripped.endswith("\n") or SENTENCE_END.search(stripped[-1:] + " "):
        return True
    return bool(source.grade_pattern) and any(
        g.end() == len(stripped) for g in source.grade_pattern.finditer(stripped)
    )


def is_complete(text: str, source: GuidelineSource) -> bool:
    """Whether a recommendation's text ends properly: with its grade marker or a sentence end.

    Catches recommendations cut off by a column or page break ("...SGLT2
    inhibitors shoul").
    """
    if CONTINUES.search(text):
        return False
    body = text.rstrip(" \t;.")
    if source.grade_pattern and any(g.end() >= len(body) for g in source.grade_pattern.finditer(body)):
        return True
    return bool(SENTENCE_END.search(text.rstrip()[-1:] + " "))


# A caption starts a line and has a separator right after its number ("Figure
# 9.3—Simplified...", "Table 2| GFR..."); an in-text reference wrapped to a
# line start ("Table 9.2 ). A") has none.
CAPTION_LINE = re.compile(r"\n[ \t]*(?:Figure|Fig\.|Table)[ \t]*\d+(?:\.\d+)*[A-Za-z]?[ \t]*[—–|:-]")


def _bound(raw: str, source: GuidelineSource) -> str:
    """Trim a recommendation (raw text, line breaks intact) to where it ends.

    The text layer runs straight on into the rationale. A recommendation box or
    paragraph ends at a line break, so cut at the first *line-ending* grade
    marker (ADA "...benefits. C\\n", KDIGO "...(1B).\\n"); without a grade
    (practice points), at the first sentence end that ends a line. The last
    marker would overshoot: rationale cites other recommendations' grades.
    A figure/table caption line always ends it -- captions sit between boxes
    in the layout, and the text layer reads straight into them. Falls back to
    a word cap, then whitespace is collapsed.
    """
    window = " ".join(raw.split(" ")[: MAX_RECOMMENDATION_WORDS * 2])  # generous; cap applied below
    if caption := CAPTION_LINE.search(window):
        window = window[: caption.start()]
    cut = None
    if source.grade_pattern:
        for g in source.grade_pattern.finditer(window):
            tail = re.match(r"[ \t]*[.;]?[ \t]*(\n|$)", window[g.end() :])
            if tail:
                cut = g.end() + len(tail.group(0).rstrip("\n"))
                break
    if cut is None:
        # Prefer a blank line (true paragraph end); a period at a line end may
        # be a mere wrap. Skip ends that are list lead-ins ("...by: (i)").
        cut = len(window)
        for pattern in (r"[.!?)\]][ \t]*\n[ \t]*\n", r"[.!?)\]][ \t]*\n"):
            ends = [e for e in re.finditer(pattern, window) if not CONTINUES.search(window[: e.start() + 1])]
            if ends:
                cut = ends[0].start() + 1
                break
    words = re.sub(r"\s+", " ", window[:cut]).strip().split(" ")
    return " ".join(words[:MAX_RECOMMENDATION_WORDS])


def text_layer_recommendations(
    source: GuidelineSource,
) -> dict[tuple[str | None, str], TextLayerRecommendation]:
    """All recommendations in the text layer, keyed by ``(label, number)`` (first occurrence wins).

    Returns an empty dict for sources without a recommendation pattern.
    """
    if source.recommendation_pattern is None:
        return {}
    pages = page_texts(source)
    full = "\n".join(pages)
    page_starts, offset = [], 0
    for text in pages:
        page_starts.append(offset)
        offset += len(text) + 1

    starts = [m for m in source.recommendation_pattern.finditer(full) if is_valid_start(full, m, source)]
    found: dict[tuple[str | None, str], TextLayerRecommendation] = {}
    for i, m in enumerate(starts):
        end = starts[i + 1].start() if i + 1 < len(starts) else len(full)
        text = _bound(full[m.start() : end], source)
        label = m.groupdict().get("label") or source.recommendation_label
        key = (label, m.group("number"))
        if key not in found:
            page = sum(1 for s in page_starts if s <= m.start())  # 1-indexed
            found[key] = TextLayerRecommendation(label=label, number=m.group("number"), text=text, page=page)
    return found
