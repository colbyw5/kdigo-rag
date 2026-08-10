"""Generic, source-config-driven chunking over a Docling document.

Walks a :class:`~docling.datamodel.document.DoclingDocument` in reading
order, tracking the current chapter/section from ``section_header`` items,
and splits ``text``/``list_item`` items at recommendation boundaries defined
by the active :class:`~kdigo_guideline_rag.ingest.sources.GuidelineSource`.
Text with no recommendation anchor becomes a narrative chunk instead. Each
``table`` item becomes its own chunk (markdown table + resolved caption, via
Docling's own rendering). Pictures are skipped -- no text content to index.
"""

import re
from dataclasses import dataclass

from docling_core.types.doc.document import DoclingDocument

from kdigo_guideline_rag.ingest.sources import GuidelineSource

GRADE_PATTERN = re.compile(r"\((\d[A-D]|Not Graded)\)", re.IGNORECASE)
CHAPTER_PATTERN = re.compile(r"^CHAPTER\s+", re.IGNORECASE)
CAPTION_ONLY_PATTERN = re.compile(r"^(Figure|Table)\s+\d+\s*\|", re.IGNORECASE)


@dataclass
class Chunk:
    """One retrievable unit of guideline text with citation metadata.

    Attributes:
        text: The chunk's content -- a recommendation/practice point plus
            trailing rationale, or a narrative window.
        source_guideline: Name of the source guideline (citation field).
        chapter: Nearest preceding chapter-level heading, if any.
        section: Nearest preceding section-level heading, if any.
        label_type: ``"Recommendation"``, ``"Practice Point"``, ``"Table"``,
            or ``None`` for narrative chunks with no recommendation anchor.
        recommendation_number: The numbered label (e.g. ``"1.1.2.1"``).
        grade: Evidence grade (e.g. ``"1B"``), if present.
        page: Source PDF page number.
    """

    text: str
    source_guideline: str
    chapter: str | None = None
    section: str | None = None
    label_type: str | None = None
    recommendation_number: str | None = None
    grade: str | None = None
    page: int | None = None


def _strip_noise(text: str, source: GuidelineSource) -> str:
    for pattern in source.noise_patterns:
        text = pattern.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def _split_narrative(text: str, source: GuidelineSource) -> list[str]:
    words = text.split()
    if len(words) <= source.narrative_chunk_size:
        return [text]
    step = source.narrative_chunk_size - source.narrative_chunk_overlap
    return [" ".join(words[i : i + source.narrative_chunk_size]) for i in range(0, len(words), step)]


def _update_heading(text: str, chapter: str | None, section: str | None) -> tuple[str | None, str | None]:
    if CHAPTER_PATTERN.match(text):
        return text, None
    return chapter, text


def _dedupe_recommendations(chunks: list[Chunk]) -> list[Chunk]:
    """Drop repeat occurrences of the same recommendation/practice point.

    KDIGO guidelines restate every recommendation twice: once in a terse
    "Summary of recommendation statements" listing, once in its real chapter
    with surrounding rationale. Both occurrences are byte-identical (the
    rationale lives in separate, unanchored narrative chunks, not attached
    to the recommendation itself) -- confirmed by direct comparison across
    several recommendations, so the first occurrence is kept with no loss.
    """
    seen: set[tuple[str, str]] = set()
    deduped: list[Chunk] = []
    for chunk in chunks:
        if chunk.label_type and chunk.recommendation_number:
            key = (chunk.label_type, chunk.recommendation_number)
            if key in seen:
                continue
            seen.add(key)
        deduped.append(chunk)
    return deduped


def chunk_document(doc: DoclingDocument, source: GuidelineSource) -> list[Chunk]:
    """Chunk a parsed Docling document per ``source``'s configuration.

    Args:
        doc: The Docling document produced from ``source.pdf_path``.
        source: Guideline-specific chunking configuration.

    Returns:
        Recommendation-anchored and narrative chunks, in reading order.
    """
    chunks: list[Chunk] = []
    chapter: str | None = None
    section: str | None = None

    for item, _level in doc.iterate_items():
        label = getattr(item, "label", None)

        if label == "table":
            table_md = item.export_to_markdown(doc).strip()
            if table_md:
                prov = getattr(item, "prov", None)
                chunks.append(
                    Chunk(
                        text=table_md,
                        source_guideline=source.name,
                        chapter=chapter,
                        section=section,
                        label_type="Table",
                        page=prov[0].page_no if prov else None,
                    )
                )
            continue

        text = getattr(item, "text", None)
        if not text:
            continue

        if label == "section_header":
            stripped = text.strip()
            # Docling occasionally mislabels non-heading content as a
            # section_header: a recommendation/practice-point statement
            # (bold text in CKD guideline), or a table/figure caption line.
            # A recommendation must fall through to content chunking or it's
            # silently dropped; a caption is redundant (the table/picture
            # item already carries it) so it's dropped outright.
            if CAPTION_ONLY_PATTERN.match(stripped):
                continue
            is_mislabeled_recommendation = (
                source.recommendation_pattern and source.recommendation_pattern.search(stripped)
            )
            if not is_mislabeled_recommendation:
                chapter, section = _update_heading(stripped, chapter, section)
                continue

        if label not in ("text", "list_item", "section_header"):
            continue

        text = _strip_noise(text, source)
        if not text:
            continue

        # Docling sometimes tags a numbered subsection heading as a plain
        # text/list_item instead of section_header (varies by document).
        if source.heading_fallback_pattern and source.heading_fallback_pattern.fullmatch(text):
            chapter, section = _update_heading(text, chapter, section)
            continue

        # Bare figure/table caption lines carry no standalone information.
        if CAPTION_ONLY_PATTERN.match(text):
            continue

        prov = getattr(item, "prov", None)
        page = prov[0].page_no if prov else None

        pattern = source.recommendation_pattern
        matches = list(pattern.finditer(text)) if pattern else []

        if not matches:
            for piece in _split_narrative(text, source):
                chunks.append(
                    Chunk(
                        text=piece,
                        source_guideline=source.name,
                        chapter=chapter,
                        section=section,
                        page=page,
                    )
                )
            continue

        if matches[0].start() > 0:
            leading = text[: matches[0].start()].strip()
            if leading:
                chunks.append(
                    Chunk(
                        text=leading,
                        source_guideline=source.name,
                        chapter=chapter,
                        section=section,
                        page=page,
                    )
                )

        for i, match in enumerate(matches):
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            chunk_text = text[match.start() : end].strip()
            grade_match = GRADE_PATTERN.search(chunk_text)
            chunks.append(
                Chunk(
                    text=chunk_text,
                    source_guideline=source.name,
                    chapter=chapter,
                    section=section,
                    label_type=match.group("label"),
                    recommendation_number=match.group("number"),
                    grade=grade_match.group(1) if grade_match else None,
                    page=page,
                )
            )

    return _dedupe_recommendations(chunks)
