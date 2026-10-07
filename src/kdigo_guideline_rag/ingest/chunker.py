"""Generic, source-config-driven chunking over a Docling document.

Walks a :class:`~docling.datamodel.document.DoclingDocument` in reading
order, tracking the current chapter/section from ``section_header`` items,
and splits ``text``/``list_item`` items at recommendation boundaries defined
by the active :class:`~kdigo_guideline_rag.ingest.sources.GuidelineSource`.
Text with no recommendation anchor becomes a narrative chunk instead. Each
``table`` item becomes its own chunk (markdown table + resolved caption, via
Docling's own rendering). Each ``picture`` item with enough text becomes a
``"Figure"`` chunk: Docling extracts text inside vector figures as the
picture's children (flowcharts, criteria lists -- e.g. KDIGO CKD Figure 48's
referral criteria), which would otherwise never reach the index.

Three stages: **extract** one segment per Docling item (or per
recommendation within an item), **dedupe** (recommendations: exact by
number; tables: exact text; narrative: exact + near-duplicate text), then
**pack** narrative segments within a section -- by default absorbing only
short fragments into a neighbor (see ``GuidelineSource.narrative_packing``).
Packing runs last because a duplicated paragraph packed alongside two
different sets of neighbors no longer looks like a duplicate.
"""

import logging
import re
from dataclasses import dataclass, replace
from difflib import SequenceMatcher

from docling_core.types.doc.document import DoclingDocument

from kdigo_guideline_rag.ingest.sources import GuidelineSource

GRADE_PATTERN = re.compile(r"\((\d[A-D]|Not Graded)\)", re.IGNORECASE)
CHAPTER_PATTERN = re.compile(r"^CHAPTER\s+", re.IGNORECASE)
CAPTION_ONLY_PATTERN = re.compile(r"^(Figure|Table)\s+\d+\s*\|", re.IGNORECASE)
# Generic figure-caption shape ("Figure 48|", "Fig. 3.", "Figure 2A:") -- not
# tied to Kidney International's "Figure N|" house style.
FIGURE_CAPTION_PATTERN = re.compile(r"^\s*(?:Figure|Fig\.?)\s*\d+[A-Za-z]?\s*[|.:]", re.IGNORECASE)

# Structural shape of a Vancouver-style biomedical citation entry (the
# convention KDIGO and most clinical guidelines use): "...Year;Vol(Issue):
# Pages" or a doi/PMID. Narrative prose only rarely contains even one of
# these; a reference-list entry contains one *per citation*. Used as a
# density fallback for reference-list text that isn't caught by a detected
# "References" heading (e.g. Docling missed/mislabeled the heading) --
# generic to citation formatting, not to any single guideline's wording.
CITATION_ENTRY_PATTERN = re.compile(
    r"(?:19|20)\d{2}\s*;\s*\d+(?:\(\d+\))?\s*:\s*\d+" r"|\bdoi:\s*10\.\d{4,9}/\S+" r"|\bPMID:\s*\d+",
    re.IGNORECASE,
)
REFERENCE_DENSITY_THRESHOLD = 2

# ASCII control characters (except tab/newline) left over after any
# source-specific glyph fixes -- never meaningful in guideline text.
CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

logger = logging.getLogger(__name__)


@dataclass
class FigureCoverage:
    """Per-document accounting of picture items, logged at ingest.

    The warning signs for a new document are ``without_any_text`` (an image
    with no extractable text at all is a raster -- a photo, chart, or scanned
    figure) and ``captioned_without_text`` (a numbered figure with little
    text: probably rasterized, or encoding meaning in color like KDIGO's risk
    heat map). Both are vision-model-fallback candidates, not silent drops.
    Docling often doesn't link captions to pictures, so the captioned count
    undercounts; ``without_any_text`` doesn't depend on caption linking.
    """

    pictures: int = 0
    indexed: int = 0
    dropped_low_text: int = 0
    without_any_text: int = 0
    captioned_without_text: int = 0


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


def _fix_chars(text: str, source: GuidelineSource) -> str:
    """Apply the source's glyph fixes, then drop any remaining control characters.

    Whitespace is left alone so table markdown keeps its line structure.
    """
    if source.char_replacements:
        text = text.translate(str.maketrans(source.char_replacements))
    return CONTROL_CHAR_PATTERN.sub("", text)


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


def _looks_like_reference_list(text: str) -> bool:
    """Density-based fallback: is this narrative window citation-list text?

    Real narrative prose almost never contains two or more Vancouver-style
    citation endings in one window; a reference-list entry has one per
    citation. Only applied to text with no recommendation anchor -- a real
    recommendation statement never has this shape.
    """
    return len(CITATION_ENTRY_PATTERN.findall(text)) >= REFERENCE_DENSITY_THRESHOLD


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


def _dedupe_tables(chunks: list[Chunk]) -> tuple[list[Chunk], int]:
    """Drop exact repeats of a table or figure (normalized text), keeping the first.

    Guidelines re-render the same figure/table in later chapters that refer
    back to it (e.g. Glomerular Diseases' Figure 8 appears in both the
    general chapter and a disease chapter, differing by one space). The
    first occurrence is the canonical, defining location.
    """
    seen: set[str] = set()
    kept: list[Chunk] = []
    for chunk in chunks:
        if chunk.label_type in ("Table", "Figure"):
            key = _narrative_key(chunk.text)
            if key in seen:
                continue
            seen.add(key)
        kept.append(chunk)
    return kept, len(chunks) - len(kept)


def _narrative_key(text: str) -> str:
    """Aggressive normalization for exact-duplicate detection: lowercase alphanumerics only."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _is_near_duplicate(words: list[str], candidates: list[list[str]], threshold: float) -> bool:
    """Whether ``words`` matches any candidate at word-sequence similarity >= ``threshold``.

    Cheap upper bounds (length ratio, then ``quick_ratio``) prune almost
    every pair before the full ``ratio`` computation runs.
    """
    for other in candidates:
        if 2 * min(len(words), len(other)) / (len(words) + len(other)) < threshold:
            continue
        matcher = SequenceMatcher(None, words, other, autojunk=False)
        if matcher.quick_ratio() >= threshold and matcher.ratio() >= threshold:
            return True
    return False


def _dedupe_narrative(chunks: list[Chunk], source: GuidelineSource) -> tuple[list[Chunk], int]:
    """Drop narrative segments that repeat a later segment, exactly or nearly.

    KDIGO restates narrative text (summary boxes, figure text) the same way it
    restates recommendations, sometimes with a word or two edited -- e.g. "are
    present" vs. "is" in an otherwise identical 88-word paragraph. Unlike
    :func:`_dedupe_recommendations`, the *last* occurrence is kept: the
    chapter-body copy, which gets packed alongside its own rationale rather
    than alongside unrelated summary bullets.

    Clinically distinct but similar text (e.g. two SGLT2i recommendations
    with different populations scored ~0.6; two evidence tables differing by
    "with/without diabetes" ~0.84) stays below the default 0.95 threshold.
    Segments with no alphanumeric content at all (e.g. a lone "-") are dropped.

    Returns:
        The kept chunks in reading order, and the number dropped.
    """
    seen_keys: set[str] = set()
    seen_words: list[list[str]] = []
    kept_reversed: list[Chunk] = []
    for chunk in reversed(chunks):
        if chunk.label_type is not None:
            kept_reversed.append(chunk)
            continue
        key = _narrative_key(chunk.text)
        if not key or key in seen_keys:
            continue
        words = re.findall(r"[a-z0-9]+", chunk.text.lower())
        long_enough = len(words) >= source.near_duplicate_min_words
        if long_enough and _is_near_duplicate(words, seen_words, source.near_duplicate_threshold):
            continue
        seen_keys.add(key)
        if long_enough:
            seen_words.append(words)
        kept_reversed.append(chunk)
    return kept_reversed[::-1], len(chunks) - len(kept_reversed)


def _absorb_fragments(chunks: list[Chunk], source: GuidelineSource) -> list[Chunk]:
    """Attach short narrative fragments to a neighbor; leave substantive segments alone.

    Docling emits one item per paragraph, bullet, or stray table-header cell,
    so many narrative segments are a few words long ("Outcomes", "the
    rate"). Segments of at least ``narrative_standalone_words`` keep their
    own chunk -- a self-contained bullet like "CKD G4 and G5: ... less
    certainty around the benefit of a lower BP target" must stay precisely
    embedded, not diluted inside a window about something else.

    Within a run of narrative segments sharing chapter/section (a run ends at
    any recommendation/practice point/table), fragments accumulate and are
    prepended to the next standalone segment -- a short lead-in usually
    introduces what follows. Fragments left at the end of a run are appended
    to the run's last standalone chunk. If the accumulated fragments reach
    the standalone size themselves, they become their own chunk; no merge
    may exceed ``narrative_chunk_size``.
    """
    out: list[Chunk] = []
    pending: list[Chunk] = []
    last_standalone_idx: int | None = None  # index into ``out``, within the current run

    def words(text: str) -> int:
        return len(text.split())

    def pending_text() -> str:
        return " ".join(c.text for c in pending)

    def end_run() -> None:
        nonlocal last_standalone_idx
        if pending:
            tail = pending_text()
            prev = out[last_standalone_idx] if last_standalone_idx is not None else None
            if prev is not None and words(prev.text) + words(tail) <= source.narrative_chunk_size:
                out[last_standalone_idx] = replace(prev, text=f"{prev.text} {tail}")
            else:
                out.append(replace(pending[0], text=tail))
            pending.clear()
        last_standalone_idx = None

    for chunk in chunks:
        if chunk.label_type is not None:
            end_run()
            out.append(chunk)
            continue
        if pending and (chunk.chapter, chunk.section) != (pending[0].chapter, pending[0].section):
            end_run()
        elif last_standalone_idx is not None:
            prev = out[last_standalone_idx]
            if (chunk.chapter, chunk.section) != (prev.chapter, prev.section):
                end_run()

        if words(chunk.text) < source.narrative_standalone_words:
            pending.append(chunk)
            if words(pending_text()) >= source.narrative_standalone_words:
                out.append(replace(pending[0], text=pending_text()))
                pending.clear()
                last_standalone_idx = len(out) - 1
            continue

        if pending and words(pending_text()) + words(chunk.text) <= source.narrative_chunk_size:
            out.append(replace(pending[0], text=f"{pending_text()} {chunk.text}"))
        else:
            if pending:
                out.append(replace(pending[0], text=pending_text()))
            out.append(chunk)
        pending.clear()
        last_standalone_idx = len(out) - 1
    end_run()
    return out


def _pack_narrative(chunks: list[Chunk], source: GuidelineSource) -> list[Chunk]:
    """Pack consecutive narrative segments in the same chapter/section into windows.

    Docling emits one item per paragraph, bullet, or stray table-header cell,
    so unpacked narrative chunks are often a few words long ("Outcomes",
    "the rate"). A window closes when adding the next segment would pass
    ``narrative_target_words``, when chapter/section changes, or when a
    recommendation/practice point/table intervenes -- so a window never
    spans two citation contexts. A window that closes under
    ``narrative_min_words`` is folded into the preceding window of the same
    section when the result stays within ``narrative_chunk_size``.
    """
    packed: list[Chunk] = []
    buffer: list[Chunk] = []

    def word_count(text: str) -> int:
        return len(text.split())

    def flush() -> None:
        if not buffer:
            return
        window = replace(buffer[0], text=" ".join(c.text for c in buffer))
        buffer.clear()
        prev = packed[-1] if packed else None
        if (
            word_count(window.text) < source.narrative_min_words
            and prev is not None
            and prev.label_type is None
            and (prev.chapter, prev.section) == (window.chapter, window.section)
            and word_count(prev.text) + word_count(window.text) <= source.narrative_chunk_size
        ):
            packed[-1] = replace(prev, text=f"{prev.text} {window.text}")
            return
        packed.append(window)

    for chunk in chunks:
        if chunk.label_type is not None:
            flush()
            packed.append(chunk)
            continue
        if buffer:
            same_context = (chunk.chapter, chunk.section) == (buffer[0].chapter, buffer[0].section)
            buffered_words = sum(word_count(c.text) for c in buffer)
            if not same_context or buffered_words + word_count(chunk.text) > source.narrative_target_words:
                flush()
        buffer.append(chunk)
    flush()
    return packed


def chunk_document(doc: DoclingDocument, source: GuidelineSource) -> list[Chunk]:
    """Chunk a parsed Docling document per ``source``'s configuration.

    Args:
        doc: The Docling document produced from ``source.pdf_path``.
        source: Guideline-specific chunking configuration.

    Returns:
        Recommendation-anchored, table, and packed narrative chunks, in
        reading order.
    """
    segments, coverage = _extract_segments(doc, source)
    segments = _dedupe_recommendations(segments)
    segments, tables_dropped = _dedupe_tables(segments)
    narrative_before = sum(c.label_type is None for c in segments)
    segments, dropped = _dedupe_narrative(segments, source)
    pack = _absorb_fragments if source.narrative_packing == "absorb" else _pack_narrative
    chunks = pack(segments, source)
    logger.info(
        "%s: dropped %d duplicate tables/figures and %d/%d narrative segments as exact/near duplicates; "
        "%s packing -> %d narrative chunks",
        source.name,
        tables_dropped,
        dropped,
        narrative_before,
        source.narrative_packing,
        sum(c.label_type is None for c in chunks),
    )
    logger.info(
        "%s: figures -- %d pictures, %d indexed, %d dropped for too little text; "
        "vision-fallback candidates: %d with no extractable text, %d captioned but text-poor",
        source.name,
        coverage.pictures,
        coverage.indexed,
        coverage.dropped_low_text,
        coverage.without_any_text,
        coverage.captioned_without_text,
    )
    return chunks


def _figure_chunks(
    item: object,
    doc: DoclingDocument,
    source: GuidelineSource,
    chapter: str | None,
    section: str | None,
    coverage: FigureCoverage,
) -> list[Chunk]:
    """Turn one Docling ``picture`` into Figure chunks from its own extracted text.

    Uses Docling's generic descendant traversal (nested groups included) and
    its linked caption, falling back to a caption-shaped child -- no
    publisher-specific layout assumptions. Text is kept verbatim (only glyph
    fixes and whitespace collapse): for clinical criteria, the source's own
    words beat a model-written rewrite, even if flowchart text reads in
    layout order.
    """
    coverage.pictures += 1
    caption = _fix_chars(item.caption_text(doc) or "", source).strip()
    body_parts: list[str] = []
    for child, _level in doc.iterate_items(root=item, traverse_pictures=True):
        if child is item:
            continue
        text = getattr(child, "text", None)
        if not text:
            continue
        text = _strip_noise(_fix_chars(text, source), source)
        if getattr(child, "label", None) == "caption" or FIGURE_CAPTION_PATTERN.match(text):
            caption = caption or text
            continue
        if text:
            body_parts.append(text)

    body = " ".join(body_parts)
    captioned = bool(FIGURE_CAPTION_PATTERN.match(caption))
    min_words = source.figure_min_words_captioned if captioned else source.figure_min_words_uncaptioned
    if len(body.split()) < min_words:
        coverage.dropped_low_text += 1
        coverage.without_any_text += not body
        coverage.captioned_without_text += captioned
        return []

    coverage.indexed += 1
    prov = getattr(item, "prov", None)
    return [
        Chunk(
            text=f"{caption}\n\n{piece}" if caption else piece,
            source_guideline=source.name,
            chapter=chapter,
            section=section,
            label_type="Figure",
            page=prov[0].page_no if prov else None,
        )
        for piece in _split_narrative(body, source)
    ]


def _extract_segments(doc: DoclingDocument, source: GuidelineSource) -> tuple[list[Chunk], FigureCoverage]:
    """Walk Docling items in reading order, emitting one segment per
    recommendation anchor, table, figure, or narrative item (before dedupe/packing).
    """
    chunks: list[Chunk] = []
    coverage = FigureCoverage()
    chapter: str | None = None
    section: str | None = None
    in_references = False

    for item, _level in doc.iterate_items():
        label = getattr(item, "label", None)

        if label == "table":
            if in_references:
                continue
            table_md = _fix_chars(item.export_to_markdown(doc), source).strip()
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

        if label == "picture":
            if not in_references:
                chunks.extend(_figure_chunks(item, doc, source, chapter, section, coverage))
            continue

        text = getattr(item, "text", None)
        if not text:
            continue
        text = _fix_chars(text, source)

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
            # A "References"/"Bibliography" heading opens a run of citation
            # entries to drop wholesale, until the next real heading.
            if source.reference_heading_pattern and source.reference_heading_pattern.match(stripped):
                in_references = True
                continue
            is_mislabeled_recommendation = (
                source.recommendation_pattern and source.recommendation_pattern.search(stripped)
            )
            if not is_mislabeled_recommendation:
                chapter, section = _update_heading(stripped, chapter, section)
                in_references = False
                continue
            in_references = False

        if label not in ("text", "list_item", "section_header"):
            continue

        # Skip citation entries outright -- checked before the numbered-
        # heading fallback below, since a numbered reference ("1. Smith J,
        # et al...") would otherwise be misread as a section heading.
        if in_references:
            continue

        text = _strip_noise(text, source)
        if not text:
            continue

        # Docling sometimes tags a numbered subsection heading as a plain
        # text/list_item instead of section_header (varies by document).
        if source.heading_fallback_pattern and source.heading_fallback_pattern.fullmatch(text):
            chapter, section = _update_heading(text, chapter, section)
            continue

        if source.reference_heading_pattern and source.reference_heading_pattern.match(text):
            in_references = True
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
                if _looks_like_reference_list(piece):
                    continue
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

    return chunks, coverage
