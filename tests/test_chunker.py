"""Unit tests for chunk_document() -- no Docling models or API calls.

Docling items are accessed via getattr/duck typing in chunker.py (label,
text, prov[0].page_no, and export_to_markdown(doc) for tables), so a
lightweight fake document lets us exercise the full chunking logic
directly, including the real bugs we found and fixed against actual PDFs.
"""

from types import SimpleNamespace

import pytest

from kdigo_guideline_rag.ingest.chunker import chunk_document
from kdigo_guideline_rag.ingest.sources import (
    KDIGO_RECOMMENDATION_PATTERN,
    NUMBERED_HEADING_PATTERN,
    GuidelineSource,
)


def _item(label, text=None, page=1):
    prov = [SimpleNamespace(page_no=page)] if page is not None else None
    return SimpleNamespace(label=label, text=text, prov=prov)


class _FakeTable:
    def __init__(self, markdown, page=1):
        self.label = "table"
        self.prov = [SimpleNamespace(page_no=page)]
        self._markdown = markdown

    def export_to_markdown(self, doc):
        return self._markdown


class _FakeDoc:
    def __init__(self, items):
        self._items = items

    def iterate_items(self):
        return iter(self._items)


@pytest.fixture
def source():
    return GuidelineSource(name="Test Guideline", pdf_path="unused.pdf")


def test_recommendation_and_practice_point_split(source):
    doc = _FakeDoc(
        [
            (_item("section_header", "Chapter 1: Evaluation"), 1),
            (_item("section_header", "1.1 Detection"), 1),
            (
                _item(
                    "text",
                    "Practice Point 1.1.1: Test people at risk. "
                    "Recommendation 1.1.2: We recommend eGFRcr (1B).",
                ),
                1,
            ),
        ]
    )
    chunks = chunk_document(doc, source)

    assert [c.label_type for c in chunks] == ["Practice Point", "Recommendation"]
    assert chunks[0].recommendation_number == "1.1.1"
    assert chunks[1].recommendation_number == "1.1.2"
    assert chunks[1].grade == "1B"
    assert all(c.chapter == "Chapter 1: Evaluation" for c in chunks)
    assert all(c.section == "1.1 Detection" for c in chunks)


def test_mislabeled_recommendation_falls_through_instead_of_dropped(source):
    """Regression test: Docling sometimes tags a recommendation statement as
    section_header (observed as bold text in the real CKD guideline). It
    must still become a content chunk, not silently vanish.
    """
    doc = _FakeDoc(
        [
            (_item("section_header", "Chapter 1: Evaluation"), 1),
            (_item("section_header", "Recommendation 1.2.2.1: We recommend X (1A)."), 1),
        ]
    )
    chunks = chunk_document(doc, source)

    assert len(chunks) == 1
    assert chunks[0].label_type == "Recommendation"
    assert chunks[0].recommendation_number == "1.2.2.1"


def test_mislabeled_caption_dropped_not_treated_as_heading(source):
    """Regression test: a table's caption sometimes arrives as section_header
    too. It must be dropped, not become a phantom section for later chunks.
    """
    doc = _FakeDoc(
        [
            (_item("section_header", "Chapter 1: Evaluation"), 1),
            (_item("section_header", "1.1 Detection"), 1),
            (_item("section_header", "Table 9| Comparison of eGFR"), 1),
            (_item("text", "Practice Point 1.1.1: Test people at risk."), 1),
        ]
    )
    chunks = chunk_document(doc, source)

    assert len(chunks) == 1
    assert chunks[0].section == "1.1 Detection"  # not the caption


def test_bare_caption_text_item_dropped(source):
    doc = _FakeDoc(
        [
            (_item("text", "Figure 8| Evaluation of cause of chronic kidney disease (CKD)."), 1),
            (_item("text", "Practice Point 1.1.1: Test people at risk."), 1),
        ]
    )
    chunks = chunk_document(doc, source)

    assert len(chunks) == 1
    assert chunks[0].label_type == "Practice Point"


def test_numbered_heading_fallback_updates_section(source):
    """A numbered subsection sometimes arrives as plain text/list_item
    instead of section_header (varies by document).
    """
    doc = _FakeDoc(
        [
            (_item("section_header", "Chapter 1: Evaluation"), 1),
            (_item("list_item", "1.1 Detection and evaluation of CKD"), 1),
            (_item("text", "Practice Point 1.1.1: Test people at risk."), 1),
        ]
    )
    chunks = chunk_document(doc, source)

    assert len(chunks) == 1
    assert chunks[0].section == "1.1 Detection and evaluation of CKD"


def test_table_becomes_its_own_chunk(source):
    doc = _FakeDoc(
        [
            (_item("section_header", "Chapter 1: Evaluation"), 1),
            (_FakeTable("Table 6| Test\n\n| A | B |\n|---|---|\n| 1 | 2 |", page=2), 1),
        ]
    )
    chunks = chunk_document(doc, source)

    assert len(chunks) == 1
    assert chunks[0].label_type == "Table"
    assert chunks[0].page == 2
    assert "Table 6" in chunks[0].text


def test_narrative_text_with_no_anchor_is_kept(source):
    doc = _FakeDoc([(_item("text", "This is background narrative with no recommendation anchor."), 1)])
    chunks = chunk_document(doc, source)

    assert len(chunks) == 1
    assert chunks[0].label_type is None
    assert chunks[0].recommendation_number is None


def test_dedup_keeps_only_first_occurrence(source):
    """KDIGO restates every recommendation twice (summary listing + real
    chapter). Both occurrences are byte-identical, so dedup keeps the first.
    """
    doc = _FakeDoc(
        [
            (_item("section_header", "Summary of recommendation statements"), 1),
            (_item("text", "Recommendation 1.1.1: We recommend X (1A)."), 1),
            (_item("section_header", "Chapter 1: Evaluation"), 2),
            (_item("text", "Recommendation 1.1.1: We recommend X (1A)."), 2),
        ]
    )
    chunks = chunk_document(doc, source)

    assert len(chunks) == 1
    assert chunks[0].page == 1  # kept the first occurrence


def test_noise_pattern_stripped(source):
    noisy_source = GuidelineSource(
        name="Draft Guideline",
        pdf_path="unused.pdf",
        noise_patterns=[__import__("re").compile(r"\bDRAFT\b")],
    )
    doc = _FakeDoc([(_item("text", "Practice Point 1.1.1: DRAFT Test people at risk. DRAFT"), 1)])
    chunks = chunk_document(doc, noisy_source)

    assert "DRAFT" not in chunks[0].text


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Recommendation 1.1.2.1: We recommend X (1B).", ("Recommendation", "1.1.2.1")),
        ("Practice Point 1.1.3.1: Do this.", ("Practice Point", "1.1.3.1")),
        ("Recommendation 1.1: We recommend Y (1A).", ("Recommendation", "1.1")),  # BP's 2-level numbering
        ("Practice Point 1.1.1: Definition text.", ("Practice Point", "1.1.1")),  # AKI's 3-level numbering
    ],
)
def test_kdigo_recommendation_pattern_generalizes_across_docs(text, expected):
    match = KDIGO_RECOMMENDATION_PATTERN.search(text)
    assert match is not None
    assert (match.group("label"), match.group("number")) == expected


@pytest.mark.parametrize(
    "text,should_match",
    [
        ("1.1 Detection and evaluation of CKD", True),
        ("1.4. AKI and AKD resolution", True),
        ("Recommendation 1.1.2.1: We recommend X.", False),
        ("just some narrative text", False),
    ],
)
def test_numbered_heading_pattern(text, should_match):
    assert (NUMBERED_HEADING_PATTERN.fullmatch(text) is not None) == should_match


def test_references_section_dropped_via_heading(source):
    """A detected References/Bibliography heading drops everything under it
    -- citation entries, tables -- until the next real heading.
    """
    doc = _FakeDoc(
        [
            (_item("section_header", "Chapter 1: Evaluation"), 1),
            (_item("text", "Practice Point 1.1.1: Test people at risk."), 1),
            (_item("section_header", "References"), 5),
            (_item("list_item", "1. Smith J, Doe K. A study of CKD. Nephrol. 2020;35(4):567-578."), 5),
            (_FakeTable("| A | B |\n|---|---|\n| 1 | 2 |", page=5), 5),
            (_item("section_header", "Chapter 2: Management"), 6),
            (_item("text", "Practice Point 2.1.1: Manage patients."), 6),
        ]
    )
    chunks = chunk_document(doc, source)

    assert [c.recommendation_number for c in chunks] == ["1.1.1", "2.1.1"]
    assert all("Smith" not in c.text for c in chunks)
    assert chunks[1].chapter == "Chapter 2: Management"


@pytest.mark.parametrize(
    "heading",
    ["References", "REFERENCES", "Bibliography", "12. References", "Works Cited", "Reference List"],
)
def test_reference_heading_pattern_generalizes(heading, source):
    from kdigo_guideline_rag.ingest.sources import REFERENCE_HEADING_PATTERN

    assert REFERENCE_HEADING_PATTERN.match(heading) is not None


def test_reference_heading_via_mistagged_list_item(source):
    """A "References" heading Docling mistags as list_item/text (not
    section_header, same failure mode as numbered subsections) must still
    open reference-skip mode, not get misread as content or a heading.
    """
    doc = _FakeDoc(
        [
            (_item("section_header", "Chapter 1: Evaluation"), 1),
            (_item("list_item", "References"), 5),
            (_item("list_item", "1. Smith J, Doe K. A study. Nephrol. 2020;35(4):567-578."), 5),
        ]
    )
    chunks = chunk_document(doc, source)

    assert chunks == []


def test_citation_dense_narrative_dropped_without_heading(source):
    """Fallback: citation-list text with no detected heading (e.g. Docling
    never emitted a References item at all) is still dropped, based on
    citation-entry density rather than any heading match.
    """
    citation_text = (
        "Smith J, Doe K. A study of CKD. Nephrol. 2020;35(4):567-578. "
        "Lee A, Park B. Outcomes in AKI. Kidney Int. 2019;96(2):210-220."
    )
    doc = _FakeDoc([(_item("text", citation_text), 5)])
    chunks = chunk_document(doc, source)

    assert chunks == []


def test_single_inline_citation_not_treated_as_reference_list(source):
    """One citation-shaped fragment inside real narrative prose isn't
    enough to trigger the density fallback -- avoids false positives on
    genuine discussion text that happens to cite one study inline.
    """
    text = "Discussion of eGFR trends references one key study (Lee A. Kidney Int. 2019;96(2):210-220.) directly."
    doc = _FakeDoc([(_item("text", text), 1)])
    chunks = chunk_document(doc, source)

    assert len(chunks) == 1


def _words(n: int, word: str = "word") -> str:
    return " ".join(f"{word}{i}" for i in range(n))


def test_ki_symbol_font_glyphs_fixed_in_text_and_tables():
    """Kidney International's symbol font decodes as wrong glyphs (>= as a
    double dagger, <= as a pound sign, bullets as \\x0f). Clinical thresholds
    must reach the index intact, in prose and tables alike.
    """
    from kdigo_guideline_rag.ingest.sources import KI_SYMBOL_FONT_MAP

    ki_source = GuidelineSource(name="KI", pdf_path="unused.pdf", char_replacements=KI_SYMBOL_FONT_MAP)
    doc = _FakeDoc(
        [
            (_item("text", "\x0f Confirm ACR ‡ 30 mg/g with SBP target £ 120 mm Hg (n ¼ 2646)."), 1),
            (_FakeTable("| eGFR |\n|---|\n| 47.9 \x06 9.5 |"), 1),
        ]
    )
    narrative, table = chunk_document(doc, ki_source)

    assert narrative.text == "• Confirm ACR ≥ 30 mg/g with SBP target ≤ 120 mm Hg (n = 2646)."
    assert "47.9 ± 9.5" in table.text
    assert "\n" in table.text  # markdown line structure preserved


def test_control_chars_stripped_without_source_mapping(source):
    """Sources without a glyph map still never index raw control characters."""
    doc = _FakeDoc([(_item("text", "Dr Kazancio\x15glu chaired the work group."), 1)])
    (chunk,) = chunk_document(doc, source)

    assert chunk.text == "Dr Kazancioglu chaired the work group."


@pytest.fixture
def greedy_source():
    return GuidelineSource(name="Test Guideline", pdf_path="unused.pdf", narrative_packing="greedy")


def test_consecutive_short_narrative_items_packed_into_one_window(source):
    doc = _FakeDoc(
        [
            (_item("section_header", "1.1 Detection"), 1),
            (_item("text", "Outcomes"), 1),
            (_item("list_item", "Assess kidney function annually."), 1),
            (_item("text", "Consider albuminuria testing in people at risk."), 2),
        ]
    )
    (chunk,) = chunk_document(doc, source)

    assert chunk.text == (
        "Outcomes Assess kidney function annually. Consider albuminuria testing in people at risk."
    )
    assert chunk.page == 1  # first segment's page
    assert chunk.section == "1.1 Detection"


def test_packing_never_crosses_a_section_or_recommendation(greedy_source):
    doc = _FakeDoc(
        [
            (_item("section_header", "1.1 Detection"), 1),
            (_item("text", _words(60, "a")), 1),
            (_item("text", "Practice Point 1.1.1: Test people at risk."), 1),
            (_item("text", _words(60, "b")), 1),
            (_item("section_header", "1.2 Staging"), 2),
            (_item("text", _words(60, "c")), 2),
        ]
    )
    chunks = chunk_document(doc, greedy_source)

    assert [c.label_type for c in chunks] == [None, "Practice Point", None, None]
    assert [c.section for c in chunks] == ["1.1 Detection"] * 3 + ["1.2 Staging"]


def test_packing_closes_window_at_target_size(greedy_source):
    """Five 100-word paragraphs at a 250-word target -> 200 + 200 + 100."""
    paragraphs = [_words(100, f"p{i}w") for i in range(5)]
    doc = _FakeDoc([(_item("text", p), 1) for p in paragraphs])
    chunks = chunk_document(doc, greedy_source)

    assert [len(c.text.split()) for c in chunks] == [200, 200, 100]


def test_small_trailing_window_folds_into_previous_same_section(greedy_source):
    """240 words fill a window; a 30-word scrap then closes at the section
    boundary and is folded back rather than left as a fragment chunk.
    """
    doc = _FakeDoc(
        [
            (_item("section_header", "1.1 Detection"), 1),
            (_item("text", _words(240, "a")), 1),
            (_item("text", _words(30, "b")), 1),
            (_item("section_header", "1.2 Staging"), 2),
            (_item("text", _words(60, "c")), 2),
        ]
    )
    chunks = chunk_document(doc, greedy_source)

    assert [len(c.text.split()) for c in chunks] == [270, 60]
    assert [c.section for c in chunks] == ["1.1 Detection", "1.2 Staging"]


def test_exact_duplicate_narrative_keeps_last_occurrence(source):
    """Summary-box text restated in the chapter body: keep the body copy,
    which packs alongside its own rationale. Normalization ignores case,
    whitespace, and punctuation differences between the two copies.
    """
    doc = _FakeDoc(
        [
            (_item("section_header", "Summary of recommendation statements"), 1),
            (_item("text", "Patients with rapidly progressive IgAN should be offered treatment."), 1),
            (_item("section_header", "2.4 Special situations"), 50),
            (
                _item("text", "Patients with rapidly progressive  IgAN should be offered treatment", page=50),
                1,
            ),
        ]
    )
    (chunk,) = chunk_document(doc, source)

    assert chunk.page == 50
    assert chunk.section == "2.4 Special situations"


def test_near_duplicate_narrative_dropped():
    """Same paragraph with one phrase edited ("are present" -> "is"), as
    observed between CKD's summary and chapter-body copies.
    """
    source = GuidelineSource(name="Test", pdf_path="unused.pdf", narrative_min_words=1)
    body = " ".join(f"w{i}" for i in range(80))
    doc = _FakeDoc(
        [
            (_item("section_header", "Summary"), 1),
            (_item("text", f"Risk factors are present when {body}"), 1),
            (_item("section_header", "3.7 SGLT2i"), 44),
            (_item("text", f"Risk factors is when {body}", page=44), 1),
        ]
    )
    (chunk,) = chunk_document(doc, source)

    assert chunk.page == 44


def test_clinically_distinct_similar_text_not_deduped():
    """Two evidence tables' titles differing only by population (~0.84
    similarity in the real corpus) must both survive.
    """
    source = GuidelineSource(name="Test", pdf_path="unused.pdf", narrative_standalone_words=1)
    doc = _FakeDoc(
        [
            (
                _item(
                    "text",
                    "Table S11. Adults with CKD, low blood pressure target versus standard target, all outcomes",
                ),
                1,
            ),
            (
                _item(
                    "text",
                    "Table S14. Adults with CKD without diabetes, low blood pressure target versus standard target",
                ),
                1,
            ),
        ]
    )
    chunks = chunk_document(doc, source)

    assert len(chunks) == 2


def test_short_near_duplicates_only_deduped_on_exact_match(source):
    """Below near_duplicate_min_words, similarity is too unstable to trust --
    "Stage 1 AKI" vs "Stage 2 AKI" must not collapse.
    """
    short_source = GuidelineSource(name="Test", pdf_path="unused.pdf", narrative_standalone_words=1)
    doc = _FakeDoc([(_item("text", "Stage 1 AKI criteria"), 1), (_item("text", "Stage 2 AKI criteria"), 1)])
    chunks = chunk_document(doc, short_source)

    assert [c.text for c in chunks] == ["Stage 1 AKI criteria", "Stage 2 AKI criteria"]


def test_punctuation_only_narrative_dropped(source):
    doc = _FakeDoc([(_item("text", "-"), 1), (_item("text", "Real narrative text."), 1)])
    (chunk,) = chunk_document(doc, source)

    assert chunk.text == "Real narrative text."


def test_absorb_keeps_substantive_segments_standalone(source):
    """Regression for the measured greedy-packing failure: a self-contained
    30+ word bullet must stay its own chunk, not be diluted into a window.
    """
    doc = _FakeDoc(
        [
            (_item("section_header", "3.1 BP targets"), 1),
            (_item("text", _words(40, "a")), 1),
            (_item("list_item", _words(35, "b")), 1),
            (_item("list_item", _words(30, "c")), 1),
        ]
    )
    chunks = chunk_document(doc, source)

    assert [len(c.text.split()) for c in chunks] == [40, 35, 30]


def test_absorb_prepends_fragments_to_next_standalone_segment(source):
    """A short lead-in ("Pediatric considerations.") introduces what follows."""
    doc = _FakeDoc(
        [
            (_item("text", "Pediatric considerations."), 1),
            (_item("text", _words(40, "a")), 1),
        ]
    )
    (chunk,) = chunk_document(doc, source)

    assert chunk.text.startswith("Pediatric considerations. a0")


def test_absorb_appends_trailing_fragments_to_previous_standalone(source):
    doc = _FakeDoc(
        [
            (_item("section_header", "1.1 Detection"), 1),
            (_item("text", _words(40, "a")), 1),
            (_item("text", "the rate"), 1),
            (_item("text", "Practice Point 1.1.1: Test people at risk."), 1),
        ]
    )
    narrative, practice_point = chunk_document(doc, source)

    assert narrative.text.endswith("a39 the rate")
    assert practice_point.label_type == "Practice Point"


def test_absorb_fragments_never_cross_sections(source):
    """A trailing fragment must not attach to a chunk with a different citation context."""
    doc = _FakeDoc(
        [
            (_item("section_header", "1.1 Detection"), 1),
            (_item("text", _words(40, "a")), 1),
            (_item("section_header", "1.2 Staging"), 2),
            (_item("text", "Outcomes"), 2),
        ]
    )
    chunks = chunk_document(doc, source)

    assert [(c.section, len(c.text.split())) for c in chunks] == [("1.1 Detection", 40), ("1.2 Staging", 1)]


def test_absorb_run_of_fragments_becomes_own_chunk_at_standalone_size(source):
    """Many tiny fragments (e.g. table-header cells emitted as text) that add
    up to standalone size form their own chunk rather than bloating a neighbor.
    """
    cells = [(_item("text", f"cell{i} label"), 1) for i in range(15)]  # 2 words each -> 30
    doc = _FakeDoc([*cells, (_item("text", _words(40, "a")), 1)])
    chunks = chunk_document(doc, source)

    assert [len(c.text.split()) for c in chunks] == [30, 40]


def test_duplicate_table_keeps_first_occurrence(source):
    """The same figure re-rendered in a later chapter, differing by whitespace."""
    doc = _FakeDoc(
        [
            (_item("section_header", "Chapter 1: General"), 1),
            (_FakeTable("Figure 8| Management\n\n| ACEi | ARB |", page=39), 1),
            (_item("section_header", "Chapter 2: IgAN"), 1),
            (_FakeTable("Figure 8| Management\n\n|  ACEi | ARB |", page=102), 1),
        ]
    )
    (table,) = chunk_document(doc, source)

    assert table.page == 39
    assert table.chapter == "Chapter 1: General"
