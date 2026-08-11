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
