"""Unit tests for onboarding building blocks -- no PDFs, no API calls."""

import json
import re

from kdigo_guideline_rag.ingest.profiles import (
    load_profile_sources,
    profile_from_source,
    sources_from_profile,
)
from kdigo_guideline_rag.ingest.scorecard import numbering_gaps
from kdigo_guideline_rag.ingest.sources import GuidelineSource

KDIGO_SOURCES = [s for s in load_profile_sources() if s.name.startswith("KDIGO")]
from kdigo_guideline_rag.ingest.text_layer import _bound, is_complete, is_valid_start

ADA_LIKE = GuidelineSource(
    name="ADA",
    pdf_path="unused.pdf",
    recommendation_pattern=re.compile(r"(?P<number>\d{1,2}\.\d{1,2}[a-z]?)\s+(?=[A-Z])"),
    recommendation_label="Recommendation",
    grade_pattern=re.compile(r"(?<=\s)([ABCE])(?=\s|$)"),
)


def test_numbering_gaps_numeric_and_lettered() -> None:
    assert numbering_gaps({"3.7.1", "3.7.3", "3.8.1"}) == ["3.7.2"]
    assert numbering_gaps({"1.1a", "1.1b", "1.1d", "1.2"}) == ["1.1c"]
    assert numbering_gaps({"1.1", "1.2", "2.1"}) == []


def test_bare_number_starts_need_context_but_labelled_ones_dont() -> None:
    text = "HbA1c 7.0 In trials, ... insulin. A 9.2 For most adults"
    starts = [
        m.group("number")
        for m in ADA_LIKE.recommendation_pattern.finditer(text)
        if is_valid_start(text, m, ADA_LIKE)
    ]
    assert starts == ["9.2"]  # "7.0 In" is a dose mid-sentence; "9.2" follows grade "A"
    kdigo = next(s for s in KDIGO_SOURCES if "CKD Evaluation" in s.name)
    merged_heading = "1.2. Staging of AKI Practice Point 1.2.1: Stage AKI by..."
    m = kdigo.recommendation_pattern.search(merged_heading)
    assert is_valid_start(merged_heading, m, kdigo)  # Docling merged the heading into the item


def test_completeness_rejects_cut_offs_and_list_lead_ins() -> None:
    assert is_complete("11.7a ... can safely continue until kidney failure. A", ADA_LIKE)
    assert not is_complete("11.7a ... SGLT2 inhibitors shoul", ADA_LIKE)
    kdigo = next(s for s in KDIGO_SOURCES if "CKD Evaluation" in s.name)
    assert is_complete("Recommendation 3.7.1: We recommend ... SGLT2i (1A).", kdigo)
    assert not is_complete("Practice Point 1.1.3.1: Proof of chronicity can be established by:", kdigo)
    assert not is_complete("Practice Point 1.1.3.1: ... established by: (i)", kdigo)


def test_text_layer_recommendation_ends_at_a_caption_line_but_not_a_wrapped_reference() -> None:
    ada = GuidelineSource(
        name="ADA", pdf_path="unused.pdf", grade_pattern=re.compile(r"[.)\]]\s+([ABCE])(?=\s|$)")
    )
    raw = "9.7 Include agents with benefit (Fig. 9.4 and\nTable 9.2 ). A\nRationale follows."
    assert _bound(raw, ada) == "9.7 Include agents with benefit (Fig. 9.4 and Table 9.2 ). A"
    raw = "9.12 Consider a GLP-1 RA A or a dual\nFlowchart label\nFigure 9.3—Simplified overview. B\n"
    assert _bound(raw, ada) == "9.12 Consider a GLP-1 RA A or a dual Flowchart label"


def test_profile_round_trip_through_json_preserves_behavior() -> None:
    for src in KDIGO_SOURCES:
        back = sources_from_profile(
            json.loads(json.dumps(profile_from_source(src))), src.name, [src.pdf_path]
        )[0]
        assert back.text_substitutions == src.text_substitutions
        assert back.char_replacements == src.char_replacements
        for text in ("CHAPTER 1", "chapter 2", "(1B)", "(not graded)", "References"):
            for f in ("chapter_pattern", "grade_pattern", "reference_heading_pattern"):
                assert bool(getattr(src, f).search(text)) == bool(getattr(back, f).search(text)), (f, text)


def test_profile_rejects_recommendation_pattern_without_number_group() -> None:
    import pytest

    with pytest.raises(ValueError, match="number"):
        sources_from_profile({"recommendation_pattern": r"Recommendation \d+"}, "X", ["x.pdf"])


def test_selector_prefers_passing_then_fewest_problems(monkeypatch) -> None:
    from kdigo_guideline_rag.ingest import onboard

    class _Ev:
        def __init__(self, passed, problems):
            self.passed, self.problems = passed, problems

    scores = {
        ("a", "docling"): _Ev(False, 1.0),
        ("a", "text_layer"): _Ev(True, 5.0),
        ("b", "docling"): _Ev(True, 2.0),
    }
    monkeypatch.setattr(
        onboard,
        "evaluate",
        lambda profile, name, pdfs: scores.get(
            (profile["id"], profile["recommendation_extractor"]), _Ev(False, 99)
        ),
    )
    best, _ = onboard.select("X", [], {"a": {"id": "a"}, "b": {"id": "b"}})
    assert best == "b/docling"
