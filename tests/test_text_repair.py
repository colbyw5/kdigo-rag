"""Unit tests for dictionary-guided PDF text repair (uses wordsegment's bundled data, no network)."""

import pytest

from kdigo_guideline_rag.ingest.text_repair import repair_missing_spaces, repair_split_ligatures


@pytest.mark.parametrize(
    "broken,fixed",
    [
        ("clear bene fi ts of", "clear benefits of"),
        ("standardized of fi ce BP", "standardized office BP"),  # word-initial: join left
        ("the fi rst line", "the first line"),  # word-initial: keep left separate
        ("glomerular fi ltration rate", "glomerular filtration rate"),
        ("risk of in fl ammation", "risk of inflammation"),
        ("Empagli fl ozin reduced", "Empagliflozin reduced"),  # unknown drug name: join all
        ("SGLT2i and fi nerenone", "SGLT2i and finerenone"),  # unknown, but "and" is a word
        ("tubulointerstitial fi brosis", "tubulointerstitial fibrosis"),
        ("speciﬁc ﬂuid", "specific fluid"),  # Unicode ligature code points
    ],
)
def test_repair_split_ligatures(broken: str, fixed: str) -> None:
    assert repair_split_ligatures(broken) == fixed


def test_ligature_repair_leaves_table_rows_intact() -> None:
    table = "| bene fi t |\n| fi rst |"
    assert repair_split_ligatures(table) == "| benefit |\n| first |"


def test_missing_spaces_repaired_with_case_and_acronyms_preserved() -> None:
    """The real BP Rec 3.1.1 extraction artifact."""
    broken = "Recommendation 3.1.1: WesuggestthatadultswithhighBPandCKDbetreatedwithatargetsystolicbloodpressure (SBP)"
    assert repair_missing_spaces(broken) == (
        "Recommendation 3.1.1: We suggest that adults with high BP and CKD be treated with a target "
        "systolic blood pressure (SBP)"
    )


def test_long_single_words_are_not_split() -> None:
    """No function word in the segmentation -> not merged prose -> leave it."""
    for word in ("ethylenediaminetetraacetic", "nonimmunosuppressivebased", "immunohistochemistrybased"):
        assert repair_missing_spaces(word) == word


def test_short_runs_untouched() -> None:
    text = "Glomerulonephritis and cyclophosphamide are normal long words."
    assert repair_missing_spaces(text) == text


@pytest.mark.parametrize(
    "broken,fixed",
    [
        (
            "ArecentKDIGOControversiesConferencelistedthefollowing",
            "A recent KDIGO Controversies Conference listed the following",
        ),
        ("NSwilldevelopspontaneousremission", "NS will develop spontaneous remission"),
        ("mycophenolatemofetilplusglucocorticoid", "mycophenolate mofetil plus glucocorticoid"),
    ],
)
def test_acronym_boundaries_and_regimen_joiners(broken: str, fixed: str) -> None:
    """Real corpus runs: the last capital of an acronym run may start the next word or not."""
    assert repair_missing_spaces(broken) == fixed
