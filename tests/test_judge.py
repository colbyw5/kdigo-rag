"""Unit tests for tier-3 deterministic checks and summary -- no API calls."""

from kdigo_guideline_rag.eval.judge import (
    ASKS_FOR_CONTEXT,
    OFFERS_OUTSIDE_KNOWLEDGE,
    check_citations,
    summarize,
)

PASSAGES = [
    (
        "KDIGO 2024 CKD Evaluation and Management, Chapter 3, Recommendation 3.4.1",
        "Recommendation 3.4.1: We suggest ... SBP of < 120 mm Hg (2B).",
    ),
    (
        "KDIGO 2021 Glomerular Diseases, Chapter 1",
        "Figure 8| Management of hypertension ... Practice Point 1.5.2: target SBP < 120",
    ),
]


def test_valid_citations_including_shortened_names_and_embedded_practice_points() -> None:
    answer = (
        "Target SBP <120 [KDIGO 2024 CKD Evaluation and Management, Chapter 3, Recommendation 3.4.1]; "
        "same in GN [KDIGO 2021 Glomerular Diseases, Chapter 1, Practice Point 1.5.2; KDIGO 2021 Glomerular, Figure 8]."
    )
    check = check_citations(answer, PASSAGES)
    assert check.total == 3 and check.invalid == []


def test_citation_to_number_not_in_passages_is_invalid() -> None:
    answer = "Start an SGLT2i [KDIGO 2024 CKD Evaluation and Management, Chapter 3, Recommendation 3.7.1]."
    assert check_citations(answer, PASSAGES).invalid == [
        "KDIGO 2024 CKD Evaluation and Management, Chapter 3, Recommendation 3.7.1"
    ]


def test_citation_to_guideline_not_in_passages_is_invalid() -> None:
    answer = "Use buffered crystalloids [KDIGO 2026 AKI/AKD, Chapter 3, Rec 3.1.2]."
    assert len(check_citations(answer, PASSAGES).invalid) == 1


def test_blood_pressure_guideline_not_mistaken_for_ckd() -> None:
    """'Blood Pressure in CKD' contains 'CKD' -- must resolve to the BP guideline."""
    answer = "Target <120 [KDIGO 2021 Blood Pressure in CKD, Chapter 3, Recommendation 3.4.1]."
    assert len(check_citations(answer, PASSAGES).invalid) == 1


def test_leakage_regexes_catch_observed_failures() -> None:
    """Both phrasings were produced by the real pipeline (oos_3 / oos_5 baseline answers)."""
    assert ASKS_FOR_CONTEXT.search("Please provide the relevant transplant guideline passages")
    assert OFFERS_OUTSIDE_KNOWLEDGE.search("should I note what general range is commonly referenced")
    assert not ASKS_FOR_CONTEXT.search("Provide SGLT2i therapy to eligible patients.")


def _judged(category: str, **verdict) -> dict:
    base = {
        "unsupported_claims": [],
        "transferred_claims": [],
        "faithful": True,
        "addresses_question": True,
        "declined": False,
        "wrongly_declined": False,
        "outside_knowledge": False,
        "asks_user_for_context": False,
        "note": "",
        "materially_faithful": True,
    }
    return {
        "category": category,
        "verdict": {**base, **verdict},
        "citations_invalid": [],
        "regex_asks_for_context": False,
        "regex_offers_outside_knowledge": False,
    }


def test_summary_counts_oos_abstention_only_when_clean() -> None:
    judged = [
        _judged("recommendation"),
        _judged(
            "recommendation",
            faithful=False,
            materially_faithful=False,
            unsupported_claims=[{"quote": "x", "material": True}],
        ),
        _judged("out_of_scope", declined=True),
        _judged("out_of_scope", declined=True, asks_user_for_context=True),
    ]
    s = summarize(judged)
    assert s["in_scope_materially_faithful"] == 0.5
    assert s["in_scope_material_claims"] == 1
    assert s["oos_correct_abstention"] == 0.5
    assert s["leaky_answers"] == 1


def test_swap_number_alters_a_value_shared_with_passages() -> None:
    from kdigo_guideline_rag.eval.judge import _swap_number

    assert _swap_number("Target SBP <120 mm Hg (2B).", "SBP of < 120 mm Hg") == "Target SBP <240 mm Hg (2B)."
    assert _swap_number("Rec 3.4.1 says 5 things.", "no shared numbers here") is None


def test_citation_fragment_inherits_guideline_within_a_bracket() -> None:
    answer = "BP first [KDIGO 2021 Glomerular Diseases, Chapter 1; Figure 8; Practice Point 1.5.2]."
    assert check_citations(answer, PASSAGES).invalid == []


def test_swap_number_skips_citation_years() -> None:
    from kdigo_guideline_rag.eval.judge import _swap_number

    answer = "Use buffered crystalloids [KDIGO 2026 AKI/AKD, Chapter 3] above 65 mm Hg."
    assert (
        _swap_number(answer, "2026 ... MAP > 65 mm Hg")
        == "Use buffered crystalloids [KDIGO 2026 AKI/AKD, Chapter 3] above 130 mm Hg."
    )


def test_judge_run_isolates_api_failures(monkeypatch) -> None:
    """A dropped connection on one row must not abort the run (observed: run 2a)."""
    import anthropic
    import httpx

    from kdigo_guideline_rag.eval import judge

    def fake_judge(row, client):
        if row["id"] == "bad":
            raise anthropic.APIConnectionError(request=httpx.Request("POST", "https://api.anthropic.com"))
        return {"id": row["id"], "verdict": {"ok": True}, "judge_error": None}

    monkeypatch.setattr(judge, "judge_answer", fake_judge)
    monkeypatch.setattr(judge, "_client", lambda: None)
    rows = [{"id": i, "category": "recommendation", "scope": None} for i in ("a", "bad", "c")]
    out = judge.judge_run(rows)
    assert [o["id"] for o in out] == ["a", "bad", "c"]
    assert out[1]["verdict"] is None and "APIConnectionError" in out[1]["judge_error"]
