"""Runs EVAL_QUESTIONS against the real graph and reports results.

Strict questions pass/fail on whether the expected citations still surface
after grading -- a regression check for retrieval + grading quality. Loose
questions have no pass/fail: the new answer is printed next to the recorded
baseline so a human can judge whether the answer is still clinically sound
and appropriately hedged, which isn't something a citation match can prove.
"""

from kdigo_guideline_rag.eval.questions import EVAL_QUESTIONS, EvalQuestion
from kdigo_guideline_rag.graph import graph


def _run_one(q: EvalQuestion) -> tuple[bool | None, str, list[tuple[str | None, str | None]]]:
    """Invoke the graph for one question.

    Returns:
        (passed, answer, citations). ``passed`` is ``None`` for loose
        questions (no pass/fail verdict), else a strict-question bool.
    """
    config = {"configurable": {"guideline_filter": "all", "user_id": f"eval-{q.id}"}}
    result = graph.invoke({"question": q.question}, config=config)
    citations = [
        (d.metadata.get("source_guideline"), d.metadata.get("recommendation_number"))
        for d in result["graded_documents"]
    ]
    if q.grading == "loose":
        return None, result["answer"], citations

    actual_guidelines = {g for g, _ in citations}
    actual_recs = {r for _, r in citations if r}
    guidelines_ok = set(q.expected_guidelines) <= actual_guidelines
    recs_ok = set(q.expected_recommendation_numbers) <= actual_recs
    return guidelines_ok and recs_ok, result["answer"], citations


def run_eval() -> bool:
    """Run the full eval set, print a report, return True iff all strict cases passed."""
    all_strict_passed = True
    for q in EVAL_QUESTIONS:
        passed, answer, citations = _run_one(q)
        print(f"\n{'=' * 70}\n[{q.id}] ({q.category}, {q.grading}) {q.question}")
        print(f"citations: {citations}")

        if q.grading == "strict":
            status = "PASS" if passed else "FAIL"
            print(
                f"-- {status} -- expected guidelines={q.expected_guidelines} "
                f"recs={q.expected_recommendation_numbers}"
            )
            all_strict_passed = all_strict_passed and bool(passed)
        else:
            print("-- loose, review by hand --")
            print(f"new answer:\n{answer}")
            if q.baseline_answer:
                print(f"\nbaseline answer:\n{q.baseline_answer}")
            if q.notes:
                print(f"\nnotes: {q.notes}")

    print(f"\n{'=' * 70}")
    strict_count = sum(1 for q in EVAL_QUESTIONS if q.grading == "strict")
    print(f"strict questions: {'all passed' if all_strict_passed else 'FAILURES'} ({strict_count} total)")
    print(
        f"loose questions: {sum(1 for q in EVAL_QUESTIONS if q.grading == 'loose')} -- see above for manual review"
    )
    return all_strict_passed
