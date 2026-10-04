"""Runs EVAL_QUESTIONS against the real graph and reports results.

Strict questions pass/fail on whether the expected citations still surface
after grading -- a regression check for retrieval + grading quality. Loose
questions have no pass/fail: the new answer is printed next to the recorded
baseline so a human can judge whether the answer is still clinically sound
and appropriately hedged, which isn't something a citation match can prove.

Every question (strict or loose) also reports two retrieval-quality metrics,
read straight off the final graph state at no extra API cost:

- **precision**: graded-relevant / retrieved for the final retrieval round.
  Junk chunks (e.g. bibliography text) usually get graded out, so they rarely
  move pass/fail -- but they still waste top-k slots, which shows up here.
- **rewrites**: whether the first retrieval found nothing relevant and the
  query had to be rewritten. ``documents`` only holds the latest round, so a
  rewrite's failed first round is visible only through this flag.

With 8 questions run once each and an LLM grader, treat these as
directional, not statistically meaningful.
"""

from dataclasses import dataclass

from kdigo_guideline_rag.eval.questions import EVAL_QUESTIONS, EvalQuestion
from kdigo_guideline_rag.graph import graph


@dataclass
class RunResult:
    """Outcome of one eval question.

    Attributes:
        passed: Strict-question verdict, or ``None`` for loose questions.
        answer: The generated answer.
        citations: ``(source_guideline, recommendation_number)`` per graded chunk.
        retrieved: Chunks retrieved in the final retrieval round.
        relevant: Chunks from that round that survived grading.
        rewrites: Number of query rewrites the run needed.
    """

    passed: bool | None
    answer: str
    citations: list[tuple[str | None, str | None]]
    retrieved: int
    relevant: int
    rewrites: int

    @property
    def precision(self) -> float:
        """Graded-relevant share of the final retrieval round (0.0 if nothing retrieved)."""
        return self.relevant / self.retrieved if self.retrieved else 0.0


def retrieval_metrics(state: dict) -> tuple[int, int, int]:
    """Extract ``(retrieved, relevant, rewrites)`` from a final graph state.

    Args:
        state: The dict returned by ``graph.invoke``.

    Returns:
        Counts for the final retrieval round and the number of rewrites.
    """
    return (
        len(state.get("documents", [])),
        len(state.get("graded_documents", [])),
        state.get("rewrite_count", 0),
    )


def _run_one(q: EvalQuestion) -> RunResult:
    """Invoke the graph for one question and score it."""
    config = {"configurable": {"guideline_filter": "all", "user_id": f"eval-{q.id}"}}
    result = graph.invoke({"question": q.question}, config=config)
    citations = [
        (d.metadata.get("source_guideline"), d.metadata.get("recommendation_number"))
        for d in result["graded_documents"]
    ]
    retrieved, relevant, rewrites = retrieval_metrics(result)

    passed = None
    if q.grading == "strict":
        actual_guidelines = {g for g, _ in citations}
        actual_recs = {r for _, r in citations if r}
        guidelines_ok = set(q.expected_guidelines) <= actual_guidelines
        recs_ok = set(q.expected_recommendation_numbers) <= actual_recs
        passed = guidelines_ok and recs_ok
    return RunResult(passed, result["answer"], citations, retrieved, relevant, rewrites)


def run_eval() -> bool:
    """Run the full eval set, print a report, return True iff all strict cases passed."""
    all_strict_passed = True
    results: list[RunResult] = []
    for q in EVAL_QUESTIONS:
        r = _run_one(q)
        results.append(r)
        passed, answer = r.passed, r.answer
        print(f"\n{'=' * 70}\n[{q.id}] ({q.category}, {q.grading}) {q.question}")
        print(f"citations: {r.citations}")
        print(
            f"retrieval: {r.relevant}/{r.retrieved} relevant (precision {r.precision:.2f}), rewrites={r.rewrites}"
        )

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
    total_retrieved = sum(r.retrieved for r in results)
    total_relevant = sum(r.relevant for r in results)
    pooled = total_relevant / total_retrieved if total_retrieved else 0.0
    rewritten = sum(1 for r in results if r.rewrites)
    print(f"retrieval precision (pooled): {total_relevant}/{total_retrieved} = {pooled:.2f}")
    print(f"questions needing a rewrite: {rewritten}/{len(results)}")
    return all_strict_passed
