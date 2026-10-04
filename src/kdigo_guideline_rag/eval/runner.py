"""Tier-2 eval: runs questions through the real graph and reports pipeline behavior.

Per question: grader precision (graded-relevant / retrieved, final retrieval
round) and rewrite count. In-scope questions **pass** when every needle
survives into the graded documents -- retrieval found it *and* the grader
kept it. Out-of-scope questions have no pass/fail here; what matters is
**grader false positives** (chunks passed as relevant to a question the
corpus doesn't answer), which is what lets the generator extrapolate.

The grader is an LLM: identical input moves precision by about +-1 chunk per
question. Use tier 1 (``retrieval.py``) for index/retrieval decisions; use
this for grader/rewrite/pipeline changes, and read deltas against that noise.
"""

from dataclasses import dataclass

from kdigo_guideline_rag.eval.questions import EvalQuestion
from kdigo_guideline_rag.eval.retrieval import matches
from kdigo_guideline_rag.graph import graph


@dataclass
class RunResult:
    """Outcome of one eval question.

    Attributes:
        passed: All needles survived grading; ``None`` for out-of-scope questions.
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
    graded = result["graded_documents"]
    citations = [
        (d.metadata.get("source_guideline"), d.metadata.get("recommendation_number")) for d in graded
    ]
    retrieved, relevant, rewrites = retrieval_metrics(result)
    passed = None
    if q.needles:
        passed = all(any(matches(n, d.page_content, d.metadata) for d in graded) for n in q.needles)
    return RunResult(passed, result["answer"], citations, retrieved, relevant, rewrites)


def run_eval(questions: list[EvalQuestion]) -> bool:
    """Run ``questions`` through the graph, print a report, return True iff all in-scope ones passed."""
    results: dict[str, RunResult] = {}
    for q in questions:
        r = results[q.id] = _run_one(q)
        print(f"\n{'=' * 70}\n[{q.id}] ({q.category}, {q.split}) {q.question}")
        print(f"citations: {r.citations}")
        print(
            f"retrieval: {r.relevant}/{r.retrieved} relevant (precision {r.precision:.2f}), rewrites={r.rewrites}"
        )
        if r.passed is not None:
            print(f"-- {'PASS' if r.passed else 'FAIL'} -- needles: {q.needles}")
        else:
            print(f"-- out of scope ({q.scope}): grader passed {r.relevant} chunk(s) as relevant --")
        print(f"answer:\n{r.answer}")
        if q.baseline_answer:
            print(f"\nbaseline answer:\n{q.baseline_answer}")
        if q.notes:
            print(f"\nnotes: {q.notes}")

    in_scope = [q for q in questions if q.needles]
    oos = [q for q in questions if not q.needles]
    passed = [q.id for q in in_scope if results[q.id].passed]
    print(f"\n{'=' * 70}")
    print(f"in-scope: {len(passed)}/{len(in_scope)} passed (all needles survived grading)")
    print(f"  failed: {[q.id for q in in_scope if not results[q.id].passed]}")
    if oos:
        fp = {q.id: results[q.id].relevant for q in oos}
        print(
            f"out-of-scope grader false positives: {sum(fp.values())} chunks across {len(oos)} questions {fp}"
        )
    in_results = [results[q.id] for q in in_scope]
    total_retrieved = sum(r.retrieved for r in in_results)
    total_relevant = sum(r.relevant for r in in_results)
    pooled = total_relevant / total_retrieved if total_retrieved else 0.0
    print(f"in-scope grader precision (pooled): {total_relevant}/{total_retrieved} = {pooled:.2f}")
    print(f"questions needing a rewrite: {sum(1 for r in results.values() if r.rewrites)}/{len(results)}")
    return len(passed) == len(in_scope)
