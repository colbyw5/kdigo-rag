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

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from kdigo_guideline_rag.eval.questions import EvalQuestion
from kdigo_guideline_rag.eval.retrieval import matches
from kdigo_guideline_rag.graph import graph
from kdigo_guideline_rag.nodes import _cite, format_context

RUNS_DIR = Path(__file__).resolve().parents[3] / "eval_runs"


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
        context: The graded passages exactly as the generate prompt saw them
            (citation headers included) -- what tier 3 judges against.
        cited_passages: ``(citation header, text)`` per graded passage.
    """

    passed: bool | None
    answer: str
    citations: list[tuple[str | None, str | None]]
    retrieved: int
    relevant: int
    rewrites: int
    context: str = ""
    cited_passages: list[tuple[str, str]] = field(default_factory=list)
    error: str | None = None

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
    return RunResult(
        passed,
        result["answer"],
        citations,
        retrieved,
        relevant,
        rewrites,
        context=format_context(result.get("generation_documents", [])),
        cited_passages=[(_cite(d), d.page_content) for d in result.get("generation_documents", [])],
    )


def save_run(questions: list[EvalQuestion], results: dict[str, RunResult], split: str) -> Path:
    """Persist a pipeline run (answers + the exact passages behind them) for tier-3 judging.

    Saved runs let the judge be re-run on identical answers -- to measure the
    judge's own noise -- without paying to regenerate them.
    """
    RUNS_DIR.mkdir(exist_ok=True)
    path = RUNS_DIR / f"{datetime.now(UTC):%Y%m%d-%H%M%S}-{split}.json"
    rows = [
        {
            "id": q.id,
            "question": q.question,
            "category": q.category,
            "scope": q.scope,
            "split": q.split,
            **asdict(results[q.id]),
        }
        for q in questions
    ]
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    return path


def run_eval(questions: list[EvalQuestion], split: str = "dev") -> bool:
    """Run ``questions`` through the graph, print a report, save the run for
    tier 3, return True iff all in-scope ones passed."""
    results: dict[str, RunResult] = {}
    for q in questions:
        try:
            r = results[q.id] = _run_one(q)
        except Exception as e:  # noqa: BLE001 -- one failed question must not discard the run
            results[q.id] = RunResult(None, "", [], 0, 0, 0, error=f"{type(e).__name__}: {e}")
            print(f"\n{'=' * 70}\n[{q.id}] ERROR -- {results[q.id].error}")
            continue
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

    errored = [q.id for q in questions if results[q.id].error]
    in_scope = [q for q in questions if q.needles and not results[q.id].error]
    oos = [q for q in questions if not q.needles and not results[q.id].error]
    passed = [q.id for q in in_scope if results[q.id].passed]
    print(f"\n{'=' * 70}")
    if errored:
        print(f"ERRORED (excluded from all metrics): {errored}")
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
    print(f"run saved for tier-3 judging: {save_run(questions, results, split)}")
    return len(passed) == len(in_scope)
