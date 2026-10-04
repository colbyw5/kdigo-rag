"""Tier-1 eval: deterministic retrieval scoring against needles.

No LLM involved -- the same index and questions always produce the same
numbers, so this is the metric for index/chunking/retrieval decisions. (The
LLM grader's precision, in ``runner.py``, moves by +-1 chunk per question on
identical input; see DECISIONS.md 2026-10-03.)

Per question, over the top-k retrieved chunks:

- **hit@k**: at least one needle found.
- **recall@k**: fraction of the question's needles found.
- **MRR**: 1 / rank of the first chunk matching any needle (0 if none).

Out-of-scope questions have no needles and aren't scored here; their
behavior (grader false positives, declining to answer) is a pipeline/answer
concern.
"""

import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from langchain_core.documents import Document

from kdigo_guideline_rag.eval.questions import EVAL_QUESTIONS, AnyOf, EvalQuestion, Requirement


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def matches(needle: Requirement, text: str, metadata: dict) -> bool:
    """Whether one chunk (text + metadata) satisfies ``needle`` (any option, for :class:`AnyOf`)."""
    if isinstance(needle, AnyOf):
        return any(matches(option, text, metadata) for option in needle.options)
    if metadata.get("source_guideline") != needle.guideline:
        return False
    if needle.label_type is not None and metadata.get("label_type") != needle.label_type:
        return False
    if needle.number is not None and metadata.get("recommendation_number") != needle.number:
        return False
    return needle.phrase is None or _norm(needle.phrase) in _norm(text)


@dataclass
class RetrievalScore:
    """Tier-1 metrics for one question."""

    question_id: str
    hit: bool
    recall: float
    reciprocal_rank: float
    found: list[Requirement]
    missing: list[Requirement]


def score(question: EvalQuestion, docs: list[Document]) -> RetrievalScore:
    """Score one question's retrieved documents (already truncated to k)."""
    found = [n for n in question.needles if any(matches(n, d.page_content, d.metadata) for d in docs)]
    first_rank = next(
        (
            rank
            for rank, d in enumerate(docs, start=1)
            if any(matches(n, d.page_content, d.metadata) for n in question.needles)
        ),
        None,
    )
    return RetrievalScore(
        question_id=question.id,
        hit=bool(found),
        recall=len(found) / len(question.needles),
        reciprocal_rank=1 / first_rank if first_rank else 0.0,
        found=found,
        missing=[n for n in question.needles if n not in found],
    )


def verify_needles(
    corpus: Iterable[tuple[str, dict]], questions: list[EvalQuestion] = EVAL_QUESTIONS
) -> list[str]:
    """Check every needle -- every option of an :class:`AnyOf` -- matches at least one chunk.

    Args:
        corpus: ``(text, metadata)`` for every indexed chunk.
        questions: Questions whose needles to check.

    Returns:
        One message per needle that matches nothing -- a gold label that
        can't be satisfied is a bug in the eval set, not a retrieval miss.
    """
    chunks = list(corpus)
    return [
        f"{q.id}: no chunk matches {option}"
        for q in questions
        for n in q.needles
        for option in (n.options if isinstance(n, AnyOf) else (n,))
        if not any(matches(option, text, meta) for text, meta in chunks)
    ]


def summarize(scores: list[RetrievalScore], questions: list[EvalQuestion]) -> dict[str, dict[str, float]]:
    """Mean hit@k / recall@k / MRR overall and per category, split, and phrasing."""
    by_id = {q.id: q for q in questions}
    groups: dict[str, list[RetrievalScore]] = defaultdict(list)
    for s in scores:
        q = by_id[s.question_id]
        for key in ("all", f"category={q.category}", f"split={q.split}", f"phrasing={q.phrasing}"):
            groups[key].append(s)
    return {
        key: {
            "n": len(group),
            "hit": sum(s.hit for s in group) / len(group),
            "recall": sum(s.recall for s in group) / len(group),
            "mrr": sum(s.reciprocal_rank for s in group) / len(group),
        }
        for key, group in groups.items()
    }


def run_retrieval_eval(questions: list[EvalQuestion], k: int = 5) -> list[RetrievalScore]:
    """Retrieve top-k for each in-scope question exactly as the ``retrieve`` node does
    (unfiltered similarity search), print per-question and grouped results.

    Returns:
        Per-question scores, for programmatic comparison across indexes.
    """
    from kdigo_guideline_rag.ingest.embed import get_vectorstore

    vectorstore = get_vectorstore(reset=False)
    scored = [q for q in questions if q.needles]
    scores = [score(q, vectorstore.similarity_search(q.question, k=k)) for q in scored]

    for s in scores:
        status = "HIT " if s.hit else "MISS"
        missing = "" if not s.missing else f"  missing: {[_describe(n) for n in s.missing]}"
        print(f"{status} recall={s.recall:.2f} rr={s.reciprocal_rank:.2f}  {s.question_id}{missing}")

    print(f"\n{'group':28} {'n':>3} {'hit@' + str(k):>7} {'recall@' + str(k):>9} {'MRR':>6}")
    for key, m in sorted(summarize(scores, scored).items(), key=lambda kv: (kv[0] != "all", kv[0])):
        print(f"{key:28} {m['n']:>3.0f} {m['hit']:>7.2f} {m['recall']:>9.2f} {m['mrr']:>6.2f}")
    return scores


def _describe(n: Requirement) -> str:
    if isinstance(n, AnyOf):
        return " OR ".join(_describe(o) for o in n.options)
    parts = [n.guideline.split(" ")[1] + " " + n.guideline.split(" ")[2]]
    parts += [p for p in (n.label_type, n.number) if p]
    if n.phrase:
        parts.append(repr(n.phrase[:40]))
    return " ".join(parts)
