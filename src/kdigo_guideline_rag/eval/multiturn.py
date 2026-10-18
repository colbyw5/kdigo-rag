"""Runs the multi-turn conversations through the real graph and scores the final turns.

Each conversation gets its own thread (so ``messages`` history accumulates
via the checkpointer) and its own memory user id. The run is saved in the
same row format as ``runner.save_run``, so tier 3 (``eval judge --run``) can
judge the final answers too.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

from kdigo_guideline_rag.eval import cache
from kdigo_guideline_rag.eval.conversations import Conversation
from kdigo_guideline_rag.eval.retrieval import score
from kdigo_guideline_rag.eval.runner import RUNS_DIR
from kdigo_guideline_rag.graph import build_local_graph
from kdigo_guideline_rag.nodes import _cite, format_context


def context_check(conv: Conversation, standalone: str) -> list[str]:
    """Problems with the condensed final question (empty list if none)."""
    lowered = standalone.lower()
    problems = [
        f"missing any of {group}"
        for group in conv.must_mention
        if not any(t.lower() in lowered for t in group)
    ]
    problems += [f"leaked {term!r}" for term in conv.must_not_mention if term.lower() in lowered]
    return problems


def run_conversations(convs: list[Conversation], split: str = "dev") -> Path:
    """Run each conversation turn by turn, print the final-turn scores, save the run."""
    graph = build_local_graph()
    rows, results = [], []
    for conv in convs:
        config = {"configurable": {"thread_id": f"conv-{conv.id}", "user_id": f"conv-{conv.id}"}}
        replayed_before, calls_before = cache.llm_counts()
        try:
            for turn in conv.turns:
                state = graph.invoke({"question": turn}, config=config)
        except Exception as e:  # noqa: BLE001 -- one failed conversation must not discard the run
            print(f"ERROR {conv.id}: {type(e).__name__}: {e}")
            continue

        standalone = state["standalone_question"]
        problems = context_check(conv, standalone)
        retrieval = score(conv, state["documents"]) if conv.needles else None
        declined = not state.get("generation_documents")
        results.append((conv, problems, retrieval, declined))

        retrieval_txt = f"retrieval recall={retrieval.recall:.2f}" if retrieval else f"declined={declined}"
        print(f"{'ok  ' if not problems else 'FLAG'} {conv.id:26} [{conv.kind}] {retrieval_txt}")
        print(f"      standalone: {standalone}")
        for p in problems:
            print(f"      ! {p}")

        docs = state.get("generation_documents", [])
        rows.append(
            {
                "id": conv.id,
                "question": standalone,
                "category": "out_of_scope" if conv.kind == "out_of_scope" else "multiturn",
                "scope": "near" if conv.kind == "out_of_scope" else None,
                "split": conv.split,
                "turns": conv.turns,
                "answer": state["answer"],
                "context": format_context(docs),
                "cited_passages": [(_cite(d), d.page_content) for d in docs],
                "error": None,
                "llm_calls": cache.llm_counts()[1] - calls_before,
                "llm_replayed": cache.llm_counts()[0] - replayed_before,
            }
        )

    scored = [r for _, _, r, _ in results if r]
    print(f"\ncontext handled correctly: {sum(not p for _, p, _, _ in results)}/{len(results)}")
    if scored:
        print(
            f"final-turn retrieval: hit {sum(r.hit for r in scored)}/{len(scored)}, "
            f"mean recall {sum(r.recall for r in scored) / len(scored):.2f}"
        )
    oos = [d for c, _, _, d in results if c.kind == "out_of_scope"]
    if oos:
        print(f"out-of-scope follow-ups declined by the fixed response: {sum(oos)}/{len(oos)}")

    print(cache.replay_note(*cache.llm_counts()))
    RUNS_DIR.mkdir(exist_ok=True)
    path = RUNS_DIR / f"{datetime.now(UTC):%Y%m%d-%H%M%S}-conversations-{split}.json"
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"run saved for tier-3 judging: {path}")
    return path
