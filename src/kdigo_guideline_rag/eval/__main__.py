"""Entry point for the eval tiers.

Usage:
    python -m kdigo_guideline_rag.eval retrieval [--split dev|heldout|all] [--k 5]
    python -m kdigo_guideline_rag.eval pipeline  [--split dev|heldout|all]
    python -m kdigo_guideline_rag.eval verify      # every needle exists in the index
    python -m kdigo_guideline_rag.eval judge [--run PATH]   # tier 3 on a saved pipeline run (default: latest)
    python -m kdigo_guideline_rag.eval judge-check --run PATH-judged.json   # judge must catch corrupted answers
    python -m kdigo_guideline_rag.eval conversations [--split dev|heldout|all]   # multi-turn final-turn scoring

Defaults to the dev split: run heldout only to confirm a decision already
made on dev.

Model responses are cached (``.eval_cache/``), so a rerun pays only for calls
whose request changed. Pass ``--no-cache`` to re-measure run-to-run variation
or to confirm a decision with fresh samples.
"""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from kdigo_guideline_rag.eval.questions import EVAL_QUESTIONS


def main() -> int:
    """Parse arguments, run the requested tier, return a process exit code."""
    parser = argparse.ArgumentParser(prog="python -m kdigo_guideline_rag.eval")
    parser.add_argument(
        "tier", choices=["retrieval", "pipeline", "verify", "judge", "judge-check", "conversations"]
    )
    parser.add_argument("--run", type=Path, default=None, help="judge: saved run file (default: latest)")
    parser.add_argument("--split", choices=["dev", "heldout", "all"], default="dev")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="call the models fresh instead of replaying cached responses (to re-measure variation)",
    )
    args = parser.parse_args()
    load_dotenv()
    if not args.no_cache and args.tier in ("pipeline", "judge", "judge-check", "conversations"):
        from kdigo_guideline_rag.eval import cache

        cache.enable()

    questions = [q for q in EVAL_QUESTIONS if args.split == "all" or q.split == args.split]

    if args.tier == "verify":
        from kdigo_guideline_rag.eval.retrieval import verify_needles
        from kdigo_guideline_rag.ingest.embed import get_vectorstore

        collection = get_vectorstore(reset=False).get(include=["documents", "metadatas"])
        from kdigo_guideline_rag.eval.conversations import CONVERSATIONS

        cases = [*EVAL_QUESTIONS, *CONVERSATIONS]
        problems = verify_needles(zip(collection["documents"], collection["metadatas"], strict=True), cases)
        print("\n".join(problems) or f"all needles verified ({sum(len(c.needles) for c in cases)})")
        from kdigo_guideline_rag.nodes import reference_documents

        try:
            print(f"reference passages present ({len(reference_documents())})")
        except LookupError as e:
            problems.append(str(e))
            print(e)
        return 1 if problems else 0

    if args.tier == "judge-check":
        from kdigo_guideline_rag.eval.judge import perturbation_check

        if args.run is None:
            raise SystemExit("judge-check needs --run PATH to a *-judged.json file")
        print(perturbation_check(args.run))
        return 0

    if args.tier == "conversations":
        from kdigo_guideline_rag.eval.conversations import CONVERSATIONS
        from kdigo_guideline_rag.eval.multiturn import run_conversations

        run_conversations(
            [c for c in CONVERSATIONS if args.split == "all" or c.split == args.split], split=args.split
        )
        return 0

    if args.tier == "judge":
        from kdigo_guideline_rag.eval.judge import run_judge

        run_judge(args.run)
        return 0

    if args.tier == "retrieval":
        from kdigo_guideline_rag.eval.retrieval import run_retrieval_eval

        run_retrieval_eval(questions, k=args.k)
        return 0

    from kdigo_guideline_rag.eval.runner import run_eval

    return 0 if run_eval(questions, split=args.split) else 1


if __name__ == "__main__":
    sys.exit(main())
