"""Entry point for the eval tiers.

Usage:
    python -m kdigo_guideline_rag.eval retrieval [--split dev|heldout|all] [--k 5]
    python -m kdigo_guideline_rag.eval pipeline  [--split dev|heldout|all]
    python -m kdigo_guideline_rag.eval verify      # every needle exists in the index
    python -m kdigo_guideline_rag.eval judge [--run PATH]   # tier 3 on a saved pipeline run (default: latest)
    python -m kdigo_guideline_rag.eval judge-check --run PATH-judged.json   # judge must catch corrupted answers

Defaults to the dev split: run heldout only to confirm a decision already
made on dev.
"""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from kdigo_guideline_rag.eval.questions import EVAL_QUESTIONS


def main() -> int:
    """Parse arguments, run the requested tier, return a process exit code."""
    parser = argparse.ArgumentParser(prog="python -m kdigo_guideline_rag.eval")
    parser.add_argument("tier", choices=["retrieval", "pipeline", "verify", "judge", "judge-check"])
    parser.add_argument("--run", type=Path, default=None, help="judge: saved run file (default: latest)")
    parser.add_argument("--split", choices=["dev", "heldout", "all"], default="dev")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    load_dotenv()

    questions = [q for q in EVAL_QUESTIONS if args.split == "all" or q.split == args.split]

    if args.tier == "verify":
        from kdigo_guideline_rag.eval.retrieval import verify_needles
        from kdigo_guideline_rag.ingest.embed import get_vectorstore

        collection = get_vectorstore(reset=False).get(include=["documents", "metadatas"])
        problems = verify_needles(
            zip(collection["documents"], collection["metadatas"], strict=True), EVAL_QUESTIONS
        )
        print("\n".join(problems) or f"all needles verified ({sum(len(q.needles) for q in EVAL_QUESTIONS)})")
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
