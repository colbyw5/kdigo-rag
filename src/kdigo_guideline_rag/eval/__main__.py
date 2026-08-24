"""Entry point for ``pixi run eval``: run the small eval set against the
real pipeline and report results.
"""

import sys

from dotenv import load_dotenv

from kdigo_guideline_rag.eval.runner import run_eval

if __name__ == "__main__":
    load_dotenv()
    passed = run_eval()
    sys.exit(0 if passed else 1)
