"""Small eval set for the retrieve/grade/generate pipeline.

Not a full automated grading harness -- ``questions.py`` records a fixed
set of questions with grounded expectations (captured from real pipeline
runs, not guessed), and ``runner.py`` re-runs them and reports whether the
expected citations still surface. Strict questions fail loudly on
regression; loose questions (synthesis, known-limitation cases) print the
new answer next to the recorded baseline for a human to eyeball, since
"correct" for those isn't a fixed citation match.
"""
