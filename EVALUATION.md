# Evaluation

How answer quality is measured, and where it stands. All results below are from 2026-10-17, on the current index (KDIGO + ADA) and prompt.

## Setup

**73 questions, plus 8 multi-turn conversations.**
- **By topic:** 40 recommendation, 10 synthesis (cross-guideline), 7 narrative, 5 table, 2 figure, and 9 out-of-scope.
- **Phrasing:** 12 are in lay wording.
- **Splits:** a seeded split puts 51 questions in dev and holds out 22. Decisions are made on dev; held-out only confirms them.

**Gold labels are anchors in the source text,** not chunk IDs: a recommendation number, a table or figure caption, or a verbatim phrase. Labels survive re-chunking. `eval-verify` checks every anchor exists in the index (85/85).

**Four tiers:**

| Tier | Command | Checks | LLM calls |
|---|---|---|---|
| 1. Retrieval | `pixi run eval-retrieval` | Is the gold evidence in the top 5? (hit@5, recall@5, MRR) | none |
| 2. Pipeline | `pixi run eval` | Does the gold evidence survive grading? Do out-of-scope questions retrieve nothing? | Sonnet 5 |
| 3. Judge | `pixi run eval-judge` | Is every claim supported by the passages the answer saw? | Opus 5.5 |
| Multi-turn | `pixi run eval-conversations` | Is a follow-up condensed into the right standalone question? | Sonnet 5 |

**The judge** is a different model from the generator. It flags each unsupported claim and marks it **material** if it changes meaning: a threshold, dose, population, recommendation strength, evidence grade, or a claim that guidelines agree. It judges against the passages only, not its own medical knowledge. Citations are checked in code against the index.

**The judge is tested** by planting errors in answers it rated faithful (`eval judge-check`). It must flag them.

## Results

**Retrieval** (tier 1, all 64 in-scope questions):

| Group | n | hit@5 | recall@5 | MRR |
|---|---|---|---|---|
| All | 64 | 0.89 | 0.85 | 0.63 |
| Single recommendation | 40 | 0.90 | 0.88 | 0.63 |
| Synthesis (cross-guideline) | 10 | 0.80 | 0.63 | 0.48 |
| Lay phrasing | 12 | 1.00 | 0.96 | 0.63 |

**Answers** (tiers 2–3):

| | Dev | Held-out |
|---|---|---|
| Gold evidence survives grading | 37/45 | 15/19 |
| Free of material errors | 0.80 | 0.89 |
| Free of any unsupported claim | 0.40 | 0.32 |
| Citations valid | 1.00 | 1.00 |
| Out-of-scope correctly declined | 6/6 | 3/3 |

"Any unsupported claim" includes harmless paraphrase and framing. The material rate is the one that matters.

**Multi-turn:** follow-up context handled correctly in 8/8 conversations. Final-turn retrieval hit 6/7, and 5/7 in-scope final answers were free of material errors.

**Judge check:** it caught 8/8 fabricated recommendations and 6/7 altered numbers. The miss changed a year in a guideline's name ("KDIGO 2021" → "KDIGO 4042"), not a clinical value. The check should skip years.

## Known gaps

- **Cross-guideline comparisons** retrieve poorly (ADA-vs-KDIGO recall@5 is 0.33). One top-5 search tends to return one guideline's trial discussion.
- **Drug names vs class names.** "finerenone" misses recommendations that say "nonsteroidal MRA".
- **Run-to-run variation.** About 1 in 4 answers differ between identical runs, so compare aggregates and per-question churn, not single answers.

## How it's used

Every index change is decided on tier 1, which is free and deterministic. Prompt changes are A/B tested on dev, both arms against the same index and judge. For example, the 2026-10 switch to a study-aid prompt:

| | Old prompt | New prompt |
|---|---|---|
| Free of material errors | 0.76 | 0.80 |
| Answers referring to "the patient" | 9 | 3 |

Six questions had material errors only on the old prompt and four only on the new one, which is within run-to-run variation. That reads as no regression, not an improvement. The prompt was kept for its framing.

**Cost:** a fresh dev run is roughly $2 to answer and $2 to judge (estimated from token counts).

**Caching.** Model responses are cached in `.eval_cache/`, keyed on the full request, so a rerun pays only for calls whose request changed. Change the generate prompt and the grading calls replay while generation and judging run fresh. A replayed call returns the earlier sample, so each run prints how many calls it replayed. Use `--no-cache` to re-measure variation or to confirm a decision.
