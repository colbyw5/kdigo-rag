"""Tier-3 eval: judge generated answers against the passages they were built from.

Two layers, deterministic first:

- **In code**: every bracketed citation must point at a guideline -- and a
  recommendation/practice point/figure/table number -- present in the passages
  the generator actually received; regexes flag answers that ask the user to
  supply passages or offer outside knowledge.
- **LLM judge** (Claude Opus 5.5 -- a different, stronger model than the
  generator, to reduce self-preference): lists unsupported claims and claims
  transferred to another population/indication *before* its booleans, so each
  verdict is tied to quoted evidence. It judges support by the passages, never
  medical truth by its own knowledge.

The judge reads a saved pipeline run (``runner.save_run``), so it can be
re-run on identical answers to measure its own noise.
"""

import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field

from kdigo_guideline_rag.eval import cache
from kdigo_guideline_rag.eval.runner import RUNS_DIR

JUDGE_MODEL = "claude-opus-5-5"

# Most specific first: "Blood Pressure in CKD" also contains "CKD".
# Most specific first: an ADA citation can mention "Chronic Kidney Disease"; a
# KDIGO BP one contains "CKD". Matched as whole words.
GUIDELINE_KEYS = (
    (r"\bada\b|standards of care", "ADA"),
    (r"blood pressure", "BP"),
    (r"glomerular", "GLOM"),
    (r"\baki\b", "AKI"),
    (r"\bckd\b", "CKD"),
)
LABEL_ALIASES = {
    "recommendation": "Recommendation",
    "rec": "Recommendation",
    "practice point": "Practice Point",
    "pp": "Practice Point",
}
CITATION_BLOCK = re.compile(r"\[([^\[\]]*(?:KDIGO|ADA|Standards of Care)[^\[\]]*)\]")
NUMBERED_REF = re.compile(
    r"\b(Recommendation|Rec|Practice Point|PP)\.?\s*(\d+(?:\.\d+)+[a-z]?)", re.IGNORECASE
)
FIGURE_REF = re.compile(r"\b(Figure|Fig|Table)\.?\s*(\d+)\b", re.IGNORECASE)

ASKS_FOR_CONTEXT = re.compile(
    r"\b(please|could you|can you|if you can)\s+(provide|share|supply|send)\b"
    r"|\bprovide (me with )?(the )?(relevant |additional |specific )?(guideline )?(passages|context|excerpts|text)\b",
    re.IGNORECASE,
)
OFFERS_OUTSIDE_KNOWLEDGE = re.compile(
    r"\b(general|prior|outside|my own) (medical |clinical )?knowledge\b"
    r"|\bcommonly (referenced|cited|used|recommended) (range|target|value|dose)"
    r"|\bshould I (note|share|provide|outline)\b",
    re.IGNORECASE,
)


def _guideline_key(text: str) -> str | None:
    lowered = text.lower()
    return next((key for pattern, key in GUIDELINE_KEYS if re.search(pattern, lowered)), None)


@dataclass
class CitationCheck:
    """Deterministic citation validity for one answer."""

    total: int
    invalid: list[str]


def check_citations(answer: str, cited_passages: list[tuple[str, str]]) -> CitationCheck:
    """Validate every ``[KDIGO ...]`` citation against the passages the generator saw.

    A citation is valid when its guideline is among the passages and every
    numbered item it names (Recommendation/Practice Point N, Figure/Table N)
    appears in a passage -- header or text -- from that guideline. Chapter
    wording isn't checked: it's free text and paraphrased legitimately.
    """
    by_guideline: dict[str, str] = {}
    for header, text in cited_passages:
        key = _guideline_key(header)
        if key:
            by_guideline[key] = by_guideline.get(key, "") + "\n" + header + "\n" + text

    total, invalid = 0, []
    for block in CITATION_BLOCK.findall(answer):
        key: str | None = None
        for citation in (c.strip() for c in block.split(";") if c.strip()):
            total += 1
            # "[KDIGO 2021 Glomerular Diseases, Chapter 1; Chapter 3]": a fragment
            # without a guideline name continues the previous citation's guideline.
            key = _guideline_key(citation) or key
            corpus = by_guideline.get(key or "", "")
            ok = bool(corpus)
            for label, number in NUMBERED_REF.findall(citation):
                canonical = LABEL_ALIASES[label.lower()]
                ok = ok and re.search(rf"\b{canonical}\s+{re.escape(number)}\b", corpus) is not None
            for label, number in FIGURE_REF.findall(citation):
                kind = "Table" if label.lower() == "table" else "Figure"
                ok = ok and re.search(rf"\b{kind}\s+{number}\b", corpus) is not None
            if not ok:
                invalid.append(citation)
    return CitationCheck(total=total, invalid=invalid)


class FlaggedClaim(BaseModel):
    """One claim the passages don't support, with its clinical weight."""

    quote: str = Field(
        description="The claim, quoted briefly from the answer, plus what the passages actually say."
    )
    material: bool = Field(
        description="True if it changes clinical meaning: a threshold, dose, interval, criterion, population or "
        "indication, recommendation strength (recommend/suggest/consider), evidence grade or its meaning, or a "
        "claim that guidelines agree or differ. False for illustrative examples, paraphrase, or framing that "
        "would not change what a clinician does."
    )


class AnswerVerdict(BaseModel):
    """LLM judge output. Flagged claims come first so the booleans follow from them.

    ``faithful`` / ``materially_faithful`` are computed in code from the claim
    lists (see :func:`judge_answer`), so they can't contradict them.

    (No field may be named ``reasoning``: on Claude Opus 5.5 / Sonnet 5.5 that
    trips the ``reasoning_extraction`` safety classifier and the request is
    refused -- found while building this judge.)
    """

    unsupported_claims: list[FlaggedClaim] = Field(
        description="Specific clinical claims in the answer that no passage supports. Empty if none."
    )
    transferred_claims: list[FlaggedClaim] = Field(
        description="Claims that apply passage content to a population, indication, or setting the passage "
        "does not cover (e.g. a drug level stated for one disease presented as the target for another). "
        "Empty if none."
    )
    addresses_question: bool = Field(description="The answer responds to the question that was asked.")
    declined: bool = Field(
        description="The answer states that the guidelines do not cover the question (in whole or in part)."
    )
    wrongly_declined: bool = Field(
        description="The answer says something is not covered although the passages clearly contain it."
    )
    outside_knowledge: bool = Field(
        description="The answer uses, or offers to provide, information from outside the passages "
        "(including offering 'general' or 'commonly referenced' values)."
    )
    asks_user_for_context: bool = Field(
        description="The answer asks the user to supply passages, guideline text, or other context."
    )
    note: str = Field(description="One sentence justifying the verdict, pointing at the decisive evidence.")


JUDGE_SYSTEM = """You evaluate answers from a retrieval-augmented assistant that answers clinicians' \
questions using only passages retrieved from clinical practice guidelines (KDIGO and ADA).

Judge only whether the answer is supported by the passages shown -- not whether it is medically true \
by your own knowledge. A claim the passages don't state is unsupported even if you believe it is correct. \
General framing that makes no specific factual claim (e.g. "individualize to the patient") is not an \
unsupported claim. Citation formatting is checked separately; do not judge it.

The expected-coverage line tells you whether the question is one the indexed guidelines cover. For a \
question they do not cover, a good answer says so plainly, does not supply or offer outside information, \
and does not stretch content about a different population or indication to fit."""

JUDGE_USER = """Question: {question}

Expected coverage: {coverage}

Passages the assistant received:
{context}

Assistant's answer:
{answer}"""


def _client():
    import anthropic

    # Judging runs ~40 long requests; one dropped connection shouldn't cost a run.
    return anthropic.Anthropic(max_retries=6, timeout=300)  # high-effort judging can take minutes


def judge_answer(row: dict, client=None) -> dict:
    """Judge one saved run row; returns deterministic checks + LLM verdict as a dict."""
    client = client or _client()
    coverage = (
        "NOT covered by the indexed guidelines -- the correct behavior is to say so."
        if row["category"] == "out_of_scope"
        else "Covered by the indexed guidelines."
    )
    request = {
        "model": JUDGE_MODEL,
        "max_tokens": 4000,
        "betas": ["server-side-fallback-2026-07-01"],
        "extra_body": {"fallbacks": "default"},
        "output_config": {"effort": "high"},
        "system": JUDGE_SYSTEM,
        "messages": [
            {
                "role": "user",
                "content": JUDGE_USER.format(
                    question=row["question"],
                    coverage=coverage,
                    context=row["context"] or "(no passages -- nothing survived relevance grading)",
                    answer=row["answer"],
                ),
            }
        ],
    }
    verdict, judge_error, cached = _judge_call(client, request)
    if verdict is not None:
        claims = verdict["unsupported_claims"] + verdict["transferred_claims"]
        verdict["faithful"] = not claims
        verdict["materially_faithful"] = not any(c["material"] for c in claims)

    citations = check_citations(row["answer"], [tuple(p) for p in row["cited_passages"]])
    return {
        "id": row["id"],
        "category": row["category"],
        "scope": row["scope"],
        "citations_total": citations.total,
        "citations_invalid": citations.invalid,
        "regex_asks_for_context": bool(ASKS_FOR_CONTEXT.search(row["answer"])),
        "regex_offers_outside_knowledge": bool(OFFERS_OUTSIDE_KNOWLEDGE.search(row["answer"])),
        "verdict": verdict,
        "judge_error": judge_error,
        "cached": cached,
    }


def _judge_call(client, request: dict) -> tuple[dict | None, str | None, bool]:
    """``(verdict, judge_error, cached)`` for one judge request, via the response cache when enabled.

    The key covers the whole request plus the verdict schema, so a changed
    rubric, model or answer is a miss. Only successful verdicts are stored.
    """
    store = cache.active_store()
    key = cache.ResponseStore.key(
        json.dumps(request, sort_keys=True), json.dumps(AnswerVerdict.model_json_schema(), sort_keys=True)
    )
    if store and (hit := store.get(cache.JUDGE_NAMESPACE, key)) is not None:
        return json.loads(hit), None, True
    response = client.beta.messages.parse(output_format=AnswerVerdict, **request)
    if response.stop_reason == "refusal" or response.parsed_output is None:
        return (
            None,
            f"stop_reason={response.stop_reason} details={getattr(response, 'stop_details', None)}",
            False,
        )
    verdict = response.parsed_output.model_dump()
    if store:
        store.put(cache.JUDGE_NAMESPACE, key, json.dumps(verdict))
    return verdict, None, False


def _judge_isolated(row: dict, client) -> dict:
    """Judge one row; an API failure becomes a recorded judge_error, not a crashed run."""
    import anthropic

    try:
        return judge_answer(row, client)
    except anthropic.APIError as e:
        return {
            "id": row["id"],
            "category": row["category"],
            "scope": row["scope"],
            "citations_total": 0,
            "citations_invalid": [],
            "regex_asks_for_context": False,
            "regex_offers_outside_knowledge": False,
            "verdict": None,
            "judge_error": f"{type(e).__name__}: {e}",
            "cached": False,
        }


def judge_run(rows: list[dict], max_workers: int = 6) -> list[dict]:
    """Judge every row of a saved run concurrently, preserving row order.

    Each row is isolated: a failure is recorded as ``judge_error`` (and
    counted in the summary) rather than aborting the remaining rows.
    """
    client = _client()
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(lambda row: _judge_isolated(row, client), rows))


def summarize(judged: list[dict]) -> dict[str, float | int]:
    """Aggregate tier-3 metrics: in-scope answer quality and out-of-scope abstention."""
    ok = [j for j in judged if j["verdict"]]
    ins = [j for j in ok if j["category"] != "out_of_scope"]
    oos = [j for j in ok if j["category"] == "out_of_scope"]

    def leaks(j: dict) -> bool:
        v = j["verdict"]
        return (
            v["asks_user_for_context"]
            or v["outside_knowledge"]
            or j["regex_asks_for_context"]
            or j["regex_offers_outside_knowledge"]
        )

    def correct_abstention(j: dict) -> bool:
        v = j["verdict"]
        return v["declined"] and v["faithful"] and not leaks(j)  # out of scope: any added claim is outside

    def rate(xs: list[dict], pred) -> float:
        return sum(map(pred, xs)) / len(xs) if xs else float("nan")

    return {
        "judged": len(ok),
        "judge_errors": len(judged) - len(ok),
        "in_scope_n": len(ins),
        "in_scope_materially_faithful": rate(ins, lambda j: j["verdict"]["materially_faithful"]),
        "in_scope_faithful_strict": rate(ins, lambda j: j["verdict"]["faithful"]),
        "in_scope_material_claims": sum(
            c["material"]
            for j in ins
            for c in j["verdict"]["unsupported_claims"] + j["verdict"]["transferred_claims"]
        ),
        "in_scope_addresses_question": rate(ins, lambda j: j["verdict"]["addresses_question"]),
        "in_scope_wrongly_declined": sum(j["verdict"]["wrongly_declined"] for j in ins),
        "citations_valid": rate(ok, lambda j: not j["citations_invalid"]),
        "leaky_answers": sum(map(leaks, ok)),
        "oos_n": len(oos),
        "oos_correct_abstention": rate(oos, correct_abstention),
    }


def latest_run() -> Path:
    """Most recent saved (unjudged) pipeline run."""
    runs = sorted(p for p in RUNS_DIR.glob("*.json") if not p.stem.endswith("-judged"))
    if not runs:
        raise SystemExit(f"No saved runs in {RUNS_DIR} -- run `pixi run eval` first.")
    return runs[-1]


def run_judge(run_path: Path | None = None) -> Path:
    """Judge a saved run, print per-question flags and the summary, save ``*-judged.json``."""
    run_path = run_path or latest_run()
    rows = json.loads(run_path.read_text())
    skipped = [r["id"] for r in rows if r.get("error")]
    if skipped:
        print(f"skipping {len(skipped)} question(s) that errored in the pipeline: {skipped}")
    judged = judge_run([r for r in rows if not r.get("error")])
    for j in judged:
        v = j["verdict"]
        if v is None:
            print(f"ERROR  {j['id']:10} {j['judge_error']}")
            continue
        flags = [
            name
            for name, on in (
                ("MATERIAL", not v["materially_faithful"]),
                ("minor-gloss", v["materially_faithful"] and not v["faithful"]),
                ("off-question", j["category"] != "out_of_scope" and not v["addresses_question"]),
                ("wrongly-declined", v["wrongly_declined"]),
                ("outside-knowledge", v["outside_knowledge"] or j["regex_offers_outside_knowledge"]),
                ("asks-for-context", v["asks_user_for_context"] or j["regex_asks_for_context"]),
                ("bad-citation", bool(j["citations_invalid"])),
                ("oos-not-declined", j["category"] == "out_of_scope" and not v["declined"]),
            )
            if on
        ]
        print(f"{'FLAG ' if flags else 'ok   '} {j['id']:10} {', '.join(flags)}")
        for claim in v["unsupported_claims"] + v["transferred_claims"]:
            print(f"        {'!' if claim['material'] else '-'} {claim['quote'][:140]}")
        for c in j["citations_invalid"]:
            print(f"        ~ invalid citation: {c[:120]}")
    summary = summarize(judged)
    print("\n" + cache.replay_note(sum(bool(j.get("cached")) for j in judged), len(judged)))
    print(
        "\n"
        + "\n".join(f"{k:30} {v:.2f}" if isinstance(v, float) else f"{k:30} {v}" for k, v in summary.items())
    )
    out = run_path.with_name(run_path.stem + "-judged.json")
    out.write_text(
        json.dumps({"run": run_path.name, "summary": summary, "judged": judged}, indent=2, ensure_ascii=False)
    )
    print(f"\nsaved: {out}")
    return out


FABRICATED_CLAIM = (
    "KDIGO also recommends repeating this assessment every 2 weeks in all such patients (1A) [{cite}]."
)
NUMBER = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(?![\w.])")


def _swap_number(answer: str, context: str) -> str | None:
    """Alter the first clinical number >= 10 that the answer shares with its passages, or None.

    Skips numbers inside ``[...]`` citations: changing "KDIGO 2026" to "KDIGO
    4052" alters a citation, not a clinical claim (citations are checked
    deterministically), and the first version of this check did exactly that.
    """
    citation_spans = [m.span() for m in re.finditer(r"\[[^\[\]]*\]", answer)]
    for match in NUMBER.finditer(answer):
        if any(start <= match.start() < end for start, end in citation_spans):
            continue
        value = match.group(1)
        if float(value) >= 10 and re.search(rf"(?<![\w.]){re.escape(value)}(?![\w.])", context):
            new = str(int(float(value) * 2)) if "." not in value else f"{float(value) + 1:.1f}"
            return answer[: match.start()] + new + answer[match.end() :]
    return None


def perturbation_check(judged_path: Path, sample: int = 8, seed: int = 42) -> dict[str, float]:
    """Validate the judge: corrupt answers it rated faithful; it must catch every corruption.

    Two corruptions per sampled answer -- a number shared with the passages
    altered (120 -> 240), and an appended fabricated recommendation carrying a
    real-looking citation. Returns detection rates (judged unfaithful).
    """
    import random

    data = json.loads(judged_path.read_text())
    rows = {r["id"]: r for r in json.loads((RUNS_DIR / data["run"]).read_text())}
    eligible = sorted(
        j["id"]
        for j in data["judged"]
        if j["verdict"]
        and j["verdict"]["faithful"]
        and j["category"] != "out_of_scope"
        and rows[j["id"]]["context"]
    )
    chosen = random.Random(seed).sample(eligible, min(sample, len(eligible)))

    cases: list[tuple[str, dict]] = []
    for qid in chosen:
        row = rows[qid]
        swapped = _swap_number(row["answer"], row["context"])
        if swapped:
            cases.append(("number", {**row, "answer": swapped}))
        cite = row["cited_passages"][0][0]
        cases.append(
            ("fabricated", {**row, "answer": row["answer"] + "\n\n" + FABRICATED_CLAIM.format(cite=cite)})
        )

    results = judge_run([case for _, case in cases])
    print(cache.replay_note(sum(bool(j.get("cached")) for j in results), len(results)))
    detected: dict[str, list[bool]] = {"number": [], "fabricated": []}
    for (kind, case), j in zip(cases, results, strict=True):
        caught = j["verdict"] is not None and not j["verdict"]["faithful"]
        detected[kind].append(caught)
        print(f"{'caught' if caught else 'MISSED'} {kind:10} {case['id']}")
    return {kind: sum(v) / len(v) for kind, v in detected.items() if v}
