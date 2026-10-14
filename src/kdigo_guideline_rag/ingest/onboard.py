"""Onboard a new guideline: decide once how to parse it, save that decision as a profile.

``onboard()`` runs when a new document is added -- never on later ingests,
which just load the saved profile (``profiles.py``):

1. **Select** (deterministic, no LLM): score every known profile x every
   recommendation extractor on the document with the parse-quality scorecard;
   take the best passing combination.
2. **Agent** (only if nothing passes): Claude Opus 5.5 inspects the document
   through tools -- Docling items, text-layer pages, pattern tests -- and
   iterates on a profile against the same scorecard until it passes or the
   budget runs out.

The agent only ever chooses *settings* (patterns, extractor, glyph fixes);
chunk text always comes verbatim from the extractors, so a model can't put
words into the index. The scorecard balances completeness against precision
(numbering gaps vs. truncated / non-verbatim recommendations) so a pattern
that matches everything can't pass.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from kdigo_guideline_rag.ingest.chunker import chunk_document
from kdigo_guideline_rag.ingest.parser import parse_pdf
from kdigo_guideline_rag.ingest.profiles import (
    PATTERN_FIELDS,
    PROFILES_DIR,
    load_profile_sources,
    profile_from_source,
    save_profile,
    sources_from_profile,
)
from kdigo_guideline_rag.ingest.scorecard import score
from kdigo_guideline_rag.ingest.sources import GuidelineSource
from kdigo_guideline_rag.ingest.text_layer import page_texts

logger = logging.getLogger(__name__)

AGENT_MODEL = "claude-opus-5-5"
AGENT_MAX_ITERATIONS = 30
EXTRACTORS = ("docling", "text_layer")

# Starting point for an agent working from scratch: no recommendation format assumed.
BLANK_PROFILE = {
    "recommendation_pattern": None,
    "recommendation_label": None,
    "grade_pattern": None,
    "chapter_pattern": None,
    "ignore_heading_pattern": None,
    "heading_fallback_pattern": None,
    "noise_patterns": [],
    "char_replacements": {},
    "text_substitutions": [],
    "recommendation_extractor": "docling",
}


@dataclass
class Evaluation:
    """A profile's combined scorecards across a guideline's PDFs."""

    profile: dict
    cards: list = field(default_factory=list)
    error: str | None = None

    @property
    def passed(self) -> bool:
        return self.error is None and bool(self.cards) and all(c.passed for c in self.cards)

    @property
    def problems(self) -> float:
        return float("inf") if self.error or not self.cards else sum(c.problems for c in self.cards)

    def report(self, max_examples: int = 4) -> dict:
        """Compact, JSON-friendly summary -- what the agent sees after each attempt."""
        if self.error:
            return {"error": self.error}
        per_pdf = []
        for c in self.cards:
            s = c.summary()
            per_pdf.append(
                {
                    "pdf": Path(c.source_pdf).name if hasattr(c, "source_pdf") else c.source,
                    "passed": s["passed"],
                    "problems": s["problems"],
                    "recommendations": s["recommendations"],
                    "text_layer_recommendations": s["text_layer_recommendations"],
                    "missing_vs_text_layer": s["missing_vs_text_layer"][:10],
                    "numbering_gaps": s["numbering_gaps"][:10],
                    "unconfirmed_gaps": s["unconfirmed_gaps"][:10],
                    "truncated": s["truncated"][:max_examples],
                    "not_verbatim": s["not_verbatim"][:max_examples],
                    "artifacts": {k: v for k, v in s["artifacts"].items() if v},
                    "glyph_suspects": s["glyph_suspects"],
                    "tiny_fragments": s["tiny_fragments"],
                    "examples": {k: v[:2] for k, v in s["examples"].items() if v},
                }
            )
        return {"passed": self.passed, "problems": round(self.problems, 2), "per_pdf": per_pdf}


def evaluate(profile: dict, name: str, pdf_paths: list[Path]) -> Evaluation:
    """Chunk every PDF with ``profile`` (from the parse cache) and score it."""
    try:
        sources = sources_from_profile(profile, name, pdf_paths)
    except (ValueError, re.error, KeyError) as e:
        return Evaluation(profile, error=f"invalid profile: {e}")
    cards = []
    for source in sources:
        try:
            card = score(chunk_document(parse_pdf(Path(source.pdf_path)), source), source)
        except Exception as e:  # noqa: BLE001 -- a bad candidate profile must not crash onboarding
            return Evaluation(profile, error=f"{Path(source.pdf_path).name}: {type(e).__name__}: {e}")
        card.source_pdf = str(source.pdf_path)
        cards.append(card)
    return Evaluation(profile, cards)


def candidate_profiles() -> dict[str, dict]:
    """Known profiles to try first: one per distinct existing source configuration."""
    candidates: dict[str, dict] = {}
    for source in load_profile_sources():
        profile = profile_from_source(source)
        key = json.dumps(
            {k: v for k, v in profile.items() if k != "recommendation_extractor"}, sort_keys=True
        )
        candidates.setdefault(key, profile)
    return {f"known_{i}": p for i, p in enumerate(candidates.values())}


def select(name: str, pdf_paths: list[Path], candidates: dict[str, dict]) -> tuple[str, Evaluation] | None:
    """Deterministic tournament: every candidate x extractor; best passing, else best overall."""
    results = []
    for label, profile in candidates.items():
        for extractor in EXTRACTORS:
            ev = evaluate({**profile, "recommendation_extractor": extractor}, name, pdf_paths)
            logger.info("candidate %s/%s: passed=%s problems=%.1f", label, extractor, ev.passed, ev.problems)
            results.append((f"{label}/{extractor}", ev))
    if not results:
        return None
    return min(results, key=lambda r: (not r[1].passed, r[1].problems))


AGENT_SYSTEM = """You configure how a clinical practice guideline PDF is parsed into \
retrieval chunks. You never write or edit guideline text -- you choose settings \
(regex patterns, an extractor, glyph fixes), and the text always comes verbatim \
from the extractors.

Goal: a profile whose scorecard passes for every PDF -- no recommendations missing \
versus the text layer, at most ~2% truncated or non-verbatim, no leftover artifacts, \
no glyph suspects you can explain as mis-decoded font characters.

Profile fields (JSON; regexes are Python `re` syntax; prefix "(?i)" for case-insensitive):
- recommendation_pattern: matches the START of each recommendation; MUST have a named \
group "number" and MAY have a named group "label" (e.g. "Recommendation", "Practice \
Point"). If there is no label word, set recommendation_label instead.
- recommendation_label: label for recommendations when the pattern has no "label" group.
- grade_pattern: matches a recommendation's evidence grade; group 1 = the grade.
- chapter_pattern: matches headings that start a new chapter/section-level unit.
- ignore_heading_pattern: headings that should NOT become the current section \
(e.g. a bare "Recommendations" box title). Matched against the whole heading.
- heading_fallback_pattern: numbered subsection headings that arrive as plain text; \
set to null if it would swallow recommendations (e.g. when recommendations themselves \
start with a bare number).
- noise_patterns: list of regexes for page furniture to remove (running headers/footers, \
download stamps, watermarks).
- char_replacements: {"wrong char": "right char"} for mis-decoded font glyphs; \
text_substitutions: [[regex, replacement], ...] for context-dependent ones.
- recommendation_extractor: "docling" (structured layout model) or "text_layer" \
(PDF text layer fills in recommendations Docling cut off or missed -- better for \
boxed recommendations in multi-column layouts).

Work method: inspect items and pages first, test patterns with test_pattern, then \
evaluate_profile; read the failures it reports and revise. Precision matters as much as \
completeness: a recommendation pattern that also matches ordinary sentences, figure or \
table numbers, or doses is wrong even if counts look good. When the scorecard passes \
(or you have made no progress in several attempts), call submit_profile with your best \
profile and a short rationale naming the evidence for each non-default setting."""


def agent_onboard(
    name: str, pdf_paths: list[Path], start_profile: dict, max_iterations: int = AGENT_MAX_ITERATIONS
) -> tuple[dict, str, Evaluation]:
    """Run the onboarding agent; returns (profile, rationale, evaluation of that profile)."""
    import anthropic
    from anthropic import beta_tool

    docs = [parse_pdf(Path(p)) for p in pdf_paths]
    state: dict = {"best": None, "submitted": None, "attempts": 0}

    @beta_tool
    def show_items(pdf_index: int = 0, start: int = 0, count: int = 40) -> str:
        """Show Docling's extracted items (label + text) in reading order.

        Args:
            pdf_index: Which PDF of this guideline (0-based).
            start: Index of the first item to show.
            count: Number of items to show (max 80).
        """
        items = [
            (str(i.label), i.text) for i, _ in docs[pdf_index].iterate_items() if getattr(i, "text", None)
        ]
        rows = [
            f"[{k}] {lab}: {txt[:220]!r}"
            for k, (lab, txt) in enumerate(items[start : start + min(count, 80)], start)
        ]
        return f"{len(items)} items total\n" + "\n".join(rows)

    @beta_tool
    def show_text_layer(pdf_index: int = 0, page: int = 1, max_chars: int = 3500) -> str:
        """Show one page of the PDF's raw text layer (soft hyphens removed).

        Args:
            pdf_index: Which PDF of this guideline (0-based).
            page: 1-based page number.
            max_chars: Truncate after this many characters (max 6000).
        """
        source = GuidelineSource(name=name, pdf_path=pdf_paths[pdf_index])
        pages = page_texts(source)
        if not 1 <= page <= len(pages):
            return f"page out of range (1..{len(pages)})"
        return pages[page - 1][: min(max_chars, 6000)]

    @beta_tool
    def test_pattern(regex: str, pdf_index: int = 0, max_samples: int = 12) -> str:
        """Count and sample matches of a regex in Docling items and in the text layer.

        Args:
            regex: Python regex to test.
            pdf_index: Which PDF of this guideline (0-based).
            max_samples: How many sample matches (with context) to show.
        """
        try:
            pattern = re.compile(regex)
        except re.error as e:
            return f"invalid regex: {e}"
        items = " \n".join(i.text for i, _ in docs[pdf_index].iterate_items() if getattr(i, "text", None))
        layer = "\n".join(page_texts(GuidelineSource(name=name, pdf_path=pdf_paths[pdf_index])))
        out = []
        for where, text in (("docling", items), ("text_layer", layer)):
            matches = list(pattern.finditer(text))
            out.append(f"{where}: {len(matches)} matches")
            for m in matches[:max_samples]:
                ctx = text[max(0, m.start() - 40) : m.end() + 60].replace("\n", "\\n")
                out.append(f"   ...{ctx}...")
        return "\n".join(out)

    @beta_tool
    def evaluate_profile(profile_json: str) -> str:
        """Chunk every PDF with this profile and return the scorecard (failures with examples).

        Args:
            profile_json: The full profile as a JSON object string.
        """
        try:
            profile = {**BLANK_PROFILE, **json.loads(profile_json)}
        except json.JSONDecodeError as e:
            return f"invalid JSON: {e}"
        ev = evaluate(profile, name, pdf_paths)
        state["attempts"] += 1
        if state["best"] is None or ev.problems < state["best"].problems:
            state["best"] = ev
        return json.dumps(ev.report(), ensure_ascii=False)

    @beta_tool
    def submit_profile(profile_json: str, rationale: str) -> str:
        """Submit the final profile. Call once, when done.

        Args:
            profile_json: The full profile as a JSON object string.
            rationale: Evidence for each non-default setting, in a few sentences.
        """
        try:
            profile = {**BLANK_PROFILE, **json.loads(profile_json)}
        except json.JSONDecodeError as e:
            return f"invalid JSON: {e} -- fix and resubmit"
        state["submitted"] = (profile, rationale)
        return "submitted -- stop now."

    client = anthropic.Anthropic(max_retries=6, timeout=600)
    task = (
        f"Guideline: {name}\nPDFs: {[Path(p).name for p in pdf_paths]}\n"
        f"Starting profile:\n{json.dumps(start_profile, indent=2, ensure_ascii=False)}\n\n"
        "Find a profile that passes the scorecard for all PDFs, then submit it."
    )
    runner = client.beta.messages.tool_runner(
        model=AGENT_MODEL,
        max_tokens=16000,
        max_iterations=max_iterations,
        system=AGENT_SYSTEM,
        tools=[show_items, show_text_layer, test_pattern, evaluate_profile, submit_profile],
        messages=[{"role": "user", "content": task}],
        output_config={"effort": "high"},
        cache_control={"type": "ephemeral"},  # each turn re-sends the growing transcript
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    usage = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0, "turns": 0}
    for message in runner:
        usage["turns"] += 1
        usage["input"] += message.usage.input_tokens
        usage["output"] += message.usage.output_tokens
        usage["cache_read"] += getattr(message.usage, "cache_read_input_tokens", 0) or 0
        usage["cache_write"] += getattr(message.usage, "cache_creation_input_tokens", 0) or 0
        if message.stop_reason == "refusal":
            logger.warning("onboarding agent refused: %s", getattr(message, "stop_details", None))
            break
        if state["submitted"]:
            break

    # Claude Opus 5.5 list prices per million tokens: $4 input, $20 output, $0.20
    # cache read, $5 cache write (5-minute TTL, 1.25x input).
    usage["approx_cost_usd"] = round(
        (usage["input"] * 4 + usage["output"] * 20 + usage["cache_read"] * 0.2 + usage["cache_write"] * 5)
        / 1e6,
        2,
    )
    usage["evaluations"] = state["attempts"]
    logger.info("onboarding agent usage: %s", usage)
    state["usage"] = usage
    if state["submitted"]:
        profile, rationale = state["submitted"]
        return profile, f"{rationale}\n[usage: {usage}]", evaluate(profile, name, pdf_paths)
    best = state["best"]
    if best is None:
        raise RuntimeError("onboarding agent made no evaluable attempt")
    return best.profile, f"(agent did not submit; best evaluated profile kept)\n[usage: {usage}]", best


def onboard(
    name: str,
    slug: str,
    pdf_paths: list[Path],
    from_scratch: bool = False,
    use_agent: bool = True,
    profiles_dir: Path = PROFILES_DIR,
) -> tuple[Path, Evaluation]:
    """Decide how to parse a new guideline and save the profile (run once per document).

    Args:
        name: ``source_guideline`` name for its chunks.
        slug: File name for the saved profile.
        pdf_paths: The guideline's PDF(s) -- one profile covers them all.
        from_scratch: Skip known profiles and start the agent from a blank
            profile (used to test the agent against hand-configured guidelines).
        use_agent: Allow the agent when no known profile passes.
        profiles_dir: Where to save the profile. The default directory is
            ingested as a live source; experiments should save elsewhere.
    """
    report: dict = {"method": None}
    chosen: Evaluation | None = None
    if not from_scratch:
        picked = select(name, pdf_paths, candidate_profiles())
        if picked and picked[1].passed:
            report |= {"method": "selector", "candidate": picked[0]}
            chosen = picked[1]
        elif picked:
            report["selector_best"] = {"candidate": picked[0], "problems": round(picked[1].problems, 2)}
    if chosen is None:
        if not use_agent:
            raise RuntimeError(f"no known profile passes for {name} and the agent is disabled: {report}")
        start = BLANK_PROFILE if from_scratch else picked[1].profile
        _, rationale, chosen = agent_onboard(name, pdf_paths, start)
        report |= {"method": "agent", "model": AGENT_MODEL, "rationale": rationale}
    report["scorecard"] = chosen.report()
    path = save_profile(slug, name, pdf_paths, chosen.profile, report, profiles_dir)
    return path, chosen


assert set(PATTERN_FIELDS) <= set(BLANK_PROFILE) | {"reference_heading_pattern"}


def main() -> None:
    """CLI: ``python -m kdigo_guideline_rag.ingest.onboard --name NAME --slug SLUG PDF [PDF ...]``."""
    import argparse

    from dotenv import load_dotenv

    parser = argparse.ArgumentParser(
        description="Decide once how to parse a new guideline; save its profile."
    )
    parser.add_argument("pdfs", nargs="+", type=Path)
    parser.add_argument("--name", required=True, help="source_guideline name for its chunks")
    parser.add_argument("--slug", required=True, help="profile file name")
    parser.add_argument("--from-scratch", action="store_true", help="skip known profiles; agent starts blank")
    parser.add_argument("--no-agent", action="store_true", help="selector only")
    parser.add_argument("--out", type=Path, default=PROFILES_DIR, help="profiles directory (default: live)")
    args = parser.parse_args()
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for noisy in ("docling", "docling_core", "httpx", "anthropic", "RapidOCR"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    path, ev = onboard(args.name, args.slug, args.pdfs, args.from_scratch, not args.no_agent, args.out)
    print(json.dumps(json.loads(path.read_text())["decided_by"], indent=2, ensure_ascii=False)[:4000])
    print(f"\nprofile saved: {path}  (passed={ev.passed}, problems={ev.problems:.1f})")


if __name__ == "__main__":
    main()
