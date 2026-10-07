# kdigo-guideline-rag

A RAG application that answers clinical questions about kidney disease management using multiple KDIGO clinical practice guidelines as its knowledge base. Built with LangGraph and deployed via LangGraph Platform.

## Quickstart

Requires [pixi](https://pixi.sh).

```bash
pixi install
cp .env.example .env      # fill in ANTHROPIC_API_KEY, VOYAGE_API_KEY, etc.
pixi run ingest            # chunk + embed guidelines into Chroma
pixi run test               # run test suite
pixi run explore            # interactive marimo notebook: ask questions, inspect retrieval/grading
pixi run eval-retrieval     # tier 1: deterministic retrieval hit@5/recall@5/MRR on the dev split (no LLM)
pixi run eval               # tier 2: full graph on the dev split -- grader precision, needle survival, OOS false positives
```

Source PDFs go in `data/` (gitignored, not committed):

- `KDIGO-2024-CKD-Guideline.pdf`
- `KDIGO-2026-AKI-AKD-Guideline-Public-Review-Draft-March-2026.pdf`
- `KDIGO-2021-Glomerular-Diseases-Guideline_English_LN-2024-Update.pdf`
- `KDIGO-2021-Blood-Pressure-in-CKD-Guideline.pdf`

All four are freely available from [kdigo.org](https://kdigo.org).

## Architecture

```
User question
    │
    ▼
┌──────────┐     ┌──────────┐     ┌────────────┐
│ retrieve │────▶│  grade   │────▶│  generate  │
└──────────┘     └──────────┘     └────────────┘
                      │ fail
                      ▼
                 ┌──────────┐
                 │ rewrite  │──── loop back to retrieve
                 └──────────┘
```

Knowledge base: KDIGO 2024 CKD Evaluation and Management, KDIGO 2026 AKI/AKD (public review draft), KDIGO 2021 Glomerular Diseases, KDIGO 2021 Blood Pressure in CKD.

## Deploy

```bash
docker compose up
```

## Example queries

- "What eGFR stage is 38 mL/min/1.73m²?"
- "Should a patient with G3b A2 be on an SGLT2 inhibitor?"
- "When should this patient be referred to nephrology?"
- "My patient with CKD just had an AKI episode — how should I manage the acute phase?"

## Design notes

Brief rationale for the non-default choices in this project; see git history / commit messages for more detail.

- **Chroma over Pinecone** — prototype-scale corpus (4 docs, single user); Chroma runs embedded with no external infra. Pinecone targets a scale (millions of vectors, multi-tenant) this project doesn't need.
- **Docling over raw text extraction (PyMuPDF) or Unstructured** — naive extraction interleaves figure/table content into paragraph text mid-sentence. Docling's layout-aware parsing isolates figures/tables cleanly and preserves heading hierarchy; compared head-to-head with Unstructured, which fragmented sentences across element boundaries and typed similar content inconsistently. Also chosen with an eye toward generalizing to other (possibly scanned) clinical guidelines later — Docling has a built-in OCR fallback.
- **Recommendation-anchored chunking, not fixed-token** — all four guidelines consistently label recommendations inline (`Recommendation X.Y:`, `Practice Point X.Y.Z:`, each graded e.g. `(1B)`). Chunking on these boundaries plus heading hierarchy keeps each recommendation + its rationale together and supports the required citation format (`[Guideline, Chapter X, Rec Y]`), which fixed-token chunking would break.
- **Figure text is indexed verbatim from Docling's extraction; a vision-model fallback is deferred, not skipped.** Clinical guidelines put real criteria in figures (e.g. KDIGO CKD Figure 48's referral criteria, which the matching practice point only references). For vector PDFs Docling already extracts that text as the picture's children, so each figure becomes one `Figure` chunk: caption + text, verbatim. A vision-LLM transcription would also handle rasterized/scanned figures and color-encoded content (the GFR × albuminuria heat map), but it puts model-written text into a clinical knowledge base where the source's own words are available — so it's reserved for figures with no extractable text, and built when a document needs it. Ingest logs a per-document figure coverage report (indexed / dropped / **no extractable text**) so a new guideline whose figures are rasterized shows up as a number, not a silent gap. Inclusion thresholds are per-source config, not KDIGO constants.
- **PDF text artifacts are repaired at ingest with a word-frequency dictionary, conservatively.** Typeset PDFs yield split ligatures ("bene fi ts", "of fi ce" — 3,198 occurrences here) and occasional runs with no spaces ("WesuggestthatadultswithhighBP…", which pushed a BP-target recommendation to rank 24 for its own question). A regex can't tell "of fi ce" → *office* from "the fi rst" → *the first*; word frequencies ([`wordsegment`](https://github.com/grantjenks/python-wordsegment)) can. Both repairs only act on strong evidence — a run is split only if the result contains a function word, so long drug and chemical names stay whole — because a wrong "fix" inside a clinical recommendation is worse than an artifact. Generic to PDF extraction and switchable per source.
- **Eval is tiered, and retrieval is scored without an LLM** — 57 questions whose gold evidence is a verbatim corpus anchor (guideline + recommendation number, table/figure caption, or phrase), machine-verified against the index so it survives re-chunking. Tier 1 (`pixi run eval-retrieval`) is deterministic hit@5/recall@5/MRR, used for every index/retrieval decision; tier 2 (`pixi run eval`) runs the full graph. The LLM grader alone moves by ±1 chunk/question on identical input, too noisy to decide chunking changes. A held-out split is reserved for confirming decisions made on dev.
- **AKI source is the 2026 public review draft**, not the 2012 final — chosen for current AKI+AKD scope; recommendations are unfinalized and may change before final publication.
- **Claude Sonnet 5 + Voyage AI over OpenAI** — Anthropic has no first-party embeddings API, so embeddings needed a separate provider regardless. Chose Voyage AI (Anthropic's own recommended embeddings partner) over OpenAI or local embeddings: at this corpus's scale (~2M tokens for a full ingestion pass) Voyage's 200M-tokens/month free allowance makes it effectively free, with no quality tradeoff. For the LLM (grade/generate/rewrite), chose Sonnet 5 over Opus 5 — ~2.5x cheaper per query and sufficient quality for a context-constrained retrieve-then-generate task, not open-ended reasoning.

## Known limitations

- **Numeric range/classification questions are unreliable** (e.g. "what eGFR stage is 38 mL/min/1.73m²?"). Semantic embeddings don't do numeric interval reasoning — "38" has no learned relationship to a table row labeled "30-44" — so retrieval can miss the correct staging table even when it's cleanly indexed. A deterministic lookup (extracting KDIGO's GFR/albuminuria category tables at ingest time) would fix this specific case, but was deliberately descoped: it's a document-specific special case that cuts against this project's generic, config-driven architecture (see Design notes), and duplicates functionality (an eGFR calculator) that's out of scope for a RAG system to reimplement. Flagging the boundary rather than building around it. Notably, the closest published prior art (Miao et al., below) doesn't address this class of problem either — it's an open gap in RAG-over-guidelines generally, not a shortfall specific to this implementation.
- **Figure text reads in layout order**, not prose ("Further evaluation and | specialist management | …") — retrievable and verbatim, but less fluent than a rewrite. **99 pictures with no extractable text** (rasters, charts, logos) aren't indexed pending the vision fallback, and the GFR × albuminuria heat map's risk levels (encoded only as cell color) remain unavailable.
- **Superseded content is indexed**: the Glomerular Diseases PDF includes both the outdated 2021 lupus nephritis chapter (marked "OUTDATED") and its 2024 replacement, so lupus questions can retrieve withdrawn recommendations.
- **Text repair is dictionary-bound**: compounds not in the word list stay merged when they contain no function word (e.g. "Eosinophilicgranulomatosis"), and segmentation can occasionally mis-split a fragment ("36ginview…" → "gin view"); both are rare (3 runs remain in 4,360 chunks).
- No hybrid search (BM25 + vector) yet — not enough evidence from a small eval set to justify the added complexity over pure vector retrieval.

## Adding a new guideline

Ingestion is config-driven: a new guideline should be a new `GuidelineSource` entry in `src/kdigo_guideline_rag/ingest/sources.py`, not new chunking code.

1. Add the entry: recommendation regex (defaults match KDIGO's `Recommendation X.Y:` / `Practice Point X.Y.Z:`), any watermark `noise_patterns`, and `char_replacements` if the publisher's fonts decode to wrong glyphs (`KI_SYMBOL_FONT_MAP` covers Kidney International's).
2. `pixi run ingest` and read the per-document log: duplicates dropped, narrative packing, and the **figure coverage report** — a high "no extractable text" count means rasterized figures (vision-fallback candidates).
3. Spot-check for glyph garbling (e.g. `≥` arriving as another symbol) — clinical thresholds depend on it.
4. Add eval questions with needles taken from the indexed text, run `pixi run eval-verify`, then `pixi run eval-retrieval` to confirm existing questions didn't regress.

## References

- Miao J, Thongprayoon C, Suppadungsuk S, Garcia Valencia OA, Cheungpasitporn W. [Integrating Retrieval-Augmented Generation with Large Language Models in Nephrology: Advancing Practical Applications](https://pmc.ncbi.nlm.nih.gov/articles/PMC10972059/). Explores RAG + LLMs for nephrology using KDIGO guidelines as the reference base — related prior work motivating this project's approach.
