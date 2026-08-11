# kdigo-guideline-rag

A RAG application that answers clinical questions about kidney disease management using multiple KDIGO clinical practice guidelines as its knowledge base. Built with LangGraph and deployed via LangGraph Platform.

## Quickstart

Requires [pixi](https://pixi.sh).

```bash
pixi install
cp .env.example .env      # fill in ANTHROPIC_API_KEY, VOYAGE_API_KEY, etc.
pixi run ingest            # chunk + embed guidelines into Chroma
pixi run test               # run test suite
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
- **AKI source is the 2026 public review draft**, not the 2012 final — chosen for current AKI+AKD scope; recommendations are unfinalized and may change before final publication.
- **Claude Sonnet 5 + Voyage AI over OpenAI** — Anthropic has no first-party embeddings API, so embeddings needed a separate provider regardless. Chose Voyage AI (Anthropic's own recommended embeddings partner) over OpenAI or local embeddings: at this corpus's scale (~2M tokens for a full ingestion pass) Voyage's 200M-tokens/month free allowance makes it effectively free, with no quality tradeoff. For the LLM (grade/generate/rewrite), chose Sonnet 5 over Opus 5 — ~2.5x cheaper per query and sufficient quality for a context-constrained retrieve-then-generate task, not open-ended reasoning.

## Known limitations

- **Numeric range/classification questions are unreliable** (e.g. "what eGFR stage is 38 mL/min/1.73m²?"). Semantic embeddings don't do numeric interval reasoning — "38" has no learned relationship to a table row labeled "30-44" — so retrieval can miss the correct staging table even when it's cleanly indexed. A deterministic lookup (extracting KDIGO's GFR/albuminuria category tables at ingest time) would fix this specific case, but was deliberately descoped: it's a document-specific special case that cuts against this project's generic, config-driven architecture (see Design notes), and duplicates functionality (an eGFR calculator) that's out of scope for a RAG system to reimplement. Flagging the boundary rather than building around it. Notably, the closest published prior art (Miao et al., below) doesn't address this class of problem either — it's an open gap in RAG-over-guidelines generally, not a shortfall specific to this implementation.
- Reference/bibliography text sometimes gets embedded as narrative chunks and can crowd out real content in retrieval results (observed on an IgA nephropathy query).
- No hybrid search (BM25 + vector) yet — not enough evidence from a small eval set to justify the added complexity over pure vector retrieval.

## References

- Miao J, Thongprayoon C, Suppadungsuk S, Garcia Valencia OA, Cheungpasitporn W. [Integrating Retrieval-Augmented Generation with Large Language Models in Nephrology: Advancing Practical Applications](https://pmc.ncbi.nlm.nih.gov/articles/PMC10972059/). Explores RAG + LLMs for nephrology using KDIGO guidelines as the reference base — related prior work motivating this project's approach.
