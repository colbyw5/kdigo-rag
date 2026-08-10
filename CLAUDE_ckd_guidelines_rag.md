# CLAUDE.md — ckd-guidelines-rag

Standing instructions for this repo. Read this first.

---

## What this project is

A RAG application that answers clinical questions about **kidney disease management** using multiple **KDIGO clinical practice guidelines** as its knowledge base. Built with **LangGraph** for orchestration and deployed via **LangGraph Platform**.

**Knowledge base (all freely available from KDIGO):**

- **KDIGO 2024 CKD Evaluation and Management** — eGFR staging, albuminuria, treatment algorithms (SGLT2i, RAS blockade), monitoring, referral criteria
- **KDIGO 2012 AKI (Acute Kidney Injury)** — AKI staging, prevention, management, renal replacement therapy initiation
- **KDIGO 2021 Glomerular Diseases** — covers APOL1-mediated kidney disease, IgA nephropathy, FSGS, lupus nephritis
- **KDIGO 2021 Blood Pressure in CKD** — hypertension targets and management specific to CKD patients

Multi-document RAG forces the system to retrieve across sources, disambiguate, and cite correctly — a more realistic and technically interesting challenge than single-document RAG.

**Target user:** a clinician or field medical team member asking questions like "my patient has eGFR 38 and A2 albuminuria — what does KDIGO recommend?" or "how does AKI management differ from CKD in this context?"

**Why it exists:** portfolio project demonstrating LangGraph orchestration, RAG pipeline design, structured memory, and LangGraph Platform deployment. Must be a working prototype, not a toy demo.

---

## Architecture

```
User question
    │
    ▼
┌──────────┐     ┌──────────┐     ┌────────────┐
│ retrieve  │────▶│  grade   │────▶│  generate  │
│ (vector   │     │ (relevance│     │ (synthesize│
│  search)  │     │  check)   │     │  + cite)   │
└──────────┘     └──────────┘     └────────────┘
                      │ fail
                      ▼
                 ┌──────────┐
                 │ rewrite  │──── loop back to retrieve
                 │ (query   │
                 │ transform)│
                 └──────────┘
```

**Graph nodes:**

- **`retrieve`** — vector search over embedded KDIGO guideline chunks across all loaded guidelines. Returns top-k relevant sections with source document metadata.
- **`grade`** — LLM evaluates whether retrieved chunks actually answer the question. Binary yes/no per chunk. Filters out irrelevant results.
- **`generate`** — synthesize answer from relevant chunks. Must include source citations with **guideline name** + chapter + recommendation number (e.g. `[CKD Guideline, Chapter 3, Rec 3.1.2]`). If no relevant chunks survive grading, route to `rewrite`.
- **`rewrite`** — LLM rewrites the user query to improve retrieval. Routes back to `retrieve`. Max 1 rewrite to avoid loops.

**Conditional edges:**

- `grade` → `generate` (if ≥1 relevant chunk)
- `grade` → `rewrite` (if 0 relevant chunks)
- `rewrite` → `retrieve` (retry with rewritten query)

**Memory:** TrustCall collection pattern (from LangGraph module 5) to remember patient context across a conversation session. E.g. "patient has eGFR 38" persists so follow-up questions don't repeat it. Store patient context in `("patient_context", user_id)` namespace.

---

## Repo structure

```
ckd-guidelines-rag/
├── CLAUDE.md               # this file
├── README.md               # project overview, setup, usage
├── langgraph.json          # LangGraph Platform config
├── requirements.txt        # Python dependencies
├── docker-compose.yml      # Redis + Postgres + LangGraph Server
├── .env.example            # template for env vars
├── data/                   # source guidelines (not committed — README explains how to obtain)
│   ├── kdigo_ckd_2024.pdf
│   ├── kdigo_aki_2012.pdf
│   ├── kdigo_glomerular_2021.pdf
│   └── kdigo_bp_ckd_2021.pdf
├── src/
│   ├── __init__.py
│   ├── ingest.py           # PDF loading, section-aware chunking, embedding into Chroma
│   ├── graph.py            # LangGraph StateGraph definition, compile
│   ├── nodes.py            # retrieve, grade, generate, rewrite node functions
│   ├── state.py            # graph state schema (TypedDict)
│   ├── memory.py           # TrustCall extractor for patient context memory
│   ├── prompts.py          # all prompt templates (system messages, grading, generation)
│   └── configuration.py    # configurable fields for assistants
├── tests/
│   ├── test_retrieval.py   # retrieval quality checks
│   └── test_graph.py       # end-to-end graph invocation tests
└── notebooks/
    └── exploration.ipynb   # development scratchpad
```

---

## Tech stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Orchestration | LangGraph | Core skill to demonstrate; graph-based RAG with conditional edges |
| LLM | OpenAI gpt-4o | Reliable, fast; swap via env var |
| Embeddings | OpenAI text-embedding-3-small | Good quality/cost tradeoff |
| Vector store | Chroma (local) | Zero-infrastructure for prototype; persistent via Docker volume |
| Memory | TrustCall + LangGraph Store | Demonstrates module 5 patterns (collection schema, JSON patches) |
| Deployment | LangGraph Platform | Demonstrates module 6 patterns (docker-compose, Redis, Postgres) |
| PDF parsing | PyMuPDF (fitz) or unstructured | Section-aware chunking matters — see chunking notes below |

---

## Implementation guidance

### PDF chunking (critical)

Clinical guidelines have hierarchical structure: chapters → sections → recommendations → rationale. Naive fixed-token chunking destroys this structure. Instead:

1. Parse each PDF preserving section headers (chapter numbers, recommendation boxes).
2. Chunk at the **recommendation level** where possible — each recommendation + its rationale/evidence summary = one chunk.
3. Attach metadata to each chunk: `source_guideline`, `chapter`, `section`, `recommendation_number`, `page`. The `source_guideline` field is critical for multi-document citation accuracy.
4. For narrative sections (background, epidemiology), use ~500-token chunks with overlap.
5. Ingest all four guidelines into the **same Chroma collection** — retrieval searches across all sources, and the `source_guideline` metadata enables filtering and proper citation.

### Grading node

Use a simple structured output schema:

```python
class GradeResult(BaseModel):
    relevant: bool = Field(description="Whether the chunk is relevant to the question")
    reasoning: str = Field(description="Brief explanation of relevance decision")
```

Grade each retrieved chunk independently. Filter to only relevant chunks before generation.

### Generation node

The generation prompt must:

1. Instruct the model to answer ONLY from the provided context (no prior knowledge).
2. Require citations in the format `[Guideline Name, Chapter X, Recommendation Y]` — the guideline name is essential since chunks come from multiple documents.
3. When chunks from multiple guidelines are relevant, synthesize across them and note where guidelines align or differ.
4. If the context doesn't contain enough information, say so explicitly — do not hallucinate.
5. Use clinical language appropriate for a healthcare professional audience.

### Patient context memory

Use TrustCall with a simple schema:

```python
class PatientContext(BaseModel):
    content: str = Field(description="A clinical fact about the patient being discussed")
```

Store as a collection in `("patient_context", user_id)` namespace. Search and inject into the system prompt so the model knows accumulated patient details without the user repeating them.

### Configuration (assistants)

Support configurable fields via `configuration.py`:

- **`guideline_filter`** — optional filter to restrict retrieval to a specific guideline (e.g. "ckd_only", "aki_only", or "all"). Defaults to "all". Implemented as a Chroma metadata filter on `source_guideline`.
- **`clinical_focus`** — so the same graph architecture could be pointed at different disease areas in the future (e.g. "kidney", "diabetes", "heart_failure").

---

## Coding standards

- **Python 3.11+**, PEP 8
- **Sphinx-style docstrings** on all public functions
- **Type hints** everywhere
- Keep prompts in `prompts.py`, not inline in node functions
- Keep state schema in `state.py`, separate from graph definition
- No secrets in code — all API keys via `.env`
- `README.md` must include: what it is, how to set up, how to run locally, how to deploy, example queries, architecture diagram (Mermaid or ASCII)

---

## What success looks like

The prototype should handle questions like:

- "What eGFR stage is 38 mL/min/1.73m²?" → G3b, with CKD guideline citation
- "Should a patient with G3b A2 be on an SGLT2 inhibitor?" → yes/no with guideline rationale
- "When should this patient be referred to nephrology?" → referral criteria from CKD guideline
- "What monitoring interval does KDIGO recommend for G3b A2?" → specific interval with citation
- "My patient with CKD just had an AKI episode — how should I manage the acute phase?" → pulls from AKI guideline, cites separately from CKD recommendations
- "What BP target for a CKD patient with albuminuria?" → pulls from BP in CKD guideline
- "What's the recommended approach for APOL1-mediated kidney disease?" → pulls from Glomerular Diseases guideline

Each answer must cite the specific **guideline name** + chapter/recommendation. When multiple guidelines are relevant, the answer should synthesize across them. If the guidelines don't cover the question, the model must say so.

---

## Stretch goals (post-weekend)

- **Hybrid retrieval:** BM25 + semantic search with reciprocal rank fusion
- **KDIGO heat map as structured lookup:** encode the GFR × albuminuria risk matrix as a deterministic lookup table alongside vector search
- **Streamlit front-end:** simple chat UI
- **Additional disease areas:** add ADA diabetes or ACC/AHA heart failure guidelines to expand beyond kidney disease
- **Evaluation:** build a small test set of CKD questions with expected answers; run as CI check
