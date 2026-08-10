"""Embed chunked guideline documents into a Chroma collection.

Rebuilds the collection from scratch on each run -- this is meant to be
re-run whenever a source PDF or the chunking logic changes, not used for
incremental updates. Deterministic chunk IDs (hash of the chunk's content
and position) make re-runs reproducible even though the collection is
recreated each time.
"""

import hashlib
import logging
import os
import time
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_voyageai import VoyageAIEmbeddings

from kdigo_guideline_rag.ingest.chunker import Chunk, chunk_document
from kdigo_guideline_rag.ingest.parser import parse_pdf
from kdigo_guideline_rag.ingest.sources import GuidelineSource

logger = logging.getLogger(__name__)

COLLECTION_NAME = "kdigo_guidelines"
EMBEDDING_MODEL = "voyage-4"
MAX_BATCH_ATTEMPTS = 4


def chunk_id(chunk: Chunk, index: int) -> str:
    """Deterministic ID for a chunk, stable across re-ingestion runs.

    Some narrative chunks are byte-identical (e.g. repeated running-footer
    text picked up as its own item on a page), so content alone can collide.
    ``index`` -- the chunk's position within its source's chunk list, which
    is itself deterministic -- guarantees uniqueness without weakening
    reproducibility.
    """
    key = "|".join(
        str(part)
        for part in (
            chunk.source_guideline,
            chunk.page,
            chunk.label_type,
            chunk.recommendation_number,
            chunk.text,
            index,
        )
    )
    return hashlib.sha256(key.encode()).hexdigest()[:32]


def chunk_to_document(chunk: Chunk) -> Document:
    """Convert a :class:`Chunk` into a LangChain ``Document``.

    Chroma rejects ``None`` metadata values, so fields that weren't set
    (e.g. ``section`` on a chapter-opening recommendation) are omitted
    rather than passed through as ``None``.
    """
    metadata = {
        "source_guideline": chunk.source_guideline,
        "chapter": chunk.chapter,
        "section": chunk.section,
        "label_type": chunk.label_type,
        "recommendation_number": chunk.recommendation_number,
        "grade": chunk.grade,
        "page": chunk.page,
    }
    metadata = {k: v for k, v in metadata.items() if v is not None}
    return Document(page_content=chunk.text, metadata=metadata)


def get_vectorstore(reset: bool = False) -> Chroma:
    """Open the Chroma collection backing the RAG pipeline.

    Args:
        reset: If ``True``, delete and recreate the collection first --
            for starting a full rebuild. If ``False``, open (and create,
            if absent) the collection as-is -- for resuming an interrupted
            ingestion run by adding one more source without wiping what's
            already there.
    """
    persist_dir = os.environ.get("CHROMA_PERSIST_DIR", "./chroma_db")
    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    embeddings = VoyageAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )
    if reset:
        vectorstore.delete_collection()
        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=persist_dir,
        )
    return vectorstore


def _add_batch_with_retry(vectorstore: Chroma, documents: list[Document], ids: list[str]) -> None:
    """Add one batch to Chroma, retrying transient embedding-API failures.

    Parsing a full guideline takes minutes; a single dropped connection on
    the embedding call shouldn't force re-parsing everything. Only network/
    server errors are retried -- a real data problem (e.g. bad metadata)
    would fail identically on every attempt, so retrying it is pointless.
    """
    for attempt in range(1, MAX_BATCH_ATTEMPTS + 1):
        try:
            vectorstore.add_documents(documents=documents, ids=ids)
            return
        except Exception:
            if attempt == MAX_BATCH_ATTEMPTS:
                raise
            delay = 2**attempt
            logger.warning(
                "Batch embed failed (attempt %d/%d), retrying in %ds",
                attempt,
                MAX_BATCH_ATTEMPTS,
                delay,
                exc_info=True,
            )
            time.sleep(delay)


def ingest_source(source: GuidelineSource, vectorstore: Chroma, batch_size: int = 128) -> int:
    """Parse, chunk, and embed one source guideline into an open collection.

    Returns:
        Number of chunks ingested.
    """
    logger.info("Parsing %s", source.name)
    doc = parse_pdf(source.pdf_path)
    chunks = chunk_document(doc, source)
    documents = [chunk_to_document(c) for c in chunks]
    ids = [chunk_id(c, i) for i, c in enumerate(chunks)]

    for i in range(0, len(documents), batch_size):
        _add_batch_with_retry(
            vectorstore,
            documents[i : i + batch_size],
            ids[i : i + batch_size],
        )
    logger.info("Ingested %s: %d chunks", source.name, len(documents))
    return len(documents)


def ingest_all(sources: list[GuidelineSource], batch_size: int = 128) -> None:
    """Parse, chunk, and embed every source guideline into a fresh Chroma
    collection.

    Args:
        sources: Guideline documents to ingest.
        batch_size: Documents per embedding API call.
    """
    vectorstore = get_vectorstore(reset=True)
    for source in sources:
        ingest_source(source, vectorstore, batch_size)
