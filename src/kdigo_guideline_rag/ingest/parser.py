"""Docling wrapper for the ingest pipeline, with an on-disk parse cache.

Parsing dominates ingestion time (~90% of a full run), while chunking and
embedding are what actually get iterated on. Parsed documents are cached as
Docling JSON keyed on the PDF's bytes and the Docling version, so a changed
PDF or a Docling upgrade re-parses automatically and anything else is a
cache hit that never loads Docling's layout/table models.
"""

import hashlib
import logging
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path

from docling.document_converter import DocumentConverter
from docling_core.types.doc.base import ImageRefMode
from docling_core.types.doc.document import DoclingDocument

from kdigo_guideline_rag.ingest.sources import DATA_DIR

logger = logging.getLogger(__name__)

CACHE_DIR = DATA_DIR / "parsed"


@lru_cache(maxsize=1)
def _converter() -> DocumentConverter:
    return DocumentConverter()


def cache_path(pdf_path: Path, cache_dir: Path = CACHE_DIR) -> Path:
    """Cache file for ``pdf_path``: keyed on its content hash and the Docling version."""
    digest = hashlib.sha256(pdf_path.read_bytes())
    digest.update(version("docling").encode())
    return cache_dir / f"{pdf_path.stem}-{digest.hexdigest()[:12]}.json"


def parse_pdf(pdf_path: Path, cache_dir: Path | None = CACHE_DIR) -> DoclingDocument:
    """Parse a PDF into a Docling document, reusing a cached parse when valid.

    Args:
        pdf_path: Path to the source PDF.
        cache_dir: Directory for cached parses; ``None`` always re-parses
            and writes nothing.

    Returns:
        The parsed Docling document, structured into typed items
        (section headers, text, tables, pictures) with page provenance.
    """
    pdf_path = Path(pdf_path)
    cached = cache_path(pdf_path, cache_dir) if cache_dir is not None else None
    if cached is not None and cached.exists():
        logger.info("Parse cache hit: %s", cached.name)
        return DoclingDocument.load_from_json(cached)

    doc = _converter().convert(str(pdf_path)).document
    if cached is not None:
        cached.parent.mkdir(parents=True, exist_ok=True)
        # Placeholders, not embedded base64: the chunker never reads image
        # pixels, and embedding them would bloat each cache file.
        doc.save_as_json(cached, image_mode=ImageRefMode.PLACEHOLDER)
        logger.info("Parsed and cached: %s", cached.name)
    return doc
