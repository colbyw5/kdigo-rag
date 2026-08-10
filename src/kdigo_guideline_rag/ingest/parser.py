"""Docling wrapper for the ingest pipeline."""

from pathlib import Path

from docling.document_converter import DocumentConverter
from docling_core.types.doc.document import DoclingDocument

_converter = DocumentConverter()


def parse_pdf(pdf_path: Path) -> DoclingDocument:
    """Parse a PDF into a Docling document.

    Args:
        pdf_path: Path to the source PDF.

    Returns:
        The parsed Docling document, structured into typed items
        (section headers, text, tables, pictures) with page provenance.
    """
    return _converter.convert(str(pdf_path)).document
