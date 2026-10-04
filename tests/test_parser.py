"""Unit tests for the Docling parse cache -- no Docling models loaded."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from docling_core.types.doc.document import DoclingDocument

from kdigo_guideline_rag.ingest import parser


class _CountingConverter:
    def __init__(self) -> None:
        self.calls = 0

    def convert(self, path: str) -> SimpleNamespace:
        self.calls += 1
        doc = DoclingDocument(name=Path(path).stem)
        doc.add_text(label="text", text=f"parsed {self.calls}")
        return SimpleNamespace(document=doc)


@pytest.fixture
def converter(monkeypatch: pytest.MonkeyPatch) -> _CountingConverter:
    fake = _CountingConverter()
    monkeypatch.setattr(parser, "_converter", lambda: fake)
    return fake


def test_second_parse_is_a_cache_hit(tmp_path: Path, converter: _CountingConverter) -> None:
    pdf = tmp_path / "guideline.pdf"
    pdf.write_bytes(b"%PDF fake")

    first = parser.parse_pdf(pdf, cache_dir=tmp_path / "cache")
    second = parser.parse_pdf(pdf, cache_dir=tmp_path / "cache")

    assert converter.calls == 1
    assert second.texts[0].text == first.texts[0].text == "parsed 1"


def test_changed_pdf_bytes_invalidate_cache(tmp_path: Path, converter: _CountingConverter) -> None:
    pdf = tmp_path / "guideline.pdf"
    pdf.write_bytes(b"%PDF v1")
    parser.parse_pdf(pdf, cache_dir=tmp_path / "cache")

    pdf.write_bytes(b"%PDF v2")
    doc = parser.parse_pdf(pdf, cache_dir=tmp_path / "cache")

    assert converter.calls == 2
    assert doc.texts[0].text == "parsed 2"


def test_cache_disabled_always_parses_and_writes_nothing(
    tmp_path: Path, converter: _CountingConverter
) -> None:
    pdf = tmp_path / "guideline.pdf"
    pdf.write_bytes(b"%PDF fake")

    parser.parse_pdf(pdf, cache_dir=None)
    parser.parse_pdf(pdf, cache_dir=None)

    assert converter.calls == 2
    assert list(tmp_path.iterdir()) == [pdf]
