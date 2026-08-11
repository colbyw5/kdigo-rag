"""Retrieval quality checks against the live Chroma collection.

Integration tests -- require VOYAGE_API_KEY and a populated collection
(run `pixi run ingest` first). Skipped automatically otherwise.
"""

import os

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not os.environ.get("VOYAGE_API_KEY"), reason="requires VOYAGE_API_KEY"),
]

EXPECTED_SOURCES = {
    "KDIGO 2024 CKD Evaluation and Management",
    "KDIGO 2026 AKI/AKD (Public Review Draft)",
    "KDIGO 2021 Glomerular Diseases",
    "KDIGO 2021 Blood Pressure in CKD",
}


@pytest.fixture(scope="module")
def vectorstore():
    from kdigo_guideline_rag.ingest.embed import get_vectorstore

    return get_vectorstore(reset=False)


def test_collection_is_populated(vectorstore):
    count = vectorstore._collection.count()
    if count == 0:
        pytest.skip("collection is empty -- run `pixi run ingest` first")
    assert count > 1000  # full corpus is ~9k chunks; a sanity floor, not an exact match


def test_all_four_guidelines_are_present(vectorstore):
    if vectorstore._collection.count() == 0:
        pytest.skip("collection is empty -- run `pixi run ingest` first")
    # Query each source directly rather than sampling rows -- a small sample
    # can land entirely within one source depending on insertion order
    # (Chroma has no "distinct metadata value" query, and count() takes no
    # filter).
    for source in EXPECTED_SOURCES:
        found = vectorstore._collection.get(where={"source_guideline": source}, limit=1)
        assert found["ids"], f"no chunks found for {source!r}"


def test_query_returns_documents_with_citation_metadata(vectorstore):
    results = vectorstore.similarity_search("SGLT2 inhibitor for CKD with albuminuria", k=3)
    if not results:
        pytest.skip("collection is empty -- run `pixi run ingest` first")
    for doc in results:
        assert doc.metadata.get("source_guideline") in EXPECTED_SOURCES
        assert doc.page_content


def test_guideline_filter_restricts_to_one_source(vectorstore):
    target = "KDIGO 2021 Blood Pressure in CKD"
    results = vectorstore.similarity_search(
        "kidney disease management",
        k=5,
        filter={"source_guideline": target},
    )
    if not results:
        pytest.skip("collection is empty -- run `pixi run ingest` first")
    assert all(doc.metadata["source_guideline"] == target for doc in results)
