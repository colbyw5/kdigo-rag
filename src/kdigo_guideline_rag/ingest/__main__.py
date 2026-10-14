"""Entry point for ``pixi run ingest``: rebuild the Chroma knowledge base.

Usage:
    python -m kdigo_guideline_rag.ingest               # full rebuild, all sources
    python -m kdigo_guideline_rag.ingest "<substring>"  # ingest one source, appending
                                                          # to the existing collection --
                                                          # for resuming after an
                                                          # interrupted full run.
"""

import logging
import sys

from dotenv import load_dotenv

from kdigo_guideline_rag.ingest.embed import get_vectorstore, ingest_all, ingest_source
from kdigo_guideline_rag.ingest.profiles import load_profile_sources

logging.basicConfig(level=logging.INFO, format="%(message)s")

if __name__ == "__main__":
    load_dotenv()

    sources = load_profile_sources()
    if len(sys.argv) > 1:
        query = sys.argv[1]
        matches = [s for s in sources if query.lower() in s.name.lower()]
        if len(matches) != 1:
            names = "\n".join(f"  - {s.name}" for s in sources)
            raise SystemExit(f"'{query}' matched {len(matches)} sources, expected 1. Options:\n{names}")
        ingest_source(matches[0], get_vectorstore(reset=False))
    else:
        ingest_all(sources)
