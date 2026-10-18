"""Response cache for eval runs: identical requests are answered from disk.

Retrieval is deterministic, so a stage whose prompt and inputs haven't changed
sends exactly the request it sent last time. Caching it makes reruns pay only
for what changed -- e.g. a generate-prompt A/B re-runs generation while the
five grading calls per question replay.

The trade-off: a replayed call returns the *same sample*, so a cached rerun
reports old results rather than re-measuring them. About 1 in 4 answers vary
between identical runs, so runs that measure that variation, or confirm a
decision, should pass ``--no-cache``. Every run prints and records how many
calls were replayed.

One SQLite file holds two namespaces: LangChain chat-model calls (the graph's
Sonnet calls, via :class:`LLMCache`) and the judge's Anthropic SDK calls (via
:meth:`ResponseStore.get` / :meth:`ResponseStore.put`). Only the eval
entry point enables it -- the served app never caches.
"""

import hashlib
import json
import sqlite3
import threading
from collections.abc import Sequence
from pathlib import Path

from langchain_core.caches import BaseCache
from langchain_core.globals import set_llm_cache
from langchain_core.load import dumps, loads
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, Generation

# What a cached entry may deserialize into -- nothing else, even if the file is tampered with.
CACHED_CLASSES = [ChatGeneration, Generation, AIMessage]

CACHE_PATH = Path(__file__).resolve().parents[3] / ".eval_cache" / "responses.sqlite"


class ResponseStore:
    """Thread-safe SQLite key-value store with per-namespace hit/miss counters."""

    def __init__(self, path: Path = CACHE_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS responses (namespace TEXT, key TEXT, value TEXT, PRIMARY KEY (namespace, key))"
        )
        self._lock = threading.Lock()
        self.hits: dict[str, int] = {}
        self.misses: dict[str, int] = {}

    @staticmethod
    def key(*parts: str) -> str:
        """Stable key for a request made of string parts."""
        return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()

    def get(self, namespace: str, key: str) -> str | None:
        """Stored value, counting the lookup as a hit or a miss."""
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM responses WHERE namespace = ? AND key = ?", (namespace, key)
            ).fetchone()
            counter = self.hits if row else self.misses
            counter[namespace] = counter.get(namespace, 0) + 1
        return row[0] if row else None

    def put(self, namespace: str, key: str, value: str) -> None:
        """Store (or overwrite) a value."""
        with self._lock, self._conn:
            self._conn.execute("INSERT OR REPLACE INTO responses VALUES (?, ?, ?)", (namespace, key, value))

    def counts(self, namespace: str) -> tuple[int, int]:
        """``(hits, lookups)`` so far in this process."""
        hits = self.hits.get(namespace, 0)
        return hits, hits + self.misses.get(namespace, 0)

    def clear(self, namespace: str | None = None) -> None:
        """Delete stored responses (one namespace, or all)."""
        with self._lock, self._conn:
            if namespace is None:
                self._conn.execute("DELETE FROM responses")
            else:
                self._conn.execute("DELETE FROM responses WHERE namespace = ?", (namespace,))


LLM_NAMESPACE = "langchain"
JUDGE_NAMESPACE = "judge"


class LLMCache(BaseCache):
    """LangChain cache backed by a :class:`ResponseStore`.

    LangChain keys a lookup on the serialized messages (``prompt``) and the
    model's parameters, bound tools and structured-output schema included
    (``llm_string``) -- so any change to a prompt, model or schema is a miss.
    """

    def __init__(self, store: ResponseStore) -> None:
        self.store = store

    def lookup(self, prompt: str, llm_string: str) -> Sequence[Generation] | None:
        """Cached generations for this exact request, or None."""
        value = self.store.get(LLM_NAMESPACE, ResponseStore.key(prompt, llm_string))
        return (
            [loads(g, allowed_objects=CACHED_CLASSES) for g in json.loads(value)]
            if value is not None
            else None
        )

    def update(self, prompt: str, llm_string: str, return_val: Sequence[Generation]) -> None:
        """Store the generations for this request."""
        value = json.dumps([dumps(g) for g in return_val])
        self.store.put(LLM_NAMESPACE, ResponseStore.key(prompt, llm_string), value)

    def clear(self, **kwargs: object) -> None:
        """Delete all cached chat-model responses."""
        self.store.clear(LLM_NAMESPACE)


_store: ResponseStore | None = None


def enable(path: Path = CACHE_PATH) -> ResponseStore:
    """Turn the cache on for chat-model calls and the judge; returns the store."""
    global _store
    _store = ResponseStore(path)
    set_llm_cache(LLMCache(_store))
    return _store


def active_store() -> ResponseStore | None:
    """The enabled store, or None when caching is off (``--no-cache``)."""
    return _store


def llm_counts() -> tuple[int, int]:
    """``(replayed, total)`` chat-model calls so far; ``(0, 0)`` when caching is off."""
    return _store.counts(LLM_NAMESPACE) if _store else (0, 0)


def replay_note(replayed: int, total: int) -> str:
    """One-line report of how much of a run came from the cache."""
    if _store is None:
        return "cache: off (--no-cache)"
    if not total:
        return "cache: no calls"
    note = f"cache: {replayed}/{total} calls replayed ({replayed / total:.0%})"
    if replayed:
        note += " -- replayed calls repeat earlier samples; use --no-cache to re-measure"
    return note
