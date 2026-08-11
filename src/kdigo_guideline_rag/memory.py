"""TrustCall extractor for user-context memory.

Remembers facts about the *user* -- their clinical role, expertise level,
recurring topics of interest -- so the assistant can calibrate explanation
depth across a conversation without being told again. Deliberately does not
store facts about any specific patient (see DECISIONS.md): this is a
guideline-education tool, not a per-patient clinical record.

Scoped per-user (not per-thread) in the ``("user_context", user_id)``
namespace -- these are self-descriptive facts about one person, so there's
no cross-contamination risk the way per-*patient* facts would have had.
"""

from uuid import uuid4

from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
from pydantic import BaseModel, Field
from trustcall import create_extractor

from kdigo_guideline_rag.nodes import get_llm
from kdigo_guideline_rag.state import GraphState

NAMESPACE_PREFIX = "user_context"


class UserContext(BaseModel):
    """A single fact about the user's clinical role or interests."""

    content: str = Field(
        description="A fact about the user's clinical role, expertise level, "
        "or recurring interests (e.g. 'is a nephrology fellow', 'frequently "
        "asks about AKI management'). Not a fact about any patient."
    )


def _extractor():
    return create_extractor(
        get_llm(),
        tools=[UserContext],
        tool_choice="UserContext",
        enable_inserts=True,
    )


def _user_id(config: RunnableConfig) -> str:
    return config.get("configurable", {}).get("user_id", "default")


def update_memory(state: GraphState, config: RunnableConfig, *, store: BaseStore) -> dict:
    """Extract any new user-context facts from the question and merge them
    into memory, then return the full current set formatted for the prompt.
    """
    namespace = (NAMESPACE_PREFIX, _user_id(config))

    existing_items = store.search(namespace)
    existing = [(item.key, "UserContext", item.value) for item in existing_items]

    result = _extractor().invoke(
        {
            "messages": [("user", state["question"])],
            "existing": existing or None,
        }
    )

    for response, meta in zip(result["responses"], result["response_metadata"], strict=True):
        key = meta.get("json_doc_id") or str(uuid4())
        store.put(namespace, key, response.model_dump())

    all_facts = [item.value["content"] for item in store.search(namespace)]
    return {"user_context": "\n".join(f"- {fact}" for fact in all_facts)}
