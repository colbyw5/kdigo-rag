"""LangGraph StateGraph definition and compilation.

prepare -> update_memory -> retrieve -> grade --[relevant]--> generate -> END
                                ^                --[none, retries left]--> rewrite --+
                                |                                                    |
                                +----------------------------------------------------+
"""

from langgraph.graph import END, START, StateGraph
from langgraph.store.memory import InMemoryStore

from kdigo_guideline_rag.memory import update_memory
from kdigo_guideline_rag.nodes import generate, grade, prepare, retrieve, rewrite, route_after_grade
from kdigo_guideline_rag.state import GraphState

builder = StateGraph(GraphState)
builder.add_node("prepare", prepare)
builder.add_node("update_memory", update_memory)
builder.add_node("retrieve", retrieve)
builder.add_node("grade", grade)
builder.add_node("generate", generate)
builder.add_node("rewrite", rewrite)

builder.add_edge(START, "prepare")
builder.add_edge("prepare", "update_memory")
builder.add_edge("update_memory", "retrieve")
builder.add_edge("retrieve", "grade")
builder.add_conditional_edges("grade", route_after_grade, {"generate": "generate", "rewrite": "rewrite"})
builder.add_edge("rewrite", "retrieve")
builder.add_edge("generate", END)

# In-memory store for prototype scale -- matches the "zero infra" approach
# used for Chroma. Swap for a persistent store (e.g. Postgres-backed) when
# deploying, so user-context memory survives process restarts.
graph = builder.compile(store=InMemoryStore())
