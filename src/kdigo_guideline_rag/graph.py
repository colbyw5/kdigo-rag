"""LangGraph StateGraph definition and compilation.

prepare -> update_memory -> retrieve -> grade --[relevant]--> generate -> END
                                ^                --[none, retries left]--> rewrite --+
                                |                                                    |
                                +----------------------------------------------------+
"""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore
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

# Served graph (langgraph.json -> `langgraph dev` / Docker deployment): compiled
# without a store or checkpointer, because the LangGraph server injects its
# own persistence (thread history + the store user-context memory lives in --
# Postgres in the Docker deployment). The server refuses to load a graph that
# brings its own store.
graph = builder.compile()


def build_local_graph(store: BaseStore | None = None) -> CompiledStateGraph:
    """Compile the graph for in-process use (eval runner, notebook, scripts).

    Outside the server nothing injects a store, and ``update_memory`` needs
    one -- so local callers get an in-memory store (per process; memory is
    lost on exit, which is fine for evaluation and exploration).

    Args:
        store: Store to use; defaults to a fresh ``InMemoryStore``.
    """
    return builder.compile(store=store or InMemoryStore())
