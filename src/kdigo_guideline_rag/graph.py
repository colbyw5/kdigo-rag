"""LangGraph StateGraph definition and compilation.

prepare -> update_memory -> condense -> retrieve -> grade --[relevant]--------------> generate -> END
                                           ^                --[none, retries left]--> rewrite --+
                                           |                                                    |
                                           +----------------------------------------------------+

``condense`` makes a follow-up self-contained using the thread's ``messages``
history; first turns pass through it unchanged.
"""

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

from kdigo_guideline_rag.memory import update_memory
from kdigo_guideline_rag.nodes import condense, generate, grade, prepare, retrieve, rewrite, route_after_grade
from kdigo_guideline_rag.state import GraphState

builder = StateGraph(GraphState)
builder.add_node("prepare", prepare)
builder.add_node("update_memory", update_memory)
builder.add_node("condense", condense)
builder.add_node("retrieve", retrieve)
builder.add_node("grade", grade)
builder.add_node("generate", generate)
builder.add_node("rewrite", rewrite)

builder.add_edge(START, "prepare")
builder.add_edge("prepare", "update_memory")
builder.add_edge("update_memory", "condense")
builder.add_edge("condense", "retrieve")
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


def build_local_graph(
    store: BaseStore | None = None, checkpointer: BaseCheckpointSaver | None = None
) -> CompiledStateGraph:
    """Compile the graph for in-process use (eval runner, notebook, scripts).

    Outside the server nothing injects persistence, so local callers get
    in-memory versions (per process; lost on exit, fine for evaluation and
    exploration): a store for ``update_memory``, and a checkpointer so a
    thread keeps its conversation history across turns. Every invocation
    therefore needs ``config["configurable"]["thread_id"]``.

    Args:
        store: Store to use; defaults to a fresh ``InMemoryStore``.
        checkpointer: Checkpointer to use; defaults to a fresh ``InMemorySaver``.
    """
    return builder.compile(store=store or InMemoryStore(), checkpointer=checkpointer or InMemorySaver())
