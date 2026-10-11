"""Streamlit chat client for the KDIGO guideline RAG graph.

Talks to a LangGraph API server over ``langgraph_sdk`` -- ``pixi run dev``
locally, the Docker deployment later (set ``LANGGRAPH_API_URL``) -- so it never
imports the RAG package. Run with ``pixi run ui`` while ``pixi run dev`` is up.
"""

import os

import streamlit as st
from langgraph_sdk import get_sync_client
from ui_helpers import GUIDELINE_FILTERS, interpreted_as, pipeline_summary, sources_from_state

API_URL = os.environ.get("LANGGRAPH_API_URL", "http://127.0.0.1:2024")
ASSISTANT_ID = "ckd_guideline_rag"

st.set_page_config(page_title="KDIGO Guideline Assistant", page_icon="🩺", layout="centered")


@st.cache_resource
def client():
    """One SDK client per server URL, shared across reruns."""
    return get_sync_client(url=API_URL)


def new_conversation() -> None:
    """Start a fresh LangGraph thread and clear the visible history."""
    st.session_state.thread_id = client().threads.create()["thread_id"]
    st.session_state.turns = []


def render_turn(turn: dict, show_debug: bool) -> None:
    """One question/answer exchange, with its sources and (optionally) pipeline details."""
    with st.chat_message("user"):
        st.markdown(turn["question"])
    with st.chat_message("assistant"):
        if turn.get("interpreted_as"):
            st.caption(f"Interpreted as: {turn['interpreted_as']}")
        st.markdown(turn["answer"])
        retrieved = [s for s in turn["sources"] if not s.reference]
        references = [s for s in turn["sources"] if s.reference]
        if retrieved:
            with st.expander(f"Sources ({len(retrieved)} passages)"):
                for s in retrieved:
                    page = f" · p. {s.page}" if s.page else ""
                    st.markdown(f"**{s.label}**{page}")
                    st.text(s.text[:1500] + ("…" if len(s.text) > 1500 else ""))
        if references:
            with st.expander("Reference: KDIGO grading tables (always provided)"):
                for s in references:
                    st.markdown(f"**{s.label}**")
                    st.text(s.text)
        if show_debug:
            st.caption("Pipeline")
            st.json(turn["summary"], expanded=False)


with st.sidebar:
    st.header("Settings")
    filter_label = st.selectbox("Search", list(GUIDELINE_FILTERS), index=0)
    user_id = st.text_input(
        "User ID", value="demo-user", help="Scopes what the assistant remembers about you."
    )
    show_debug = st.toggle("Show pipeline details", value=False)
    if st.button("New conversation", use_container_width=True):
        new_conversation()
    st.divider()
    st.caption(
        "Answers come only from four indexed KDIGO guidelines (CKD 2024, AKI/AKD 2026 draft, "
        "Glomerular 2021, BP in CKD 2021) and cite their sources. Follow-ups use the conversation "
        "for context; each answer is still built only from passages retrieved for it. Not medical advice."
    )
    st.caption(f"Server: `{API_URL}`")

st.title("KDIGO Guideline Assistant")

try:
    if "thread_id" not in st.session_state:
        new_conversation()
except Exception as e:  # noqa: BLE001 -- surface any connection problem in the UI
    st.error(f"Can't reach the LangGraph server at {API_URL} — is `pixi run dev` running?\n\n{e}")
    st.stop()

for turn in st.session_state.turns:
    render_turn(turn, show_debug)

if question := st.chat_input("Ask about CKD, AKI, glomerular disease, or BP in CKD…"):
    with st.spinner("Retrieving, grading, and answering…"):
        try:
            state = client().runs.wait(
                st.session_state.thread_id,
                ASSISTANT_ID,
                input={"question": question},
                config={
                    "configurable": {"user_id": user_id, "guideline_filter": GUIDELINE_FILTERS[filter_label]}
                },
            )
        except Exception as e:  # noqa: BLE001 -- show server/graph errors instead of crashing the app
            st.error(f"The request failed: {e}")
            st.stop()
    turn = {
        "question": question,
        "answer": state.get("answer", "(no answer returned)"),
        "sources": sources_from_state(state),
        "summary": pipeline_summary(state),
        "interpreted_as": interpreted_as(state),
    }
    st.session_state.turns.append(turn)
    render_turn(turn, show_debug)
