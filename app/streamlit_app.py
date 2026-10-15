"""Streamlit chat client for the guideline RAG graph (KDIGO + ADA).

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

st.set_page_config(page_title="Kidney Guideline Explorer", page_icon="🩺", layout="centered")


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
            with st.expander("What the evidence grades mean"):
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
        "User ID",
        value="demo-user",
        help="The assistant remembers your background and interests under this ID.",
    )
    show_debug = st.toggle("Show pipeline details", value=False)
    if st.button("New conversation", use_container_width=True):
        new_conversation()
    st.divider()
    st.caption(
        "A study aid for learning what the guidelines recommend: KDIGO (CKD 2024, AKI/AKD 2026 draft, "
        "Glomerular 2021, BP in CKD 2021) and the ADA Standards of Care 2026 (sections 9-11). "
        "Answers come only from the guideline text and cite their sources. "
        "Not medical advice, and not for decisions about individual patients."
    )
    st.caption(f"Server: `{API_URL}`")

st.title("Kidney Guideline Explorer")
st.caption("Learn what the KDIGO and ADA guidelines recommend, with every claim cited.")

try:
    if "thread_id" not in st.session_state:
        new_conversation()
except Exception as e:  # noqa: BLE001 -- surface any connection problem in the UI
    st.error(f"Can't reach the LangGraph server at {API_URL} — is `pixi run dev` running?\n\n{e}")
    st.stop()

for turn in st.session_state.turns:
    render_turn(turn, show_debug)

if question := st.chat_input(
    "Ask what the guidelines say about CKD, AKI, glomerular disease, blood pressure or diabetes…"
):
    with st.spinner("Searching the guidelines…"):
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
