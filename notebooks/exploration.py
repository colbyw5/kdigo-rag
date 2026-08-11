import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")

with app.setup:
    from dotenv import load_dotenv

    load_dotenv()


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # kdigo-guideline-rag — exploration

    Ask a question, inspect what got retrieved and graded, and see the
    final cited answer. Uses the already-ingested Chroma collection --
    run `pixi run ingest` first if it's empty.
    """)
    return


@app.cell
def _():
    from kdigo_guideline_rag.graph import graph

    return (graph,)


@app.cell(hide_code=True)
def _(mo):
    question = mo.ui.text(
        value="Should a patient with G3b A2 CKD be on an SGLT2 inhibitor?",
        label="Question",
        full_width=True,
    )
    guideline_filter = mo.ui.dropdown(
        options=["all", "ckd_only", "aki_only", "glomerular_only", "bp_only"],
        value="all",
        label="Guideline filter",
    )
    user_id = mo.ui.text(value="explorer", label="User ID (for memory)")
    run_button = mo.ui.run_button(label="Ask")
    mo.vstack([question, mo.hstack([guideline_filter, user_id]), run_button])
    return guideline_filter, question, run_button, user_id


@app.cell
def _(graph, guideline_filter, mo, question, run_button, user_id):
    mo.stop(not run_button.value, mo.md("*Click **Ask** to run a query.*"))

    config = {
        "configurable": {
            "guideline_filter": guideline_filter.value,
            "user_id": user_id.value,
        }
    }
    result = graph.invoke({"question": question.value}, config=config)
    return (result,)


@app.cell(hide_code=True)
def _(mo, result):
    mo.md(f"""
    ## Answer\n\n{result['answer']}
    """)
    return


@app.cell(hide_code=True)
def _(mo, result):
    mo.md(f"""
    ## Pipeline internals

    - Retrieved: **{len(result["documents"])}** chunks
    - Graded relevant: **{len(result["graded_documents"])}**
    - Rewrites used: **{result["rewrite_count"]}**
    - User context (memory): {result.get("user_context") or "*(none learned yet)*"}
    """)
    return


@app.cell(hide_code=True)
def _(mo, result):
    rows = [
        {
            "guideline": doc.metadata.get("source_guideline"),
            "chapter": doc.metadata.get("chapter"),
            "type": doc.metadata.get("label_type"),
            "rec #": doc.metadata.get("recommendation_number"),
            "page": doc.metadata.get("page"),
            "text": doc.page_content[:200],
        }
        for doc in result["graded_documents"]
    ]
    mo.vstack(
        [
            mo.md("### Graded-relevant chunks (what the answer was built from)"),
            mo.ui.table(rows, selection=None),
        ]
    )
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
