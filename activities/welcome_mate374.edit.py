# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.24.0",
# ]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Hello, MATE 374!

    Use these two controls to check that the browser notebook responds to you.
    No Python installation is required.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    name = mo.ui.text(
        label="Your name or preferred name",
        placeholder="Type your name",
    )
    intended_gpa = mo.ui.slider(
        start=0.0,
        stop=5.0,
        step=0.1,
        value=4.0,
        label="Completely unofficial target GPA for MATE 374",
        show_value=True,
    )
    mo.vstack([name, intended_gpa], align="start", gap=1)
    return intended_gpa, name


@app.cell(hide_code=True)
def _(intended_gpa, mo, name):
    display_name = name.value.strip() or "future materials modeler"
    if intended_gpa.value < 3.0:
        response = "WHAT?"
    elif intended_gpa.value > 4.0:
        response = "impossible"
    else:
        response = "good work"

    mo.callout(
        mo.md(
            f"""
            ## Welcome to MATE 374, {display_name}!

            Your entirely unofficial target is **{intended_gpa.value:.1f}**.

            **{response}**

            The slider is only a joke: it does not predict or affect your grade.
            If this message changes when you edit your name or move the slider,
            the interactive notebook is working.
            """
        ),
        kind="success",
    )
    return


if __name__ == "__main__":
    app.run()
