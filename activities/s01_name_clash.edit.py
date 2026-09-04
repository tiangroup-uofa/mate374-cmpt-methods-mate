# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.24.0",
# ]
# ///

# This notebook must open with a deliberate multiple-definition error.
# mate374: build-execute = false

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # One name defined twice

    This notebook deliberately breaks one of marimo's rules. The two code cells
    below both define the global name `i`, so marimo reports a name conflict.
    """)
    return


@app.cell
def _():
    i = 3
    i
    return (i,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Remove the conflict

    1. Open **Edit code** if the code cells are hidden.
    2. In the **lower** code cell, change both occurrences of `i` to `_i`.
    3. Run that cell again.

    The error disappears because a leading underscore makes `_i` private to its
    cell. The names `i` and `_i` are intentionally simplistic for this tiny
    demonstration. In course calculations, use descriptive names for values that
    carry physical or numerical meaning.
    """)
    return


@app.cell
def _():
    i = 5
    i
    return (i,)


@app.cell
def _(mo):
    mo.md(r"""
    A private name is useful for a short-lived intermediate value. A quantity that
    other cells need should instead have one descriptive global name, defined in
    one cell.
    """)
    return


if __name__ == "__main__":
    app.run()
