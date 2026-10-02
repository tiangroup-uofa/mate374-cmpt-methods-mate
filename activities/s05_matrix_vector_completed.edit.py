# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "numpy>=2.0"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    return mo, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # S05 · Matrix–vector products (completed)

    A $3\times3$ matrix times a three-component vector is defined.
    """)
    return


@app.cell
def _(np):
    A = np.array([[2.0, 1.0, 0.0],
                  [1.0, 3.0, 1.0],
                  [0.0, 1.0, 4.0]])
    v = np.array([1.0, 2.0, 3.0])
    A @ v
    return (v,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `B` below is $3\times2$: each row has two entries, so `B @ v` with a
    three-component `v` raises `ValueError`. Two products are defined:
    `B` times a two-component vector `w`, and the $2\times3$ transpose
    `B.T` times `v`.
    """)
    return


@app.cell
def _(np):
    B = np.array([[1.0, 2.0],
                  [3.0, 4.0],
                  [5.0, 6.0]])
    B.shape
    return (B,)


@app.cell
def _(B, np):
    w = np.array([1.0, 1.0])
    B @ w, (B @ w).shape
    return


@app.cell
def _(B, v):
    B.T @ v, (B.T @ v).shape
    return


if __name__ == "__main__":
    app.run()
