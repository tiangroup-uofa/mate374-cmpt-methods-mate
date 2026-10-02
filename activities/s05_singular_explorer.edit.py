# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "numpy>=2.0"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    return mo, np


@app.cell
def _(mo):
    M_input = mo.ui.matrix(
        [[1, 3, -9, 6, 4],
         [2, -1, 6, 7, 1],
         [3, 2, -3, 15, 5],
         [8, -1, 1, 4, 2],
         [11, 1, -2, 18, 7]],
        step=1,
        precision=0,
        label="Drag one entry of M",
    )
    M_input
    return (M_input,)


@app.cell
def _(M_input, np):
    M = np.array(M_input.value, dtype=float)
    print(f"cond(M) = {np.linalg.cond(M):.3g}")
    print(f"det(M)  = {np.linalg.det(M):.3g}")
    return (M,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
