# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "numpy", "scipy", "matplotlib"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="Mock final · Q2 · completed")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

    return mo, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Mock final · Q2 · Balancing a chemical reaction

    $\mathrm{FeS_2}+a\,\mathrm{O_2}+b\,\mathrm{H_2O}\rightarrow c\,\mathrm{Fe_2O_3}+d\,\mathrm{H_2SO_4}$.

    Save your edited notebook when finished.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### (a) Balance equations · 10 points

    Write one balance equation for each of Fe, S, O, and H. Rearrange the four equations into matrix form A x = b, with x = (a, b, c, d).
    """)
    return


@app.cell
def _(np):
    # Rows: Fe, S, O, H. Unknowns: a, b, c, d.
    A = np.array([
        [0, 0, 2, 0],
        [0, 0, 0, 1],
        [2, 1, -3, -4],
        [0, 2, 0, -2],
    ], dtype=float)
    rhs = np.array([1, 2, 0, 0], dtype=float)
    print(A, rhs, sep="\n")
    return A, rhs


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### (b) Solution · 10 points

    Solve the matrix equation for a, b, c, d using any suitable method. Identify your method and report the four coefficients.
    """)
    return


@app.cell
def _(A, np, rhs):
    # LU-based direct solve.
    coefficients = np.linalg.solve(A, rhs)
    print("a, b, c, d =", coefficients)
    print("Residual =", A @ coefficients - rhs)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### (c) An additional product · 5 points

    Suppose e FeO also forms as a third product. Are the element balances sufficient to determine all five unknown coefficients uniquely? Explain.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    No. Adding FeO introduces a fifth unknown but no new element balance. Four independent equations cannot determine five unknowns uniquely. An additional condition, such as the ratio of the two iron oxides, is needed.
    """)
    return


if __name__ == "__main__":
    app.run()
