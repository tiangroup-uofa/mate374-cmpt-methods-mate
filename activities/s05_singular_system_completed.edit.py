# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "numpy>=2.0"]
# ///

# Native NumPy may raise LinAlgError for this singular matrix, while the
# browser build returns numbers, so the build exports it without executing.
# mate374: build-execute = false

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
    # S05 Example 2 · A 5 × 5 system with a hidden dependency (completed)

    Results differ between NumPy builds. In the browser, `inv` and `solve`
    return two different vectors; NumPy on a laptop may stop with
    `LinAlgError: Singular matrix`.
    """)
    return


@app.cell
def _(np):
    M = np.array([[ 1,  3, -9,  6, 4],
                  [ 2, -1,  6,  7, 1],
                  [ 3,  2, -3, 15, 5],
                  [ 8, -1,  1,  4, 2],
                  [11,  1, -2, 18, 7]], dtype=float)
    x_true = np.array([0.0, 1.0, 1.0, 0.0, 1.0])
    return M, x_true


@app.cell
def _(M, x_true):
    b = M @ x_true
    b
    return (b,)


@app.cell
def _(M, b, np):
    x_inv = np.linalg.inv(M) @ b
    x_solve = np.linalg.solve(M, b)
    x_inv, x_solve
    return x_inv, x_solve


@app.cell
def _(M, b, x_inv, x_solve):
    M @ x_inv - b, M @ x_solve - b
    return


@app.cell
def _(M, np):
    np.linalg.det(M), np.linalg.cond(M), np.abs(np.linalg.inv(M)).max()
    return


@app.cell
def _(M, np):
    np.linalg.matrix_rank(M)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The same calculation with `dtype=np.float32`, which carries about
    7 significant digits:
    """)
    return


@app.cell
def _(M, np, x_true):
    M_single = np.array(M, dtype=np.float32)
    b_single = M_single @ np.array(x_true, dtype=np.float32)
    np.linalg.inv(M_single) @ b_single, np.linalg.solve(M_single, b_single)
    return


if __name__ == "__main__":
    app.run()
