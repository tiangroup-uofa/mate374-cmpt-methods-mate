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
    # S05 Example 1 · A well-conditioned 3 × 3 system (completed)

    Solve $\mathbf{A}\mathbf{x}=\mathbf{b}$ with `inv` and `solve`, then check
    the residual, the relative residual, and the condition number.
    """)
    return


@app.cell
def _(np):
    A = np.array([[6.0, 2.0, 3.0],
                  [4.0, 11.0, 6.0],
                  [7.0, 8.0, 16.0]])
    b = np.array([19.0, 44.0, 71.0])
    return A, b


@app.cell
def _(A, b, np):
    x_inv = np.linalg.inv(A) @ b
    x_solve = np.linalg.solve(A, b)
    x_inv, x_solve
    return (x_solve,)


@app.cell
def _(A, b, np, x_solve):
    residual = A @ x_solve - b
    residual_norm = np.linalg.norm(residual)
    relative_residual = residual_norm / np.linalg.norm(b)
    kappa = np.linalg.cond(A)
    residual, relative_residual, kappa
    return


if __name__ == "__main__":
    app.run()
