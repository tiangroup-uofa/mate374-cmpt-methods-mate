# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "numpy>=2.0", "scipy>=1.14"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    from scipy.linalg import solve_triangular
    return mo, np, solve_triangular


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # A3 Q2 · Direct-method checks

    **Q2.1:** Carry out partial-pivoted elimination by hand first. Predict which
    row becomes the first pivot row. Use the calculation below to check your
    final vector against the original four equations.
    """)
    return


@app.cell
def _(np):
    A4 = np.array([[0, 2, 1, -1], [4, 2, 0, 2],
                   [2, 1, 3, 1], [0, 4, 0, 2]], dtype=float)
    b4 = np.array([-1.5, 7.0, 5.5, 1.0])
    x4_check = np.linalg.solve(A4, b4)
    print("NumPy solution:", x4_check)
    print("Residual A @ x - b:", A4 @ x4_check - b4)
    # Replace this vector with your own manual answer to check it.
    x4_manual = np.array([0.0, 0.0, 0.0, 0.0])
    print("Your answer's residual:", A4 @ x4_manual - b4)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Q2.2 · Two loads, one factorization

    The supplied factors refer to a separate three-equation system.
    Inspect their product, then follow how each right-hand side passes through
    the same two triangular solves. `lower=True` selects forward substitution
    for the lower-triangular matrix. The second call uses backward substitution.
    Include the code and printed intermediate vectors if you choose the code route.
    """)
    return


@app.cell
def _(np):
    A3 = np.array([[4, 2, -2], [2, 4, 0], [-2, 2, 4]], dtype=float)
    L = np.array([[1, 0, 0], [0.5, 1, 0], [-0.5, 1, 1]])
    U = np.array([[4, 2, -2], [0, 3, 1], [0, 0, 2]], dtype=float)
    b1 = np.array([5.5, 4.0, -2.0])
    b2 = np.array([-1.0, 4.0, 6.5])
    print("L @ U =\n", L @ U)
    print("L @ U - A =\n", L @ U - A3)
    return A3, L, U, b1, b2


@app.cell
def _(A3, L, U, b1, b2, solve_triangular):
    y1 = solve_triangular(L, b1, lower=True)
    x1 = solve_triangular(U, y1)
    y2 = solve_triangular(L, b2, lower=True)
    x2 = solve_triangular(U, y2)
    print("y1 =", y1, "\nx1 =", x1, "\nA @ x1 =", A3 @ x1)
    print("y2 =", y2, "\nx2 =", x2, "\nA @ x2 =", A3 @ x2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Interpretation:** Which part of the calculation was reused when the load
    changed? For Q2.2(d), use the leading-order operation estimates in the handout
    to compare a thousand fresh eliminations with one factorization and a thousand
    pairs of triangular solves. The small matrices above illustrate the procedure.
    """)
    return


if __name__ == "__main__":
    app.run()
