# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "numpy", "scipy", "matplotlib"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="Midterm · Q3")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Midterm · Q3 · mass spectrometry

    Complete the following if you want to use Python code:

    1. Enter the intensity matrix $A$ and measured intensity vector $b$.
    2. Solve $Ax=b$ and verify the calculated intensities.
    3. Recalculate the amounts using the corrected measurement and find the absolute relative errors.
    """)
    return


@app.cell
def _():
    # Click Save when finished.
    import marimo as mo
    import numpy as np

    return mo, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### (a) Matrix formulation · 10 points
    """)
    return


@app.cell
def _(np):
    # Enter the 4-by-4 intensity matrix A and measured intensity vector b from the exam paper.
    # Replace the NaN values with the corresponding table entries.
    A = np.full((4, 4), np.nan)
    b = np.full(4, np.nan)
    return A, b


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## (b) Amounts and verification · 15 points
    """)
    return


@app.cell
def _(A, b, mo, np):
    # Solve Ax = b, then compare the calculated and measured intensities.
    x = np.linalg.solve(A, b)
    calculated_intensities = A @ x
    check_passed = bool(np.allclose(calculated_intensities, b))
    mo.callout(
        "PASS: calculated intensities match b."
        if check_passed else
        "NOT YET: check the entered matrix, vector, and solution.",
        kind="success" if check_passed else "danger",
    )
    return calculated_intensities, x


@app.cell
def _(b, calculated_intensities, x):
    x, calculated_intensities, b
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## (c) Measurement correction · 10 points

    Use the corrected measured intensity vector $b_{\mathrm{true}}=[22,34,29,31]$. Calculate $x_{\mathrm{true}}$ and the componentwise absolute relative error $|x-x_{\mathrm{true}}|/x_{\mathrm{true}}$.
    """)
    return


@app.cell
def _(np):
    # Corrected intensity vector provided in the exam question.
    b_true = np.array([22, 34, 29, 31])
    return (b_true,)


@app.cell
def _(A, b_true, np, x):
    # Calculate corrected amounts and componentwise absolute relative errors.
    x_true = np.linalg.solve(A, b_true)
    e_rel = np.abs(x - x_true) / x_true

    x_true, e_rel
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## (d) Error amplification · 5 points

    Your results from part (c) show that some relative errors in the calculated amounts are much larger than the sensor error. Which concept from this course explains this behaviour? Briefly explain your reasoning.
    """)
    return


@app.cell
def _():
    ## Use any tests below (optional)
    return


if __name__ == "__main__":
    app.run()
