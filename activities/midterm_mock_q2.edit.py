# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "numpy", "scipy", "matplotlib"]
# ///

# mate374: build-execute = false

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="Mock midterm · Q2")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Mock midterm · Q2 · van der Waals equation

    Complete the following if you want to use Python code:

    1. Implement `F(V)` and select a valid bracket.
    2. Find the volume using bisection and check the pressure.
    3. Compare with Newton–Raphson starting at 0.20 L.
    """)
    return


@app.cell
def _():
    # Click Save when finished.
    import marimo as mo
    import numpy as np
    from scipy.optimize import root_scalar

    return mo, np, root_scalar


@app.cell
def _():
    # Given constants: use L, atm, mol, and K.
    n = 3.0
    T = 380.0
    P = 85.0
    a = 4.7
    b = 0.0371
    R = 0.08206
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### (a) Residual function · 5 points

    Write your residual and identify the unknown and its units on the answer sheet.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### (b) A valid bracket · 5 points

    Implement `F(V)`. Select a sign-changing bracket within $V>nb$ that does not cross a singularity.
    """)
    return


@app.cell
def _(np):
    # Replace NaN with your residual expression using the given constants.
    def F(V):
        return np.nan

    return (F,)


@app.cell
def _(F, np):
    volumes = np.array([0.05, 0.10, 0.15, 0.20, 0.25])
    residuals = np.array([F(v_) for v_ in volumes])
    print("Volume, residuals")
    for v_, r_ in zip(volumes, residuals):
        print(v_, r_)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### (c) Bisection · 20 points

    Find the volume to at least five significant figures and check the calculated pressure. Fix the placeholders below.
    """)
    return


@app.cell
def _(F, root_scalar):
    # Replace ... with your bracket endpoints in L.
    # use bisect_result.root for the final root
    bisect_result = root_scalar(F, bracket=[..., ...], method="bisect", xtol=1e-10)
    bisect_result
    return (bisect_result,)


@app.cell
def _(F, bisect_result, mo, np):
    # Volume tolerance is 1e-10 L; allow 1e-4 atm in the pressure residual.
    check_passed = bool(np.isclose(F(bisect_result.root), 0, atol=1e-4))
    mo.callout(
        "PASS: residual check ok."
        if check_passed else
        "NOT YET: check the root and calculated pressure.",
        kind="success" if check_passed else "danger",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### (d) Newton–Raphson · 10 points

    Show the first update from 0.20 L. Does Newton–Raphson converge to the same root?
    """)
    return


@app.cell
def _(np):
    # Implement the derivative of your residual.
    # This is Optional, if you choose
    def dF(V):
        return np.nan

    return (dF,)


@app.cell
def _(F, dF, root_scalar):
    # The following needs to be fixed
    # If you do not provide the derivative, just remove fprime=dF
    newton_result = root_scalar(F, x0=0.20, fprime=dF, method="newton", xtol=1e-10)
    newton_result
    return


if __name__ == "__main__":
    app.run()
