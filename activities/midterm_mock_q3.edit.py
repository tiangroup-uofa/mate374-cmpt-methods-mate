# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "numpy", "scipy", "matplotlib"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="Mock midterm · Q3")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Mock midterm · Q3 · growth-rate fitting

    Complete the following if you want to use Python code:

    1. Transform the concentration and growth-rate data.
    2. Fit a line and calculate the model parameters.
    3. Find the concentration at $\mu=2.5\ \mathrm{h^{-1}}$.
    """)
    return


@app.cell
def _():
    # Click Save when finished.
    import marimo as mo
    import numpy as np
    from numpy.polynomial import Polynomial

    return Polynomial, mo, np


@app.cell
def _(np):
    # Given data: C in g/L and mu in 1/h.
    C = np.array([0.7, 0.9, 1.2, 1.4])
    mu = np.array([1.0, 2.0, 3.0, 4.0])
    mu_max = 2.1
    mu_target = 2.5
    return mu_max, mu_target


@app.cell
def _(mu_max):
    # The original growth rate function
    def mu_model(C, k_m, k_1):
        return mu_max * C ** 2 / (k_m + k_1 * C ** 2)

    return (mu_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### (a) Unknown parameters · 5 points

    Identify the two unknown model parameters on the answer sheet.

    ### (b) Linearization · 5 points

    Express $A$ and $B$ in $1/\mu=A/C^2+B$ in terms of the model parameters.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### (c) Linear regression · 20 points

    Complete the transformed data and parameter calculations below. Report the parameter units.
    """)
    return


@app.cell
def _(np):
    # Replace NaN values with the transformed arrays calculated from C and mu.
    x = np.full(4, np.nan)
    y = np.full(4, np.nan)
    return x, y


@app.cell
def _(Polynomial, x, y):
    # Fit y = A*x + B. convert() gives coefficients in the original x coordinate.
    fit = Polynomial.fit(x, y, deg=1).convert()
    B, A = fit.coef
    A, B
    return


@app.cell
def _(np):
    # Use your expression in question b to 
    # calculate the model parameters from A, B, and mu_max.
    k_m = np.nan
    k_1 = np.nan
    k_m, k_1
    return k_1, k_m


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### (d) Concentration · 10 points

    Find the positive concentration at $\mu=2.5\ \mathrm{h^{-1}}$. Fix the placeholders below.
    """)
    return


@app.cell
def _(np):
    # Calculate C_target using your fitted model, in g/L.
    # You can either use linearized model, or root finding with mu_model
    C_target = np.nan
    C_target
    return (C_target,)


@app.cell
def _(C_target, k_1, k_m, mu_model, mu_target):
    # Calculate the model growth rate at C_target, in 1/h.
    mu_check = mu_model(C_target, k_m=k_m, k_1=k_1)
    mu_check, mu_target
    return (mu_check,)


@app.cell
def _(mo, mu_check, mu_target, np):
    check_passed = bool(np.isclose(mu_check, mu_target))
    mo.callout(
        "PASS: calculated growth rate matches 2.5 1/h."
        if check_passed else
        "NOT YET: check the concentration and calculated growth rate.",
        kind="success" if check_passed else "danger",
    )
    return


if __name__ == "__main__":
    app.run()
