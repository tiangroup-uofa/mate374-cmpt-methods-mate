# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.23.16",
#     "numpy>=2.0",
# ]
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
    mo.md(r"""
    ### Grid-refinement experiment

    Approximate $\int_0^\pi \sin(x)\,\mathrm{d}x = 2$ with NumPy's
    trapezoidal rule. Change the grid resolution—or edit the code.
    """)
    return


@app.cell
def _(mo):
    points = mo.ui.slider(
        start=5,
        stop=101,
        step=4,
        value=21,
        label="Number of grid points, N",
        show_value=True,
    )
    points
    return (points,)


@app.cell
def _(np, points):
    x = np.linspace(0.0, np.pi, points.value)
    y = np.sin(x)
    approximation = np.trapezoid(y, x=x)
    absolute_error = abs(approximation - 2.0)
    spacing = x[1] - x[0]
    return absolute_error, approximation, spacing


@app.cell
def _(absolute_error, approximation, mo, spacing):
    mo.callout(
        mo.md(
            f"""
            **NumPy result**

            - Grid spacing: $\\Delta x = {spacing:.4f}$
            - Trapezoidal approximation: $I_N = {approximation:.8f}$
            - Absolute error: $|I_N - 2| = {absolute_error:.2e}$
            """
        ),
        kind="success" if absolute_error < 1.0e-3 else "warn",
    )
    return


if __name__ == "__main__":
    app.run()
