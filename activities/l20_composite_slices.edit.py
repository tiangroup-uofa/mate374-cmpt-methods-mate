# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "numpy>=2.0", "scipy>=1.12"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from scipy import integrate

    return integrate, mo, np


@app.cell
def _(np):
    x = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    y = np.array([0.0, 0.625, 2.0, 4.875, 10.0])
    return x, y


@app.cell
def _(np, x, y):
    trapezoid_slices = np.sum((x[1:] - x[:-1]) * (y[:-1] + y[1:]) / 2)
    return (trapezoid_slices,)


@app.cell
def _(mo, np, x):
    spacings = np.diff(x)
    N = len(x) - 1
    mo.stop(
        not (np.allclose(spacings, spacings[0]) and N % 2 == 0),
        mo.md("The 1/3 slice formula needs equal spacing and an even number of subintervals."),
    )
    h = spacings[0]
    return (h,)


@app.cell
def _(h, np, y):
    simpson_slices = h / 3 * (
        y[0] + y[-1]
        + 4 * np.sum(y[1:-1:2])
        + 2 * np.sum(y[2:-1:2])
    )
    return (simpson_slices,)


@app.cell
def _(integrate, mo, np, simpson_slices, trapezoid_slices, x, y):
    mo.md(f"""
    **Trapezoidal:** slices {trapezoid_slices:.8f}; NumPy {np.trapezoid(y, x):.8f}

    **Simpson's 1/3:** slices {simpson_slices:.8f}; SciPy {integrate.simpson(y, x=x):.8f}
    """)
    return


if __name__ == "__main__":
    app.run()
