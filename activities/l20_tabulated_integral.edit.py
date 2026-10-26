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
def _(integrate, np, x, y):
    trapezoid = np.trapezoid(y, x)
    simpson = integrate.simpson(y, x=x)
    return simpson, trapezoid


@app.cell
def _(mo, simpson, trapezoid):
    mo.md(f"""
    **Trapezoidal rule:** {trapezoid:.8f}

    **Simpson's rule:** {simpson:.8f}
    """)
    return


if __name__ == "__main__":
    app.run()
