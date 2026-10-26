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
    x = np.array([0.0, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50])  # mm
    f = np.array([1.000, 0.649, 0.363, 0.173, 0.069, 0.023, 0.006])  # excess carbon, wt%
    reference = 34.36  # analytical reference, g/m² (rounded)
    return f, reference, x


@app.cell
def _(f, integrate, np, x):
    M_trapezoid = 78.5 * np.trapezoid(f, x)
    M_simpson = 78.5 * integrate.simpson(f, x=x)
    return M_simpson, M_trapezoid


@app.cell
def _(M_simpson, M_trapezoid, mo, reference):
    mo.md(f"""
    **Trapezoidal uptake:** {M_trapezoid:.2f} g/m²

    **Simpson's uptake:** {M_simpson:.2f} g/m²

    **Reference:** {reference:.2f} g/m²
    """)
    return


if __name__ == "__main__":
    app.run()
