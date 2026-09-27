# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "numpy>=2.0", "scipy"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    from numpy.polynomial import Polynomial
    from scipy.interpolate import lagrange, CubicSpline, PchipInterpolator
    return CubicSpline, PchipInterpolator, Polynomial, lagrange, np


@app.cell
def _(np):
    # Keep input values distinct and increasing.
    x_data = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y_data = np.array([1.1, 1.7, 3.2, 4.8, 7.5])
    x_query = 1.5
    return x_data, x_query, y_data


@app.cell
def _(CubicSpline, PchipInterpolator, Polynomial, lagrange, np, x_data, x_query, y_data):
    p = Polynomial.fit(x_data, y_data, deg=len(x_data) - 1)
    p_lagrange = lagrange(x_data, y_data)
    y_linear = np.interp(x_query, x_data, y_data)
    cubic = CubicSpline(x_data, y_data)
    pchip = PchipInterpolator(x_data, y_data)

    print(f"Estimates at x = {x_query}:")
    print("Global polynomial:", p(x_query))
    print("Lagrange polynomial:", p_lagrange(x_query))
    print("Piecewise linear:", y_linear)
    print("Cubic spline:", cubic(x_query))
    print("PCHIP:", pchip(x_query))
    return


if __name__ == "__main__":
    app.run()
