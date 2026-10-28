# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "numpy>=2.0", "scipy>=1.12"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    from numpy import trapezoid
    from scipy.integrate import simpson

    return np, simpson, trapezoid


@app.cell
def _(np):
    def trapezoid_method(y, x):
        return np.sum((x[1:] - x[:-1]) * (y[:-1] + y[1:]) / 2)


    def simpson_method(y, x):
        h = x[1] - x[0]
        # We also need to check that the spacings are equal and the number of intervals is even:
        # Equal spacing:            np.allclose(np.diff(x), h)
        # Even number of intervals: (len(x) - 1) % 2 == 0
        return h / 3 * (y[0] + 4 * np.sum(y[1:-1:2])
                        + 2 * np.sum(y[2:-1:2]) + y[-1])

    return simpson_method, trapezoid_method


@app.cell
def _(np, simpson, simpson_method, trapezoid, trapezoid_method):
    x = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    y = np.array([0.0, 0.625, 2.0, 4.875, 10.0])

    print("trapezoid:", trapezoid_method(y, x), trapezoid(y, x))
    print("simpson:  ", simpson_method(y, x), simpson(y, x=x))
    return


if __name__ == "__main__":
    app.run()
