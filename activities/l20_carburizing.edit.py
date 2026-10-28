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
    # depth x in mm, excess carbon f in wt%
    x = np.array([0.0, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50])
    f = np.array([1.000, 0.649, 0.363, 0.173, 0.069, 0.023, 0.006])
    return f, x


@app.cell
def _(f, simpson, trapezoid, x):
    print("trapezoid:", 78.5 * trapezoid(f, x))
    print("simpson:  ", 78.5 * simpson(f, x=x))
    return


@app.cell
def _(f, np, x):
    def trapezoid_method(y, x):
        return np.sum((x[1:] - x[:-1]) * (y[:-1] + y[1:]) / 2)


    def simpson_method(y, x):
        h = x[1] - x[0]
        return h / 3 * (y[0] + 4 * np.sum(y[1:-1:2])
                        + 2 * np.sum(y[2:-1:2]) + y[-1])


    print("trapezoid method:", 78.5 * trapezoid_method(f, x))
    print("simpson method:  ", 78.5 * simpson_method(f, x))
    return


if __name__ == "__main__":
    app.run()
