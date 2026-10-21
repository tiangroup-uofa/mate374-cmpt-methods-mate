# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "numpy>=2.0", "scipy>=1.15"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    from scipy.differentiate import derivative

    return derivative, np


@app.cell
def _(np):
    h = 1.0
    x = np.arange(0.0, 6.0, h)
    fx = x**3
    print("np.diff / h :", np.diff(fx) / h, "  <- only n-1 values")
    print("np.gradient :", np.gradient(fx, h, edge_order=2))
    print("exact 3x^2  :", 3 * x**2)
    return


@app.cell
def _(np):
    xu = np.array([0.0, 0.5, 1.5, 2.0, 3.5])   # Uneven spacing
    print("np.gradient(y, x):", np.gradient(xu**3, xu, edge_order=2))
    print("exact 3x^2       :", 3 * xu**2)
    return


@app.cell
def _(derivative, np):
    res = derivative(np.sin, 1.0)     # Needs a callable function f(x)
    print("df    =", res.df, "  exact cos(1) =", np.cos(1.0))
    print("error =", res.error, "  function calls =", res.nfev)
    return


if __name__ == "__main__":
    app.run()
