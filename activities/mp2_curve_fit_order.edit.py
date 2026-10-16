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
    import scipy as sp
    return (np, sp)


@app.cell
def _(np, sp):
    t = np.array([0.0, 1.0, 2.0, 3.0, 4.0])        # h
    c = np.array([2.0, 1.2, 0.75, 0.42, 0.26])     # mol/L

    def model(c0, k, t):
        return c0 * np.exp(-k * t)

    params, cov = sp.optimize.curve_fit(model, xdata=t, ydata=c, p0=[1.0, 1.0])
    print("c0, k =", params)
    print("standard errors:", np.sqrt(np.diag(cov)))
    return


if __name__ == "__main__":
    app.run()
