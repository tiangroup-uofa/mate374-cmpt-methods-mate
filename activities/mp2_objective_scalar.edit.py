# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "numpy>=2.0", "scipy"]
# ///

# This notebook opens with a deliberate error for students to fix.
# mate374: build-execute = false

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    import scipy as sp
    return (np, sp)


@app.cell
def _(np):
    t = np.array([0.0, 1.0, 2.0, 3.0, 4.0])        # h
    c = np.array([2.0, 1.2, 0.75, 0.42, 0.26])     # mol/L

    def model(t, c0, k):
        return c0 * np.exp(-k * t)

    def objective(params):
        c0, k = params
        return model(t, c0, k) - c
    return (objective,)


@app.cell
def _(objective, sp):
    fit = sp.optimize.minimize(objective, x0=[1.0, 1.0])
    fit.x
    return


if __name__ == "__main__":
    app.run()
