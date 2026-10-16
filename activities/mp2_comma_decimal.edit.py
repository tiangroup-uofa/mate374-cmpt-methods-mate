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
def _(sp):
    # Intended: target value P = 2.5, so the root is sqrt(2.5) = 1.5811
    P = 2,5

    def residual(x):
        return x**2 - P

    solution = sp.optimize.root_scalar(residual, bracket=[0.0, 5.0], method="bisect")
    solution.root
    return


if __name__ == "__main__":
    app.run()
