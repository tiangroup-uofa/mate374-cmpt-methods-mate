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
def _():
    def F(x, y):
        return [x**2 + 2*x + 2*y**2 - 25,
                2*x**3 - y**2 + 4*y - 20]
    return (F,)


@app.cell
def _(F, sp):
    solution = sp.optimize.root(F, [1.0, 1.0])
    solution.x
    return


if __name__ == "__main__":
    app.run()
