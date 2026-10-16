# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "numpy>=2.0"]
# ///

# This notebook opens with a deliberate error for students to fix.
# mate374: build-execute = false

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    return (np,)


@app.cell
def _(np):
    # Intended: four concentrations 0.5, 1.0, 1.5, 2.0 mol/L
    c = np.array([0.5, 1.0, 1,5, 2.0])
    print("c =", c, "shape:", c.shape)
    return (c,)


@app.cell
def _(c, np):
    V = np.array([1.0, 2.0, 3.0, 4.0])   # volumes in L
    moles = c * V
    moles
    return


if __name__ == "__main__":
    app.run()
