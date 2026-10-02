# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "numpy>=2.0"]
# ///

# The second product deliberately raises a shape error.
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
    A = np.array([[2.0, 1.0, 0.0],
                  [1.0, 3.0, 1.0],
                  [0.0, 1.0, 4.0]])
    v = np.array([1.0, 2.0, 3.0])
    A @ v
    return (v,)


@app.cell
def _(np, v):
    B = np.array([[1.0, 2.0],
                  [3.0, 4.0],
                  [5.0, 6.0]])
    B @ v
    return (B,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
