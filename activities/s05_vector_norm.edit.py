# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "numpy>=2.0"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    return (np,)


@app.cell
def _(np):
    v = np.array([3.0, 4.0, 12.0])
    np.linalg.norm(v)
    return


@app.cell
def _(np):
    coord_a = np.array([0.0, 0.0, 0.0])
    coord_b = np.array([1.805, 1.805, 0.0])
    np.linalg.norm(coord_b - coord_a)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
