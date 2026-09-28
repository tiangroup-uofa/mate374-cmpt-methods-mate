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
    s = 2.0                         # One number
    t = np.float64(2.0)
    print(s, t)
    return


@app.cell
def _(np):
    a = np.array([1, 2, 3])          # One-dimensional array
    row = np.array([[1, 2, 3]])      # One row
    column = np.array([[1], [2], [3]])  # One column
    print("a:", a, "shape:", a.shape)
    print("row:", row, "shape:", row.shape)
    print("column:\n", column, "shape:", column.shape)
    return (a,)


@app.cell
def _(a, np):
    r = np.atleast_2d(a)             # Make a row from a
    c = r.T                         # Turn the row into a column
    print("a.T:", a.T, "shape:", a.T.shape)
    print("r:", r, "shape:", r.shape)
    print("c:\n", c, "shape:", c.shape)
    return


if __name__ == "__main__":
    app.run()
