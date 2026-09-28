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
    A = np.array([[1, 2, 3],
                  [4, 5, 6]])
    B = np.array([[1, 0],
                  [0, 1],
                  [1, 1]])
    print("A @ B:\n", A @ B, "shape:", (A @ B).shape)
    print("B @ A:\n", B @ A, "shape:", (B @ A).shape)
    return


@app.cell
def _(np):
    a = np.array([1, 2, 3])
    row = np.atleast_2d(a)
    column = row.T
    print("a @ a:", a @ a)           # A single number
    print("row @ column:\n", row @ column)
    print("Shape:", (row @ column).shape)
    print("column @ row:\n", column @ row)
    print("Shape:", (column @ row).shape)
    return


if __name__ == "__main__":
    app.run()
