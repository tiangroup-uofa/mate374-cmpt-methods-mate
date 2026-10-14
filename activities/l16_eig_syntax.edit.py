# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "numpy>=2.0"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell
def _():
    import numpy as np
    return (np,)


@app.cell
def _(np):
    H = np.array([[4.0, 6.0], [6.0, 4.0]])
    return (H,)


@app.cell
def _(H, np):
    # General square matrix: eigenvectors are columns.
    values, vectors = np.linalg.eig(H)
    print("Eigenvalues:", values)
    print("Eigenvectors (columns):\n", vectors)
    return


@app.cell
def _(H, np):
    # Real symmetric matrix: eigenvalues are sorted in ascending order.
    symmetric_values, symmetric_vectors = np.linalg.eigh(H)
    print("Eigenvalues:", symmetric_values)
    print("Eigenvectors (columns):\n", symmetric_vectors)
    return


if __name__ == "__main__":
    app.run()
