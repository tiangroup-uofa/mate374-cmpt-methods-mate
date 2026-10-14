# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "numpy"]
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
    # Two-bead spring chain, with unit spring constants in consistent units.
    # Predict the force vector before running. Try changing a displacement.
    A = np.array([[2.0, -1.0], [-1.0, 1.0]])
    x = np.array([1.0, 2.0])

    print("Matrix shape:", A.shape)
    print("Vector shape:", x.shape)
    print("First row, first column:", A[0, 0])
    print("Second row, second column:", A[1, 1])
    print("Force vector A @ x:", A @ x)
    # Check by hand: 2*x[0] - x[1], then -x[0] + x[1].
    return


if __name__ == "__main__":
    app.run()
