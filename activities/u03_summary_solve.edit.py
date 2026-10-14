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
    # Predict how doubling the force b changes the displacements x.
    A = np.array([[2.0, -1.0], [-1.0, 1.0]])
    b = np.array([0.0, 1.0])

    x = np.linalg.solve(A, b)
    r = A @ x - b

    print("Displacements:", x)
    print("Reconstructed forces:", A @ x)
    print("Residual:", r)
    print("Residual norm:", np.linalg.norm(r))
    print("Determinant:", np.linalg.det(A))
    print("Condition number:", np.linalg.cond(A))
    return


if __name__ == "__main__":
    app.run()
