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
    A = np.array([[4.0, -1.0, 0.0],
                  [-1.0, 4.0, -1.0],
                  [0.0, -1.0, 4.0]])
    b = np.array([1.0, 2.0, 3.0])
    x = np.linalg.solve(A, b)

    # Intended: the residual vector A x - b, which should be close to zero
    A * x - b
    return


if __name__ == "__main__":
    app.run()
