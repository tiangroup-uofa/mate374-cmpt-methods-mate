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
    # Intended: square roots of the positions 1 to 5 m
    x = np.array([1, 2, 3, 4, 5])
    y = np.zeros_like(x)
    for i in range(x.size):
        y[i] = np.sqrt(x[i])

    print("x dtype:", x.dtype)
    print("y:", y)
    return


if __name__ == "__main__":
    app.run()
