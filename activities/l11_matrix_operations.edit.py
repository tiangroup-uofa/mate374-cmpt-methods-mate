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
    A = np.array([[1, 2],
                  [3, 4]])
    B = np.array([[5, 6],
                  [7, 8]])
    print("A + B:\n", A + B)         # Add matching entries
    print("2 * A:\n", 2 * A)         # Double every entry
    print("A.T:\n", A.T)             # Exchange rows and columns
    return


if __name__ == "__main__":
    app.run()
