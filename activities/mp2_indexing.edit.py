# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "numpy>=2.0"]
# ///

# This notebook opens with a deliberate error for students to fix.
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
    A = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])

    # Intended: the entry a_23 (row 2, column 3), which is 6
    A[2, 3]
    return


if __name__ == "__main__":
    app.run()
