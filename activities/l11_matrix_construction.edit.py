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
                  [4, 5, 6],
                  [7, 8, 9]])
    print(A)
    print("Shape:", A.shape)
    print("First row, second column:", A[0, 1])
    return


@app.cell
def _(np):
    r1 = np.array([1, 2, 3])
    r2 = np.array([4, 5, 6])
    r3 = np.array([7, 8, 9])
    B = np.vstack([r1, r2, r3])       # Stack rows
    print(B)
    print("Shape:", B.shape)
    return


@app.cell
def _(np):
    c1 = np.atleast_2d([1, 4, 7]).T
    c2 = np.atleast_2d([2, 5, 8]).T
    c3 = np.atleast_2d([3, 6, 9]).T
    C = np.hstack([c1, c2, c3])       # Join columns side by side
    print(C)
    print("Shape:", C.shape)
    return


if __name__ == "__main__":
    app.run()
