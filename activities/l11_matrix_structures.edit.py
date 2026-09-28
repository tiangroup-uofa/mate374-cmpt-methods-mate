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
    I = np.eye(3)                    # Ones on the diagonal
    D = np.diag([2, 3, 4])
    x = np.array([1, 2, 3])
    print("I:\n", I)
    print("D:\n", D)
    print("I @ x:", I @ x)
    print("D @ x:", D @ x)
    return


@app.cell
def _(np):
    A = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])
    print("Upper triangular:\n", np.triu(A))
    print("Lower triangular:\n", np.tril(A))
    return


@app.cell
def _(np):
    B = np.array([[2, 1],
                  [1, 3]])          # Symmetric
    print("B:\n", B)
    print("B.T:\n", B.T)
    return


if __name__ == "__main__":
    app.run()
