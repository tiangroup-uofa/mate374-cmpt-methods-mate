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
    scalar = np.float64(2.0)
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    column = a.reshape(-1, 1)
    row = a.reshape(1, -1)
    print("Scalar:", scalar, "shape:", scalar.shape)
    print("a:", a, "shape:", a.shape)
    print("a.T:", a.T, "shape:", a.T.shape)
    print("Column:\n", column, "shape:", column.shape)
    print("Row:\n", row, "shape:", row.shape)
    print("Transposed column:\n", column.T, "shape:", column.T.shape)
    return a, b, column, row, scalar


@app.cell
def _(a, b, column, np, row, scalar):
    print("a + b:", a + b)
    print("scalar * a:", scalar * a)
    print("Entrywise product a * b:", a * b)
    print("Inner product a @ b:", a @ b)
    print("Inner product np.dot(a, b):", np.dot(a, b))
    print("Length of a:", np.linalg.norm(a))
    print("row @ column:", row @ column, "shape:", (row @ column).shape)
    print("column @ row:\n", column @ row, "shape:", (column @ row).shape)
    return


@app.cell
def _(np):
    A = np.array([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])
    x = np.array([1.0, 0.0, -1.0])
    B = np.array([[1.0, 0.0],
                  [0.0, 1.0],
                  [1.0, 1.0]])
    print("A shape:", A.shape, "B shape:", B.shape)
    print("Entry in first row, second column:", A[0, 1])
    print("Transpose A.T:\n", A.T, "shape:", A.T.shape)
    print("Matrix-vector product A @ x:", A @ x, "shape:", (A @ x).shape)
    print("Entrywise product A * x:\n", A * x, "shape:", (A * x).shape)
    print("A @ B:\n", A @ B, "shape:", (A @ B).shape)
    print("B @ A:\n", B @ A, "shape:", (B @ A).shape)
    return


@app.cell
def _(np):
    I = np.eye(3)
    D = np.diag([2.0, 3.0, 4.0])
    C = np.array([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0],
                  [7.0, 8.0, 9.0]])
    print("Identity:\n", I)
    print("Diagonal:\n", D)
    print("Upper triangular part:\n", np.triu(C))
    print("Lower triangular part:\n", np.tril(C))
    print("I @ C:\n", I @ C)
    return


if __name__ == "__main__":
    app.run()
