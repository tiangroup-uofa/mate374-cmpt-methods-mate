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
    M = np.array([[ 1,  3, -9,  6, 4],
                  [ 2, -1,  6,  7, 1],
                  [ 3,  2, -3, 15, 5],
                  [ 8, -1,  1,  4, 2],
                  [11,  1, -2, 18, 7]], dtype=float)
    x_true = np.array([0.0, 1.0, 1.0, 0.0, 1.0])
    b = M @ x_true
    return M, b


@app.cell
def _(M, b, np):
    M32 = M.astype(np.float32)
    b32 = b.astype(np.float32)
    x32_inv = np.linalg.inv(M32) @ b32
    x32_inv, M32 @ x32_inv - b32
    return (M32,)


@app.cell
def _(M, M32, np):
    np.abs(np.linalg.inv(M)).max(), np.abs(np.linalg.inv(M32)).max()
    return


@app.cell
def _(np):
    np.spacing(np.float64(7e15)), np.spacing(np.float32(7e15))
    return


if __name__ == "__main__":
    app.run()
