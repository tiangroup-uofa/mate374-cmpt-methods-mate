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
    A_inv = np.linalg.inv(A)

    # Intended: condition number from its definition ||A|| ||A^-1||
    print("norm(A) * norm(inv(A)):", np.linalg.norm(A) * np.linalg.norm(A_inv))
    print("np.linalg.cond(A):     ", np.linalg.cond(A))
    return


if __name__ == "__main__":
    app.run()
