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
                  [4, 5, 6]])
    x = np.array([1, 0, -1])
    print("A @ x:", A @ x)
    return


if __name__ == "__main__":
    app.run()
