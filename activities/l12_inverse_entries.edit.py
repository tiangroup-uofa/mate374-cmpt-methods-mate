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
    K_weak = np.array([[5.001, -5.0],
                       [-5.0, 5.0]])
    K_equal = np.array([[10.0, -5.0],
                        [-5.0, 5.0]])
    return K_equal, K_weak


@app.cell
def _(K_weak, np):
    np.linalg.inv(K_weak)
    return


@app.cell
def _(K_equal, np):
    np.linalg.inv(K_equal)
    return


if __name__ == "__main__":
    app.run()
