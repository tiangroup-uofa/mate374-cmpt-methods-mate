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
    f = np.array([0.0, 1.0])

    K_weak = np.array([[5.001, -5.0],
                       [-5.0, 5.0]])
    K_equal = np.array([[10.0, -5.0],
                        [-5.0, 5.0]])

    u_weak = np.linalg.inv(K_weak) @ f
    u_equal = np.linalg.inv(K_equal) @ f
    u_weak, u_equal
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
