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

    f_before = np.array([-1.0, 1.0])
    f_after = np.array([-1.0, 1.001])
    return K_equal, K_weak, f_after, f_before


@app.cell
def _(K_weak, f_after, f_before, np):
    u_weak_before = np.linalg.inv(K_weak) @ f_before
    u_weak_after = np.linalg.inv(K_weak) @ f_after
    u_weak_before, u_weak_after
    return


@app.cell
def _(K_equal, f_after, f_before, np):
    u_equal_before = np.linalg.inv(K_equal) @ f_before
    u_equal_after = np.linalg.inv(K_equal) @ f_after
    u_equal_before, u_equal_after
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
