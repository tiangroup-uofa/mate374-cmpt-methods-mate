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
    np.linalg.cond(K_weak), np.linalg.cond(K_equal)
    return K_equal, K_weak


@app.cell
def _(K_equal, K_weak, np):
    f = np.array([-1.0, 1.0])
    delta_f = np.array([0.0, 0.001])
    relative_force_change = np.linalg.norm(delta_f) / np.linalg.norm(f)

    bound_weak = np.linalg.cond(K_weak) * relative_force_change
    bound_equal = np.linalg.cond(K_equal) * relative_force_change
    relative_force_change, bound_weak, bound_equal
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
