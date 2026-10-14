# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "numpy", "scipy"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    from scipy.optimize import root
    return np, root


@app.cell
def _(np, root):
    def F(x):
        x1, x2 = x
        return np.array([x1**2 + x2**2 - 5, x1 - x2 - 1])

    def J(x):
        x1, x2 = x
        return np.array([[2*x1, 2*x2], [1.0, -1.0]])

    # Predict which intersection is reached. Try x0 = [-1.5, -2.5].
    x0 = np.array([1.5, 0.5])
    result = root(F, x0, jac=J, method="hybr")
    # Try omitting jac=J to let hybr estimate the Jacobian.
    print("Root:", result.x)
    print("Success:", result.success)
    print("Residual:", F(result.x))
    return


if __name__ == "__main__":
    app.run()
