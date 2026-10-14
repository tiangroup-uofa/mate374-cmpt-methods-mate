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
    from scipy.optimize import minimize, root
    return minimize, np, root


@app.cell
def _(minimize, np, root):
    # Harmonic energy for two displacement coordinates, in arbitrary units.
    def energy(x):
        return 0.5*x[0]**2 + 2*x[1]**2

    def gradient(x):
        return np.array([x[0], 4*x[1]])

    def hessian(x):
        return np.array([[1.0, 0.0], [0.0, 4.0]])

    # Predict the minimum. Change the starting coordinates and run again.
    x0 = np.array([0.8, 0.45])
    minimum = minimize(energy, x0, jac=gradient, method="BFGS")
    stationary = root(gradient, x0, jac=hessian)

    print("Minimize: coordinates, energy, success")
    print(minimum.x, minimum.fun, minimum.success)
    print("Gradient at minimizer:", gradient(minimum.x))
    print("Root: coordinates, success")
    print(stationary.x, stationary.success)
    print("Gradient at root:", gradient(stationary.x))
    return


if __name__ == "__main__":
    app.run()
