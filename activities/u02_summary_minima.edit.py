# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "numpy>=2.0", "scipy"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    from scipy.optimize import root_scalar, minimize_scalar, minimize
    return minimize, minimize_scalar, np, root_scalar


@app.cell
def _():
    def f(x):
        return (x - 2.0)**2 - 1.0

    def df(x):
        return 2.0 * (x - 2.0)

    def d2f(x):
        return 2.0

    return d2f, df, f


@app.cell
def _(d2f, df, f, minimize, minimize_scalar, np, root_scalar):
    x_grid = np.linspace(0.0, 4.0, 1001)
    x_grid_min = x_grid[np.argmin(f(x_grid))]
    minimum = minimize_scalar(f, bounds=(0.0, 4.0), method="bounded")
    stationary = root_scalar(df, x0=3.5, fprime=d2f, method="newton")
    local_minimum = minimize(lambda x: f(x[0]), x0=[3.5], method="BFGS")

    print("Grid minimum:", x_grid_min)
    print("Bounded minimum:", minimum.x, "success:", minimum.success)
    print("Stationary point:", stationary.root, "converged:", stationary.converged)
    print("Second derivative there:", d2f(stationary.root))
    print("General optimizer:", local_minimum.x, "success:", local_minimum.success)
    return


if __name__ == "__main__":
    app.run()
