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
    from scipy.optimize import root_scalar, minimize_scalar
    return minimize_scalar, np, root_scalar


@app.cell
def _():
    def f(x):
        """A quadratic function centered at x=2.0"""
        return (x - 2.0)**2 - 1.0

    def df(x):
        """1st derivative of f"""
        return 2.0 * (x - 2.0)

    def d2f(x):
        """2nd derivative of f"""
        return 2.0

    return d2f, df, f


@app.cell
def _(d2f, df, f, minimize_scalar, np, root_scalar):
    # Naive grid search to find minimum
    x_grid = np.linspace(0.0, 4.0, 1001)
    x_grid_min = x_grid[np.argmin(f(x_grid))]
    print("Grid minimum:", x_grid_min)

    # Use minimize_scalar (bounded methods)
    minimum = minimize_scalar(f, bounds=(0.0, 4.0), method="bounded")
    print("Bounded minimum:", minimum.x, "success:", minimum.success)

    # Newton-Raphson stationary point search using 1st derivative
    stationary = root_scalar(df, x0=3.5, fprime=d2f, method="newton")

    print("Newton-Raphson stationary point:", stationary.root, "converged:", stationary.converged)
    print("Second derivative there:", d2f(stationary.root))
    return


if __name__ == "__main__":
    app.run()
