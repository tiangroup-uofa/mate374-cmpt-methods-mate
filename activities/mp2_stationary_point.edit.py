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
    import scipy as sp
    return (np, sp)


@app.cell
def _():
    def E(x):
        return x**4 - 3*x**2 + x

    def E_prime(x):
        return 4*x**3 - 6*x + 1

    def E_second(x):
        return 12*x**2 - 6
    return (E, E_prime, E_second)


@app.cell
def _(E, E_prime, E_second, sp):
    # Intended: a minimum of E, found as a root of E'(x) = 0
    solution = sp.optimize.root_scalar(E_prime, x0=0.0, fprime=E_second, method="newton")
    x_star = solution.root
    print("x* =", x_star, " converged:", solution.converged)
    print("E(x*) =", E(x_star))
    print("E''(x*) =", E_second(x_star))
    return


if __name__ == "__main__":
    app.run()
