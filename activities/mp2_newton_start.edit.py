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
    T = 300.0   # K
    P = 50.0    # atm

    def F(V):
        return 0.5 * T / (V + 0.1) - 5 / V**2 - P

    def F_prime(V):
        return -0.5 * T / (V + 0.1)**2 + 10 / V**3
    return (F, F_prime)


@app.cell
def _(F, F_prime, sp):
    # Intended: the gas volume in L at 50 atm
    solution = sp.optimize.root_scalar(F, x0=10.0, fprime=F_prime, method="newton")
    print("V =", solution.root)
    print(solution)
    return


@app.cell
def _(F, F_prime, sp):
    # A second start: does this root make sense for a gas at 50 atm?
    other = sp.optimize.root_scalar(F, x0=0.1, fprime=F_prime, method="newton")
    print("V =", other.root, " converged:", other.converged)
    return


if __name__ == "__main__":
    app.run()
