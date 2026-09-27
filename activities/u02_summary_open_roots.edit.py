# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "scipy"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    from scipy.optimize import root_scalar
    return (root_scalar,)


@app.cell
def _():
    def F(x):
        """Residual function to calculate sqrt(2)."""
        return x**2 - 2.0

    def dF(x):
        """1st derivative of F(x)"""
        return 2.0 * x

    return F, dF


@app.cell
def _(F, dF, root_scalar):
    # x0: initial guess; fprime optionally supplies the derivative.
    newton = root_scalar(F, x0=1.0, fprime=dF, method="newton")
    print("Newton:", newton.root, "converged:", newton.converged)
    return


if __name__ == "__main__":
    app.run()
