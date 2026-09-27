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
def _(root_scalar):
    # The target to be reached so f(x) = P
    P = 2.0

    def f(x):
        """Normal function"""
        return x**2

    def F(x):
        """The residual function"""
        return f(x) - P

    result = root_scalar(F, bracket=(1.0, 2.0), method="bisect")
    # Use result.root and result.converged for the final result!
    print("Root:", result.root)
    print("Converged:", result.converged)
    print("Residual:", F(result.root))
    return


if __name__ == "__main__":
    app.run()
