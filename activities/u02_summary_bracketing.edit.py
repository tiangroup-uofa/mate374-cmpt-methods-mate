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
    # Change P and choose endpoints that bracket the new root.
    P = 2.0

    def f(x):
        return x**2

    def F(x):
        return f(x) - P

    result = root_scalar(F, bracket=(1.0, 2.0), method="bisect")
    print("Root:", result.root)
    print("Converged:", result.converged)
    print("Residual:", F(result.root))
    return


if __name__ == "__main__":
    app.run()
