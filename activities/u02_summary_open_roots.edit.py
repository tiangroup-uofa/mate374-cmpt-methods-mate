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
        return x**2 - 2.0

    def dF(x):
        return 2.0 * x

    return F, dF


@app.cell
def _(F, dF, root_scalar):
    newton = root_scalar(F, x0=1.0, fprime=dF, method="newton")
    secant = root_scalar(F, x0=1.0, x1=2.0, method="secant")
    print("Newton:", newton.root, "converged:", newton.converged)
    print("Secant:", secant.root, "converged:", secant.converged)
    return


@app.cell
def _(F):
    # Compare with g(x) = 2/x. Both rearrange x² = 2.
    def g(x):
        return 0.5 * (x + 2.0 / x)

    x = 1.0
    tolerance = 1e-8
    for iteration in range(1, 21):
        x_next = g(x)
        print(f"{iteration}: x = {x_next:.10f}, residual = {F(x_next):.2e}")
        if abs(x_next - x) < tolerance and abs(F(x_next)) < tolerance:
            break
        x = x_next
    else:
        print("Tolerance not reached within 20 iterations.")
    return


if __name__ == "__main__":
    app.run()
