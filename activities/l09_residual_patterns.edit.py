# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "numpy>=2.0", "matplotlib>=3.9"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def imports():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from numpy.polynomial import Polynomial
    return Polynomial, mo, np, plt


@app.cell(hide_code=True)
def introduction(mo):
    mo.md(r"""
    ## What does a pattern in the residuals tell us?
    These synthetic data follow $y=1+0.4x+0.9x^2+1.2x^3+\mathrm{noise}$.
    **Predict:** what pattern will remain when a straight line is fitted to this cubic trend?
    Change the polynomial degree from 1 to 10. The data and axis scales stay fixed.

    The coefficients are listed in ascending powers of $x$: $[a_0,a_1,\ldots,a_n]$.
    """)
    return


@app.cell
def data(np):
    x = np.linspace(-2.0, 2.0, 31)
    noise = np.random.default_rng(18).normal(0.0, 0.20, size=x.size)
    y = 1.0 + 0.4*x + 0.9*x**2 + 1.2*x**3 + noise
    return x, y


@app.cell(hide_code=True)
def controls(mo):
    degree = mo.ui.number(1, 10, step=1, value=1, label="Polynomial degree")
    degree
    return (degree,)


@app.cell
def fit(Polynomial, degree, np, x, y):
    f_hat = Polynomial.fit(x, y, deg=degree.value)
    residual = y - f_hat(x)
    coefficients = f_hat.convert().coef
    sse = np.sum(residual**2)
    print("Coefficients in ascending powers of x:")
    for i, value in enumerate(coefficients):
        print(f"a{i} = {value:.5g}")
    print(f"SSE = {sse:.4f}")
    return coefficients, f_hat, residual, sse


@app.cell(hide_code=True)
def plotting_helper(Polynomial, np):
    def draw_fit(axes, x_data, y_data, model, residuals, fitted_degree):
        grid = np.linspace(x_data.min(), x_data.max(), 400)
        axes[0].scatter(x_data, y_data, s=25, color="#246b9e", label="Noisy cubic data", zorder=3)
        axes[0].plot(grid, model(grid), color="#29805c", lw=2, label=f"Degree {fitted_degree} fit")
        axes[0].set(xlabel="Input, x", ylabel="Response, y", title=f"Degree {fitted_degree}", ylim=(-10, 15))
        axes[0].legend(fontsize=9)
        axes[1].axhline(0, color="0.4", lw=1)
        axes[1].scatter(x_data, residuals, s=25, color="#c25b24", zorder=3)
        # Use the degree-one residual range for every degree.
        reference = y_data - Polynomial.fit(x_data, y_data, 1)(x_data)
        limit = 1.15 * np.max(np.abs(reference))
        axes[1].set(xlabel="Input, x", ylabel="Residual, y − f̂(x)",
                    title="Signed residuals", ylim=(-limit, limit))
        for ax in axes:
            ax.set_xlim(-2.15, 2.15)
            ax.grid(alpha=0.18)
            ax.spines[["top", "right"]].set_visible(False)
    return (draw_fit,)


@app.cell(hide_code=True)
def plot(degree, draw_fit, f_hat, plt, residual, x, y):
    figure, _axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    draw_fit(_axes, x, y, f_hat, residual, degree.value)
    figure
    return (figure,)


if __name__ == "__main__":
    app.run()
