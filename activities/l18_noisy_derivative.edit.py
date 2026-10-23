# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "numpy>=2.0", "scipy>=1.13", "matplotlib>=3.9"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from numpy.polynomial import Polynomial
    from scipy.interpolate import CubicSpline

    return CubicSpline, Polynomial, mo, np, plt


@app.cell
def _(mo):
    n_slider = mo.ui.slider(steps=[6, 11, 21, 41, 81], value=21, label="points", show_value=True)
    sigma_slider = mo.ui.slider(steps=[0.0, 0.05, 0.1, 0.2, 0.5], value=0.1, label="noise σ", show_value=True)
    return n_slider, sigma_slider


@app.cell
def _(CubicSpline, Polynomial, n_slider, np, sigma_slider):
    rng = np.random.default_rng(374)
    x = np.linspace(1.0, 3.0, n_slider.value)
    h = x[1] - x[0]
    y_true = x**3
    y = y_true + rng.normal(0.0, sigma_slider.value, x.size)
    dy_true = 3 * x**2

    dy_central = np.gradient(y, h, edge_order=2)
    dy_spline = CubicSpline(x, y)(x, 1)        # interpolate, then differentiate

    p = Polynomial.fit(x, y, 3)                # fit a cubic, then differentiate
    dy_fit = p.deriv()(x)


    def percent(error, reference):
        return 100 * np.linalg.norm(error) / np.linalg.norm(reference)


    summary = f"""
    | spacing $h$ = {h:.3f} | percentage error |
    |---|---|
    | data | {percent(y - y_true, y_true):.2f} % |
    | central derivative | {percent(dy_central - dy_true, dy_true):.2f} % |
    | cubic-spline derivative | {percent(dy_spline - dy_true, dy_true):.2f} % |
    | cubic-fit derivative | {percent(dy_fit - dy_true, dy_true):.2f} % |
    """
    return dy_central, dy_fit, dy_spline, p, summary, x, y


@app.cell
def _(dy_central, dy_fit, dy_spline, mo, n_slider, np, p, plt, sigma_slider, summary, x, y):
    x_fine = np.linspace(1.0, 3.0, 300)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), layout="constrained", width_ratios=[1, 2])
    axes[0].plot(x_fine, x_fine**3, color="0.6", label="exact  x³")
    axes[0].plot(x, y, "o", ms=4, color="k", label="noisy data")
    axes[0].plot(x_fine, p(x_fine), "--", color="C2", label="cubic fit")
    axes[0].set(xlabel="x", ylabel="y")
    axes[1].plot(x_fine, 3 * x_fine**2, color="0.6", label="exact  3x²")
    axes[1].plot(x, dy_central, "o-", ms=4, lw=1.0, color="C3", label="central difference")
    axes[1].plot(x, dy_spline, "s:", ms=3, lw=0.8, color="C4", label="cubic spline")
    axes[1].plot(x_fine, p.deriv()(x_fine), "--", lw=1.8, color="C2", label="cubic fit")
    _lo = min(dy_central.min(), dy_spline.min(), 3.0)
    _hi = max(dy_central.max(), dy_spline.max(), 27.0)
    _pad = 0.05 * (_hi - _lo)
    axes[1].set(xlabel="x", ylabel="dy/dx", ylim=(_lo - _pad, _hi + _pad), title="Derivative")
    axes[0].set(title="Data")
    for _ax in axes:
        _ax.grid(alpha=0.25)
        _ax.legend(fontsize=8)
    mo.vstack(
        [
            mo.hstack([mo.vstack([n_slider, sigma_slider]), mo.md(summary)], justify="start", align="center", gap=2, widths=[2, 3]),
            fig,
        ]
    )
    return



if __name__ == "__main__":
    app.run()
