# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "numpy>=2.0", "scipy>=1.12", "matplotlib>=3.9"]
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
    from scipy import integrate

    return Polynomial, integrate, mo, np, plt


@app.cell
def _(mo):
    n_slider = mo.ui.slider(steps=[6, 11, 21, 41, 81, 161], value=21, label="points", show_value=True)
    sigma_slider = mo.ui.slider(steps=[0.0, 0.05, 0.1, 0.2, 0.5, 1.0], value=0.1, label="noise σ", show_value=True)
    return n_slider, sigma_slider


@app.cell
def _(Polynomial, integrate, np):
    EXACT = 20.0          # integral of x³ from 1 to 3


    def experiment(n, sigma, rng):
        """One noisy data set: percentage errors of the data, the derivative, and three integrals."""
        x = np.linspace(1.0, 3.0, n)
        y_true = x**3
        y = y_true + rng.normal(0.0, sigma, n)
        dy = np.gradient(y, x[1] - x[0], edge_order=2)
        P = Polynomial.fit(x, y, 3).integ()          # fit a cubic, then integrate it
        return {
            "data": 100 * np.linalg.norm(y - y_true) / np.linalg.norm(y_true),
            "central derivative": 100 * np.linalg.norm(dy - 3 * x**2) / np.linalg.norm(3 * x**2),
            "trapezoid integral": 100 * abs(np.trapezoid(y, x) - EXACT) / EXACT,
            "Simpson integral": 100 * abs(integrate.simpson(y, x=x) - EXACT) / EXACT,
            "cubic-fit integral": 100 * abs(P(3.0) - P(1.0) - EXACT) / EXACT,
        }, x, y


    def typical(n, sigma, repeats=300):
        """Root-mean-square percentage error over many noise draws."""
        rng = np.random.default_rng(0)
        runs = [experiment(n, sigma, rng)[0] for _ in range(repeats)]
        return {k: np.sqrt(np.mean([r[k] ** 2 for r in runs])) for k in runs[0]}

    return experiment, typical


@app.cell
def _(experiment, mo, n_slider, np, plt, sigma_slider, typical):
    one, x, y = experiment(n_slider.value, sigma_slider.value, np.random.default_rng(374))
    rms = typical(n_slider.value, sigma_slider.value)

    _lines = [f"| h = {x[1] - x[0]:.4f} | this data set | typical (300 data sets) |", "|---|---|---|"]
    for _k in one:
        _lines.append(f"| {_k} | {one[_k]:.3f} % | {rms[_k]:.3f} % |")
    _table = mo.md("\n".join(_lines))

    _ns = [6, 11, 21, 41, 81, 161]
    _curves = [typical(_n, sigma_slider.value, repeats=150) for _n in _ns]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0), layout="constrained")
    _xf = np.linspace(1, 3, 200)
    axes[0].fill_between(x, 0, y, color="C2", alpha=0.25, lw=0)
    axes[0].plot(_xf, _xf**3, color="0.6", label="exact  x³")
    axes[0].plot(x, y, "o", ms=3.5, color="k", label="noisy data")
    axes[0].set(xlabel="x", ylabel="y", title="Data and the trapezoidal area")
    for _k, _c in [("central derivative", "C3"), ("data", "0.5"), ("trapezoid integral", "C2"), ("Simpson integral", "C4")]:
        axes[1].loglog(_ns, [_d[_k] for _d in _curves], "o-", ms=4, color=_c, label=_k)
    axes[1].axvline(n_slider.value, color="0.7", lw=0.8, ls=":")
    axes[1].set(xlabel="number of points", ylabel="typical error (%)", title=f"σ = {sigma_slider.value}")
    for _ax in axes:
        _ax.grid(alpha=0.25, which="both")
        _ax.legend(fontsize=8)

    mo.vstack(
        [
            mo.hstack([mo.vstack([n_slider, sigma_slider]), _table], justify="start", align="center", gap=2, widths=[2, 3]),
            fig,
        ]
    )
    return


if __name__ == "__main__":
    app.run()
